import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np

try:
    import orthomosaic._dense  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orthomosaic import _gpu_depth, densify  # noqa: E402
from orthomosaic._gpu_depth import GPUDepthSweep, cpu_sweep, same_bits, sweep_case  # noqa: E402
from orthomosaic.backend import cuda_available  # noqa: E402

# The CUDA kernels compiled as C++ for the CPU: every "thread" is a call of the kernel function with
# blockIdx / threadIdx set, so the kernels' arithmetic and indexing are checked without a GPU.
_SHIM = r"""
#include <cmath>
using std::isnan; using std::isfinite; using std::sqrt;
struct om_dim3 { unsigned x, y, z; };
static om_dim3 blockIdx, threadIdx, blockDim;
#define __global__
%(src)s
#define OM_LAUNCH(CALL) \
    blockDim.x = bx; blockDim.y = by; blockDim.z = bz; \
    for (unsigned z = 0; z < (unsigned)gz; z++) for (unsigned y = 0; y < (unsigned)gy; y++) \
    for (unsigned x = 0; x < (unsigned)gx; x++) for (unsigned tz = 0; tz < (unsigned)bz; tz++) \
    for (unsigned ty = 0; ty < (unsigned)by; ty++) for (unsigned tx = 0; tx < (unsigned)bx; tx++) { \
        blockIdx.x = x; blockIdx.y = y; blockIdx.z = z; threadIdx.x = tx; threadIdx.y = ty; threadIdx.z = tz; CALL; }
#define F(i) ((float*)a[i])
#define I(i) (*(int*)a[i])
extern "C" void L_om_dw_warp(int gx, int gy, int gz, int bx, int by, int bz, void** a) {
    OM_LAUNCH(om_dw_warp(F(0), F(1), I(2), *(float*)a[3], *(float*)a[4], F(5), I(6), I(7), F(8),
                         (long long*)a[9], (int*)a[10], (int*)a[11], (double*)a[12], F(13), F(14)))
}
extern "C" void L_om_dw_hsum(int gx, int gy, int gz, int bx, int by, int bz, void** a) {
    OM_LAUNCH(om_dw_hsum(F(0), F(1), F(2), I(3), I(4), I(5), F(6)))
}
extern "C" void L_om_dw_vsum(int gx, int gy, int gz, int bx, int by, int bz, void** a) {
    OM_LAUNCH(om_dw_vsum(F(0), F(1), F(2), I(3), I(4), I(5), *(double*)a[6], F(7)))
}
extern "C" void L_om_dw_combine(int gx, int gy, int gz, int bx, int by, int bz, void** a) {
    OM_LAUNCH(om_dw_combine(F(0), I(1), *(long long*)a[2], I(3), I(4), F(5), F(6), F(7), F(8), (int*)a[9]))
}
"""
_SCALARS = {np.int32: ctypes.c_int32, np.int64: ctypes.c_int64, np.float32: ctypes.c_float, np.float64: ctypes.c_double}


def _compiler():
    if os.name == "nt":                      # the shim is built as a POSIX shared library
        return None
    return next((c for c in ("g++", "c++", "clang++") if shutil.which(c)), None)


class _Kernel:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, grid, block, args, shared_mem=0):
        grid, block = tuple(grid) + (1,) * (3 - len(grid)), tuple(block) + (1,) * (3 - len(block))
        keep = []
        ptrs = (ctypes.c_void_p * len(args))()
        for j, v in enumerate(args):
            if isinstance(v, np.ndarray):
                assert v.flags.c_contiguous
                ptrs[j] = v.ctypes.data
            else:
                c = _SCALARS[type(v)](v)
                keep.append(c)
                ptrs[j] = ctypes.addressof(c)
        self.fn(*grid, *block, ptrs)


class _Module:
    _cache = {}

    def __init__(self, code, options=()):
        key = hashlib.sha256(code.encode()).hexdigest()[:16]
        if key not in self._cache:
            d = tempfile.mkdtemp(prefix="om_cuda_cpu_")
            src, lib = os.path.join(d, "k.cpp"), os.path.join(d, "k.so")
            with open(src, "w") as fh:
                fh.write(_SHIM % {"src": code})
            subprocess.run([_compiler(), "-O2", "-std=c++11", "-ffp-contract=off", "-fPIC", "-shared", src, "-o", lib],
                           check=True)
            self._cache[key] = ctypes.CDLL(lib)
        self.lib = self._cache[key]

    def get_function(self, name):
        fn = getattr(self.lib, "L_" + name)
        fn.argtypes = [ctypes.c_int] * 6 + [ctypes.POINTER(ctypes.c_void_p)]
        return _Kernel(fn)


class _Stream:
    def __init__(self, non_blocking=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def cpu_cupy():
    """A stand-in for the parts of CuPy that GPUDepthSweep uses: numpy arrays, kernels run on the CPU."""
    cp = types.SimpleNamespace()
    cp.RawModule = _Module
    cp.asarray = lambda a, dtype=None: np.array(a, dtype=dtype, copy=True)
    cp.asnumpy = lambda a: np.array(a, copy=True)
    cp.empty, cp.full, cp.zeros = np.empty, np.full, np.zeros

    class OutOfMemoryError(MemoryError):
        pass

    cp.cuda = types.SimpleNamespace(Stream=_Stream,
                                    runtime=types.SimpleNamespace(memGetInfo=lambda: (8 << 30, 16 << 30)),
                                    memory=types.SimpleNamespace(OutOfMemoryError=OutOfMemoryError))
    return cp


def _sweepers():
    # The kernels reproduce the x86-64 CPU build, which has no FMA instructions. ARM builds of
    # _sweep.h may fuse multiply-adds, so there the CPU result is not the reference; on such a
    # machine CUDABackend's self-test fails and the depth maps stay on the CPU.
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return []
    out = []
    if _compiler():
        out.append(("cpu-compiled CUDA source", GPUDepthSweep(cpu_cupy())))
    if cuda_available():
        import cupy
        out.append(("CUDA device", GPUDepthSweep(cupy)))
    return out


def test_gpu_sweep_bit_identical_to_cpu_kernel():
    cases = [(37, 53, 4, 5, 2, 9), (64, 48, 1, 3, 2, 5), (11, 9, 3, 5, 2, 4), (5, 4, 2, 5, 1, 3),
             (30, 31, 5, 2, 3, 6), (2, 40, 2, 1, 2, 3), (40, 3, 4, 4, 4, 3), (70, 90, 16, 5, 2, 4)]
    sweepers = _sweepers()
    if not sweepers:
        print("skipped: needs x86-64 and a C++ compiler or a CUDA device")
        return
    for name, gpu in sweepers:
        rng = np.random.default_rng(7)
        for H, W, n, r, top_k, nh in cases:
            case = sweep_case(rng, H, W, n, r, nh)
            got = gpu.sweep(*case, r, top_k)
            want = cpu_sweep(*case, r, top_k)
            assert got is not None, name
            assert all(same_bits(g, w) for g, w in zip(got, want)), (name, H, W, n, r)
        gpu.self_test()
        print("ok", name)


def test_gpu_hypothesis_families_match_cpu():
    """The linear and refine hypotheses the GPU makes itself == _fast's maps."""
    sweepers = _sweepers()
    if not sweepers:
        return
    rng = np.random.default_rng(5)
    H, W = 33, 47
    ref, mu, sd, srcs, ray, G, _ = sweep_case(rng, H, W, 4, 5, 3)
    base = rng.uniform(0.02, 0.2, (H, W)).astype(np.float32)
    step = rng.uniform(1e-4, 5e-3, (H, W)).astype(np.float32)
    iu = rng.uniform(0.02, 0.2, (H, W)).astype(np.float32)
    iu[3, 4] = np.nan
    ds = rng.uniform(-0.05, 5e-3, (H, W)).astype(np.float32)       # some refinements fall below 1e-6
    fams = [densify._Hypotheses(lambda i: densify._fast.hyp_linear(base, step, i), 11, ("linear", base, step)),
            densify._Hypotheses(lambda q: densify._fast.hyp_refine(iu, ds, q - 3), 7, ("refine", iu, ds, 3))]
    for name, gpu in sweepers:
        for hyps in fams:
            got = gpu.sweep(ref, mu, sd, srcs, ray, G, hyps, 5, 2)
            want = cpu_sweep(ref, mu, sd, srcs, ray, G, hyps, 5, 2)
            assert all(same_bits(g, w) for g, w in zip(got, want)), (name, hyps.spec[0])
        print("ok", name, "hypothesis families")


def test_densify_sweep_same_depth_on_gpu():
    """densify._sweep with opt.gpu == without."""
    sweepers = _sweepers()
    if not sweepers:
        return
    rng = np.random.default_rng(3)
    H, W = 45, 61
    ref, _, _, srcs, ray, G, hyps = sweep_case(rng, H, W, 4, 5, 19)

    class Cam:
        def __init__(self, R, C, par):
            self.R, self.C = R, C
            self.f, self.k1, self.k2, self.k3, self.cx, self.cy = par

    cam_r = Cam(np.eye(3), np.zeros(3), [W, 0.0, 0.0, 0.0, W / 2, H / 2])
    cams_s = [Cam(g[:9].reshape(3, 3), -g[:9].reshape(3, 3).T @ g[9:12], list(g[12:])) for g in G]
    for name, gpu in sweepers:
        class Backend:
            depth_sweep = gpu.sweep
        cpu_opt, gpu_opt = densify.DepthOptions(), densify.DepthOptions(gpu=Backend())
        a = densify._sweep(ref, cam_r, srcs, cams_s, densify._Hypotheses(lambda i: hyps[i].copy(), len(hyps)),
                           cpu_opt, ray)
        b = densify._sweep(ref, cam_r, srcs, cams_s, densify._Hypotheses(lambda i: hyps[i].copy(), len(hyps)),
                           gpu_opt, ray)
        assert all(same_bits(x, y) for x, y in zip(a, b)), name
        print("ok", name, "densify._sweep")


def test_gpu_sweep_memory_budget_falls_back():
    if not _compiler():
        return
    gpu = GPUDepthSweep(cpu_cupy())
    gpu.budget = 1000                        # nothing fits: the caller runs the CPU loop
    case = sweep_case(np.random.default_rng(1), 20, 20, 2, 2, 3)
    assert gpu.sweep(*case, 2, 2) is None and gpu.cpu_fallbacks == 1
    assert _gpu_depth.device_bytes(20, 20, 800, 2) > 1000


if __name__ == "__main__":
    for t in (test_gpu_sweep_bit_identical_to_cpu_kernel, test_gpu_hypothesis_families_match_cpu,
              test_densify_sweep_same_depth_on_gpu, test_gpu_sweep_memory_budget_falls_back):
        t()
        print("ok", t.__name__)
