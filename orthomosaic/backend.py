"""Compute backends: Cython (CPU), CUDA (CuPy raw kernels) and Apple GPU (Metal via MLX).

Both expose the same small interface used by the pipeline:

    match(d1, d2)                         -> (idx, best, second)   numpy int32
    match_mutual(d1, d2)                  -> (idx12, best12, second12, idx21)
    new_block(H, W)                       -> accumulator handle
    upload(img_uint8_rgb)                 -> device image
    warp_accumulate(block, dev_img, M, gain, power, mode)
    finalize(block, mode)                 -> numpy uint8 RGBA
    finalize_float(block, mode)           -> (numpy float32 (H,W,3) blended channels, covered (H,W) bool)
    xp                                    -> array module (numpy / cupy) for dense 3D
    sample_view(img, cam, X0, Y0, gsd, Z) -> (samples (H,W,ch) float32, valid (H,W))
    to_numpy(a)                           -> numpy array
    box(a, r)                             -> (2r+1)^2 window mean (dense matching)
    asarray(a, dtype)                     -> device array
"""
from __future__ import annotations

import logging
import os

import numpy as np

from . import _core, _mvs

log = logging.getLogger(__name__)

MODE_FEATHER = 0
MODE_MAX = 1


class CPUBackend:
    name = "cpu"
    parallel_blocks = True
    xp = np

    def match(self, d1: np.ndarray, d2: np.ndarray):
        return _core.match_hamming(_as_u64(d1), _as_u64(d2))

    def match_mutual(self, d1: np.ndarray, d2: np.ndarray):
        """-> (idx12, best12, second12, idx21)"""
        return _core.match_hamming_mutual(_as_u64(d1), _as_u64(d2))

    def new_block(self, H: int, W: int):
        return np.zeros((H, W, 3), np.float32), np.zeros((H, W), np.float32)

    def upload(self, img: np.ndarray):
        return np.ascontiguousarray(img)

    def image_nbytes(self, dev_img) -> int:
        return dev_img.nbytes

    def warp_accumulate(self, block, dev_img, M, gain, power, mode):
        acc, wsum = block
        _core.warp_accumulate(acc, wsum, dev_img, np.ascontiguousarray(M, np.float64).ravel(),
                              np.ascontiguousarray(gain, np.float32), float(power), int(mode))

    def finalize(self, block, mode):
        acc, wsum = block
        return _core.finalize(acc, wsum, int(mode))

    def finalize_float(self, block, mode):
        acc, wsum = block
        return (acc / np.maximum(wsum, 1e-12)[..., None] if mode == 0 else acc.copy()), wsum > 0

    def sample_view(self, img, cam, X0, Y0, gsd, Z):
        """Z: (H, W) heights, or a (D, H, W) stack of height hypotheses."""
        R, C, f, k1, k2, k3, cx, cy = cam
        Z = np.ascontiguousarray(Z, np.float32)
        out = np.zeros(Z.shape + (img.shape[2],), np.float32)
        valid = np.zeros(Z.shape, np.uint8)
        Zs, os_, vs = Z.reshape((-1,) + Z.shape[-2:]), out.reshape((-1,) + out.shape[-3:]), valid.reshape((-1,) + Z.shape[-2:])
        for d in range(Zs.shape[0]):
            _mvs.sample_view(img, R, C, float(f), float(k1), float(k2), float(k3), float(cx), float(cy),
                             float(X0), float(Y0), float(gsd), Zs[d], os_[d], vs[d])
        return out, valid

    def to_numpy(self, a):
        return np.asarray(a)

    def box(self, a, r):
        return _box_cumsum(np, a, r)

    def asarray(self, a, dtype=None):
        return np.asarray(a, dtype)

    def argmin0(self, a):
        idx = np.argmin(a, axis=0)
        return idx, np.take_along_axis(a, idx[None], axis=0)[0]

    def argmax0(self, a):
        idx = np.argmax(a, axis=0)
        return idx, np.take_along_axis(a, idx[None], axis=0)[0]

    def topk_mean(self, st, k):
        """Mean of the k largest values along axis 0."""
        n = st.shape[0]
        return np.partition(st, n - k, axis=0)[n - k:].mean(axis=0)


def _box_cumsum(xp, a, r):
    """Mean over a (2r+1)^2 window (zero padding) via float64 summed-area tables."""
    if xp is np and isinstance(a, np.ndarray) and a.ndim >= 2 and a.dtype in (np.float32, np.float64):
        from . import _fast
        shape = a.shape
        batch = int(np.prod(shape[:-2]))
        planes = np.ascontiguousarray(a).reshape((batch,) + shape[-2:])
        return _fast.box_sat_mean(planes, r).reshape(shape)
    k = 2 * r + 1
    p = xp.pad(a.astype(xp.float64), [(0, 0)] * (a.ndim - 2) + [(r + 1, r), (r + 1, r)])
    c = xp.cumsum(xp.cumsum(p, axis=-2), axis=-1)
    s = c[..., k:, k:] - c[..., :-k, k:] - c[..., k:, :-k] + c[..., :-k, :-k]
    return (s / (k * k)).astype(xp.float32)


def _as_u64(d: np.ndarray) -> np.ndarray:
    d = np.ascontiguousarray(d, np.uint8)
    if d.ndim != 2 or d.shape[1] % 8:
        raise ValueError("descriptors must be a 2D array with a byte width divisible by 8")
    return d.view(np.uint64)


_CUDA_HELP = ("Install the CUDA headers CuPy needs to compile kernels: "
              "pip install \"pyOrthomosaic[cuda]\"  (or: pip install \"cupy-cuda12x[ctk]\"), "
              "or install the CUDA Toolkit and set CUDA_PATH to it.")


def _prepare_cuda_env():
    """CuPy compiles kernels at runtime and needs the CUDA headers. If CUDA_PATH is not
    set, point it at an installed toolkit (the pip [ctk] headers are found by CuPy itself)."""
    if os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME"):
        return
    import glob
    candidates = ["/usr/local/cuda", "/opt/cuda"]
    candidates += sorted(glob.glob("/usr/local/cuda-*"), reverse=True)
    candidates += sorted(glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*"), reverse=True)
    for c in candidates:
        if os.path.isfile(os.path.join(c, "include", "cuda_runtime.h")):
            os.environ["CUDA_PATH"] = c
            log.debug("CUDA_PATH set to %s", c)
            return


def cuda_available() -> bool:
    if os.environ.get("ORTHO_DISABLE_CUDA") == "1":
        return False
    try:
        _prepare_cuda_env()
        import cupy  # noqa: F401
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def mps_available() -> bool:
    """Apple-silicon GPU usable through MLX (pip install "pyOrthomosaic[mps]")."""
    if os.environ.get("ORTHO_DISABLE_MPS") == "1":
        return False
    import platform
    import sys
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core as mx
        return bool(mx.metal.is_available())
    except Exception:
        return False


def _select_mps(prefer):
    try:
        from ._mlx import MLXBackend
        be = MLXBackend()
        log.info("Using Apple GPU backend: %s", be.device_name)
        return be
    except Exception as exc:
        msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        if prefer in ("mps", "metal"):
            raise RuntimeError(f"Apple GPU (MPS) backend unusable: {msg}") from exc
        log.warning("Apple GPU present but unusable (%s); falling back to CPU", msg)
        return None


def release_memory():
    """Hand freed memory back to the system: Python cycles, CuPy's device and pinned-host pools
    (kept for reuse until freed) and MLX's buffer cache (Apple silicon: shared with the CPU).
    Only touches GPU libraries that are already imported; safe to call at any time."""
    import gc
    import sys
    gc.collect()
    cp = sys.modules.get("cupy")
    if cp is not None:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    mx = sys.modules.get("mlx.core")
    if mx is not None:
        try:
            (getattr(mx, "clear_cache", None) or mx.metal.clear_cache)()
        except Exception:
            pass


def select_backend(prefer: str = "auto"):
    """prefer: 'auto' (CUDA > Apple GPU > CPU) | 'cuda' | 'mps' | 'cpu'."""
    prefer = prefer.lower()
    if prefer in ("mps", "metal"):
        if not mps_available():
            raise RuntimeError("MPS backend requested but no Apple-silicon GPU / MLX found. "
                               'Install it with: pip install "pyOrthomosaic[mps]"')
        return _select_mps(prefer)
    if prefer in ("auto", "cuda", "gpu"):
        if cuda_available():
            try:
                from ._cuda import CUDABackend
                be = CUDABackend()
                log.info("Using CUDA backend: %s", be.device_name)
                return be
            except Exception as exc:  # compile/driver problems -> fall back
                msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                if "header" in msg.lower() or "nvrtc" in msg.lower() or "CUDA_PATH" in msg:
                    msg += ". " + _CUDA_HELP
                if prefer != "auto":
                    raise RuntimeError(f"CUDA backend unusable: {msg}") from exc
                log.warning("CUDA present but unusable (%s); falling back to CPU", msg)
        elif prefer != "auto":
            raise RuntimeError("CUDA backend requested but no CUDA device / CuPy found")
    if prefer == "auto" and mps_available():
        be = _select_mps(prefer)
        if be is not None:
            return be
    log.info("Using CPU backend (Cython, %d threads)", os.cpu_count() or 1)
    return CPUBackend()
