"""CUDA backend. Kernels are compiled at runtime by NVRTC through CuPy,
so no nvcc / toolkit build step is required -- only a driver + cupy wheel."""
from __future__ import annotations

import logging

import cupy as cp
import numpy as np

log = logging.getLogger(__name__)

_SRC = r"""
extern "C" __global__
void hamming_match(const unsigned long long* __restrict__ d1,
                   const unsigned long long* __restrict__ d2,
                   const int n1, const int n2,
                   int* __restrict__ out_idx, int* __restrict__ out_best,
                   int* __restrict__ out_second)
{
    // One thread per query; the train set is streamed through shared memory.
    extern __shared__ unsigned long long tile[];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long q0 = 0, q1 = 0, q2 = 0, q3 = 0;
    if (i < n1) {
        q0 = d1[4 * i]; q1 = d1[4 * i + 1]; q2 = d1[4 * i + 2]; q3 = d1[4 * i + 3];
    }
    int best = 1 << 30, second = 1 << 30, bi = -1;
    for (int base = 0; base < n2; base += blockDim.x) {
        const int j = base + threadIdx.x;
        if (j < n2) {
            tile[4 * threadIdx.x + 0] = d2[4 * j + 0];
            tile[4 * threadIdx.x + 1] = d2[4 * j + 1];
            tile[4 * threadIdx.x + 2] = d2[4 * j + 2];
            tile[4 * threadIdx.x + 3] = d2[4 * j + 3];
        }
        __syncthreads();
        const int lim = min((int)blockDim.x, n2 - base);
        if (i < n1) {
            for (int k = 0; k < lim; ++k) {
                const int d = __popcll(q0 ^ tile[4 * k]) + __popcll(q1 ^ tile[4 * k + 1])
                            + __popcll(q2 ^ tile[4 * k + 2]) + __popcll(q3 ^ tile[4 * k + 3]);
                if (d < best) { second = best; best = d; bi = base + k; }
                else if (d < second) { second = d; }
            }
        }
        __syncthreads();
    }
    if (i < n1) { out_idx[i] = bi; out_best[i] = best; out_second[i] = second; }
}

extern "C" __global__
void warp_accumulate(float* __restrict__ acc, float* __restrict__ wsum, const int H, const int W,
                     const unsigned char* __restrict__ img, const int h, const int w,
                     const double m0, const double m1, const double m2,
                     const double m3, const double m4, const double m5,
                     const float g0, const float g1, const float g2,
                     const float power, const int mode)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= W || r >= H) return;
    const double u = m0 * c + m1 * r + m2;
    const double v = m3 * c + m4 * r + m5;
    if (u < 0.0 || v < 0.0 || u > w - 1 || v > h - 1) return;
    double d = fmin(fmin(u + 0.5, w - 0.5 - u), fmin(v + 0.5, h - 0.5 - v));
    const double half = 0.5 * (double)min(w, h);
    float wt = (float)(d / half);
    if (power != 1.0f) wt = powf(wt, power);
    if (wt <= 0.0f) return;
    const int idx = r * W + c;
    if (mode == 1 && wt <= wsum[idx]) return;
    const int x0 = (int)u, y0 = (int)v;
    const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    const float fx = (float)(u - x0), fy = (float)(v - y0);
    const unsigned char* p00 = img + 3 * (y0 * w + x0);
    const unsigned char* p01 = img + 3 * (y0 * w + x1);
    const unsigned char* p10 = img + 3 * (y1 * w + x0);
    const unsigned char* p11 = img + 3 * (y1 * w + x1);
    const float g[3] = {g0, g1, g2};
    for (int ch = 0; ch < 3; ++ch) {
        const float val = (1.f - fy) * ((1.f - fx) * p00[ch] + fx * p01[ch])
                        + fy * ((1.f - fx) * p10[ch] + fx * p11[ch]);
        if (mode == 1) acc[3 * idx + ch] = val * g[ch];
        else           acc[3 * idx + ch] += val * g[ch] * wt;
    }
    if (mode == 1) wsum[idx] = wt;
    else           wsum[idx] += wt;
}
"""

_SRC += r"""
extern "C" __global__
void sample_view(const float* __restrict__ img, const int h, const int w, const int nch,
                 const double r00, const double r01, const double r02,
                 const double r10, const double r11, const double r12,
                 const double r20, const double r21, const double r22,
                 const double c0, const double c1, const double c2,
                 const double f, const double k1, const double k2, const double k3, const double cx, const double cy,
                 const double X0, const double Y0, const double gsd,
                 const float* __restrict__ Z, const int H, const int W, const int Hs,
                 float* __restrict__ out, unsigned char* __restrict__ valid)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= W || r >= H) return;
    const int idx = r * W + c;
    valid[idx] = 0;
    const double dx = X0 + (c + 0.5) * gsd - c0;
    const double dy = Y0 - ((r % Hs) + 0.5) * gsd - c1;   // Z may be a (D, Hs, W) stack
    const double dz = (double)Z[idx] - c2;
    const double zc = r20 * dx + r21 * dy + r22 * dz;
    if (zc <= 1e-6) return;
    const double nx = (r00 * dx + r01 * dy + r02 * dz) / zc;
    const double ny = (r10 * dx + r11 * dy + r12 * dz) / zc;
    const double r2 = nx * nx + ny * ny;
    const double d = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
    const double u = f * d * nx + cx, v = f * d * ny + cy;
    if (u < 0.0 || v < 0.0 || u > w - 1 || v > h - 1) return;
    const int x0 = (int)u, y0 = (int)v;
    const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    const float fx = (float)(u - x0), fy = (float)(v - y0);
    for (int ch = 0; ch < nch; ++ch) {
        const float a = img[(y0 * w + x0) * nch + ch], b = img[(y0 * w + x1) * nch + ch];
        const float e = img[(y1 * w + x0) * nch + ch], g = img[(y1 * w + x1) * nch + ch];
        out[idx * nch + ch] = (1.f - fy) * ((1.f - fx) * a + fx * b) + fy * ((1.f - fx) * e + fx * g);
    }
    valid[idx] = 1;
}
"""

_module = cp.RawModule(code=_SRC, options=("-std=c++11",))
_k_match = _module.get_function("hamming_match")
_k_warp = _module.get_function("warp_accumulate")
_k_sample = _module.get_function("sample_view")


class CUDABackend:
    name = "cuda"
    parallel_blocks = False  # the GPU already parallelises inside each block
    xp = cp

    def __init__(self):
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        self.device_name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        free, total = cp.cuda.runtime.memGetInfo()
        self.free_mem = free
        self._self_test()
        self._depth = None
        try:                                   # depth maps: only when bit-identical to the CPU here
            from ._gpu_depth import GPUDepthSweep
            depth = GPUDepthSweep(cp)
            depth.self_test()
            self._depth = depth
        except Exception as exc:
            msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            log.warning("GPU depth maps unavailable (%s): depth maps run on the CPU", msg)

    @property
    def depth_on_gpu(self) -> bool:
        return self._depth is not None

    def depth_sweep(self, ref, mu, sd, srcs, ray, G, hyps, r, top_k):
        """densify._sweep's hypothesis loop on the GPU, bit-identical to the CPU
        (``_gpu_depth.GPUDepthSweep``); None runs it on the CPU."""
        if self._depth is None:
            return None
        return self._depth.sweep(ref, mu, sd, srcs, ray, G, hyps, r, top_k)

    def depth_stats(self) -> tuple:
        """(sweeps run on the GPU, sweeps sent back to the CPU)."""
        return (0, 0) if self._depth is None else (self._depth.sweeps, self._depth.cpu_fallbacks)

    def _self_test(self):
        """Compile and run every kernel (and the CuPy ops the pipeline uses) once, so any
        toolkit / driver problem surfaces now and select_backend() can fall back to CPU."""
        rng = np.random.default_rng(0)
        d = rng.integers(0, 256, (40, 32), dtype=np.uint8)
        idx, best, _ = self.match(d, d)
        if not (idx == np.arange(40)).all() or best.max() != 0:
            raise RuntimeError("CUDA self-test failed (matcher)")
        img = self.upload(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8))
        blk = self.new_block(8, 8)
        self.warp_accumulate(blk, img, np.array([[1, 0, 2], [0, 1, 2]], float), np.ones(3), 1.0, 0)
        out = self.finalize(blk, 0)
        if out[..., 3].max() != 255:
            raise RuntimeError("CUDA self-test failed (warp)")
        z = cp.zeros((4, 4), cp.float32)
        s, v = self.sample_view(cp.ones((8, 8, 1), cp.float32), (np.diag([1.0, -1.0, -1.0]), np.array([0, 0, 10.0]),
                                                                  10.0, 0.0, 0.0, 0.0, 3.5, 3.5), -1.0, 1.0, 0.5, z)
        a = cp.cumsum(cp.sort(s[..., 0], axis=0), axis=1)
        if float(cp.asnumpy(v).sum()) <= 0 or not np.isfinite(float(a.sum())):
            raise RuntimeError("CUDA self-test failed (sampler)")

    def match(self, d1: np.ndarray, d2: np.ndarray):
        n1, n2 = len(d1), len(d2)
        if not n1 or not n2:
            return (np.full(n1, -1, np.int32), np.full(n1, 1 << 30, np.int32),
                    np.full(n1, 1 << 30, np.int32))
        a = cp.asarray(np.ascontiguousarray(d1, np.uint8).view(np.uint64).reshape(n1, 4))
        b = cp.asarray(np.ascontiguousarray(d2, np.uint8).view(np.uint64).reshape(n2, 4))
        idx = cp.full(n1, -1, cp.int32)
        best = cp.full(n1, 1 << 30, cp.int32)
        second = cp.full(n1, 1 << 30, cp.int32)
        threads = 128
        _k_match(((n1 + threads - 1) // threads,), (threads,),
                 (a, b, np.int32(n1), np.int32(n2), idx, best, second),
                 shared_mem=threads * 4 * 8)
        return cp.asnumpy(idx), cp.asnumpy(best), cp.asnumpy(second)

    def match_mutual(self, d1: np.ndarray, d2: np.ndarray):
        idx12, best12, second12 = self.match(d1, d2)
        idx21, _, _ = self.match(d2, d1)
        return idx12, best12, second12, idx21

    def new_block(self, H: int, W: int):
        return cp.zeros((H, W, 3), cp.float32), cp.zeros((H, W), cp.float32)

    def upload(self, img: np.ndarray):
        return cp.asarray(np.ascontiguousarray(img))

    def image_nbytes(self, dev_img) -> int:
        return int(dev_img.nbytes)

    def warp_accumulate(self, block, dev_img, M, gain, power, mode):
        acc, wsum = block
        H, W = wsum.shape
        h, w = dev_img.shape[:2]
        m = np.asarray(M, np.float64).ravel()
        g = np.asarray(gain, np.float32)
        bs = (32, 8)
        grid = ((W + bs[0] - 1) // bs[0], (H + bs[1] - 1) // bs[1])
        _k_warp(grid, bs, (acc, wsum, np.int32(H), np.int32(W), dev_img, np.int32(h), np.int32(w),
                           *[np.float64(x) for x in m], *[np.float32(x) for x in g],
                           np.float32(power), np.int32(mode)))

    def finalize(self, block, mode):
        acc, wsum = block
        covered = wsum > 0
        if mode == 0:
            rgb = acc / cp.maximum(wsum, 1e-12)[..., None]
        else:
            rgb = acc
        rgb = cp.clip(cp.rint(rgb), 0, 255).astype(cp.uint8)
        out = cp.empty(wsum.shape + (4,), cp.uint8)
        out[..., :3] = rgb * covered[..., None]
        out[..., 3] = covered.astype(cp.uint8) * 255
        return cp.asnumpy(out)

    def finalize_float(self, block, mode):
        acc, wsum = block
        vals = acc / cp.maximum(wsum, 1e-12)[..., None] if mode == 0 else acc
        return cp.asnumpy(vals.astype(cp.float32)), cp.asnumpy(wsum > 0)

    def sample_view(self, img, cam, X0, Y0, gsd, Z):
        R, C, f, k1, k2, k3, cx, cy = cam
        Z = cp.ascontiguousarray(cp.asarray(Z, cp.float32))
        lead = Z.shape[:-2]
        Hs, W = Z.shape[-2:]
        H = int(np.prod(lead, dtype=np.int64)) * Hs if lead else Hs
        h, w, nch = img.shape
        out = cp.zeros((H, W, nch), cp.float32)
        valid = cp.zeros((H, W), cp.uint8)
        Rf = [np.float64(v) for v in np.asarray(R, np.float64).ravel()]
        Cf = [np.float64(v) for v in np.asarray(C, np.float64).ravel()]
        bs = (32, 8)
        grid = ((W + bs[0] - 1) // bs[0], (H + bs[1] - 1) // bs[1])
        _k_sample(grid, bs, (img, np.int32(h), np.int32(w), np.int32(nch), *Rf, *Cf,
                             np.float64(f), np.float64(k1), np.float64(k2), np.float64(k3), np.float64(cx), np.float64(cy),
                             np.float64(X0), np.float64(Y0), np.float64(gsd), Z, np.int32(H), np.int32(W),
                             np.int32(Hs), out, valid))
        return out.reshape(*lead, Hs, W, nch), valid.reshape(*lead, Hs, W)

    def to_numpy(self, a):
        return cp.asnumpy(a)

    def box(self, a, r):
        from .backend import _box_cumsum
        return _box_cumsum(cp, a, r)

    def asarray(self, a, dtype=None):
        return cp.asarray(a, dtype)

    def topk_mean(self, st, k):
        return -cp.sort(-st, axis=0)[:k].mean(axis=0)

    def argmin0(self, a):
        idx = cp.argmin(a, axis=0)
        return idx, cp.take_along_axis(a, idx[None], axis=0)[0]

    def argmax0(self, a):
        idx = cp.argmax(a, axis=0)
        return idx, cp.take_along_axis(a, idx[None], axis=0)[0]
