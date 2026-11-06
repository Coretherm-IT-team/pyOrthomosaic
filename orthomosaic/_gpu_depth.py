from __future__ import annotations

import threading

import numpy as np

SRC = r"""
extern "C" __global__
void om_dw_warp(const float* __restrict__ hyp_a, const float* __restrict__ hyp_b, const int kind,
                const float hyp_t, const float lo, const float* __restrict__ ray, const int H, const int W,
                const float* __restrict__ srcs, const long long* __restrict__ src_off,
                const int* __restrict__ src_h, const int* __restrict__ src_w, const double* __restrict__ geo,
                float* __restrict__ J, float* __restrict__ V)
{
    /* om_sw_warp_row for one source and pixel; the hypothesis is made here as _fast makes it:
       kind 0: a map, 1: hyp_linear (a + t * b), 2: hyp_refine (max(a + t * b, 1e-6), NaN kept) */
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int s = blockIdx.z;
    if (x >= W || y >= H) return;
    const long long HW = (long long)H * W, p = (long long)y * W + x;
    float inv;
    if (kind == 0) {
        inv = hyp_a[p];
    } else if (kind == 1) {
        float t = hyp_t * hyp_b[p];
        inv = hyp_a[p] + t;
    } else {
        float t = hyp_t * hyp_b[p];
        t = hyp_a[p] + t;
        inv = (t >= lo || isnan(t)) ? t : lo;
    }
    const double d = 1.0 / ((double)inv);
    const double rx = ray[p], ry = ray[HW + p], rz = ray[2 * HW + p];
    const double* g = geo + 18 * s;
    const double X = (g[9] + (d * (((g[0] * rx) + (g[1] * ry)) + (g[2] * rz))));
    const double Y = (g[10] + (d * (((g[3] * rx) + (g[4] * ry)) + (g[5] * rz))));
    const double Z = (g[11] + (d * (((g[6] * rx) + (g[7] * ry)) + (g[8] * rz))));
    const double nx = (X / Z), ny = (Y / Z);
    const double r2 = ((nx * nx) + (ny * ny));
    const double dd = (g[12] * (1.0 + (r2 * (g[13] + (r2 * (g[14] + (r2 * g[15])))))));
    const double u = (Z <= 1e-6) ? -1.0 : ((dd * nx) + g[16]);
    const double v = ((dd * ny) + g[17]);
    const long long w = src_w[s], h = src_h[s];
    const double wl = ((double)w - 1.001), hl = ((double)h - 1.001);
    float Jv = 0.0f, Vv = 0.0f;
    /* NaN fails the CPU's bounds test too but is then undefined there; never reached for finite input */
    if (!(u < 0.0 || v < 0.0 || u > wl || v > hl) && !isnan(u) && !isnan(v)) {
        const long long x0 = (long long)u, y0 = (long long)v;
        const double fx = (u - (double)x0), fy = (v - (double)y0);
        const float* img = srcs + src_off[s];
        const double a00 = img[y0 * w + x0], a01 = img[y0 * w + x0 + 1];
        const double a10 = img[(y0 + 1) * w + x0], a11 = img[(y0 + 1) * w + x0 + 1];
        const double val = ((((a00 * (1.0 - fx)) + (a01 * fx)) * (1.0 - fy))
                            + (((a10 * (1.0 - fx)) + (a11 * fx)) * fy));
        if (isfinite(val)) {
            Jv = (float)val;
            Vv = 1.0f;
        }
    }
    J[s * HW + p] = Jv;
    V[s * HW + p] = Vv;
}

extern "C" __global__
void om_dw_hsum(const float* __restrict__ J, const float* __restrict__ V, const float* __restrict__ ref,
                const int H, const int W, const int r, float* __restrict__ hs)
{
    /* om_sw_hsum_row: running window sums along one row of one source -> hs (n, 4, H, W) */
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockIdx.y;
    if (y >= H) return;
    const long long HW = (long long)H * W, row = (long long)y * W;
    const float* Jr = J + s * HW + row;
    const float* Vr = V + s * HW + row;
    const float* Rr = ref + row;
    float* h0 = hs + 4 * s * HW + row;
    float* h1 = h0 + HW;
    float* h2 = h1 + HW;
    float* h3 = h2 + HW;
    const int lim = r < W ? r : W;
    double sv = 0, sj = 0, sjj = 0, sij = 0;
    for (int x = 0; x < lim; x++) {
        sv = (sv + Vr[x]);
        sj = (sj + Jr[x]);
        sjj = (sjj + (Jr[x] * Jr[x]));
        sij = (sij + (Jr[x] * Rr[x]));
    }
    for (int x = 0; x < W; x++) {
        if (x + r < W) {
            sv = (sv + Vr[x + r]);
            sj = (sj + Jr[x + r]);
            sjj = (sjj + (Jr[x + r] * Jr[x + r]));
            sij = (sij + (Jr[x + r] * Rr[x + r]));
        }
        if (x - r - 1 >= 0) {
            sv = (sv - Vr[x - r - 1]);
            sj = (sj - Jr[x - r - 1]);
            sjj = (sjj - (Jr[x - r - 1] * Jr[x - r - 1]));
            sij = (sij - (Jr[x - r - 1] * Rr[x - r - 1]));
        }
        h0[x] = (float)sv;
        h1[x] = (float)sj;
        h2[x] = (float)sjj;
        h3[x] = (float)sij;
    }
}

extern "C" __global__
void om_dw_vsum(const float* __restrict__ hs, const float* __restrict__ mu, const float* __restrict__ sd,
                const int H, const int W, const int r, const double kk, float* __restrict__ out)
{
    /* the vertical sums of sweep_hypothesis (om_sw_cs_add / _addsub / _sub, add before subtract)
       down one column of one source, and om_sw_ncc_row -> out (n, H, W) */
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockIdx.y;
    if (x >= W) return;
    const long long HW = (long long)H * W;
    const float* h0 = hs + 4 * s * HW;
    const float* h1 = h0 + HW;
    const float* h2 = h1 + HW;
    const float* h3 = h2 + HW;
    float* o = out + s * HW;
    double c0 = 0, c1 = 0, c2 = 0, c3 = 0, t, cov, mj, m2, mij, sdj, ncc;
    const int lim = r < H ? r : H;
    for (int yy = 0; yy < lim; yy++) {
        const long long q = (long long)yy * W + x;
        c0 = (c0 + h0[q]);
        c1 = (c1 + h1[q]);
        c2 = (c2 + h2[q]);
        c3 = (c3 + h3[q]);
    }
    for (int y = 0; y < H; y++) {
        if (y + r < H) {
            const long long a = (long long)(y + r) * W + x;
            if (y - r - 1 >= 0) {
                const long long b = (long long)(y - r - 1) * W + x;
                t = (c0 + h0[a]); c0 = (t - h0[b]);
                t = (c1 + h1[a]); c1 = (t - h1[b]);
                t = (c2 + h2[a]); c2 = (t - h2[b]);
                t = (c3 + h3[a]); c3 = (t - h3[b]);
            } else {
                c0 = (c0 + h0[a]);
                c1 = (c1 + h1[a]);
                c2 = (c2 + h2[a]);
                c3 = (c3 + h3[a]);
            }
        } else if (y - r - 1 >= 0) {
            const long long b = (long long)(y - r - 1) * W + x;
            c0 = (c0 - h0[b]);
            c1 = (c1 - h1[b]);
            c2 = (c2 - h2[b]);
            c3 = (c3 - h3[b]);
        }
        const long long p = (long long)y * W + x;
        cov = (c0 * kk);
        if (cov <= 0.9) {
            o[p] = -1.0f;
            continue;
        }
        mj = ((c1 * kk) / cov);
        m2 = ((c2 * kk) / cov);
        mij = ((c3 * kk) / cov);
        sdj = (m2 - (mj * mj));
        if (sdj > 1e-6) {
            sdj = sqrt(sdj);
        } else {
            sdj = 1e-3;
        }
        ncc = ((mij - (mu[p] * mj)) / (sd[p] * sdj));
        if (ncc > 1.0) {
            o[p] = (float)1.0;
        } else if (ncc < -1.0) {
            o[p] = (float)-1.0;
        } else {
            o[p] = (float)ncc;
        }
    }
}

extern "C" __global__
void om_dw_combine(const float* __restrict__ ncc, const int n, const long long HW, const int kk, const int i,
                   float* __restrict__ best, float* __restrict__ prev, float* __restrict__ s_prev_best,
                   float* __restrict__ s_next_best, int* __restrict__ idx)
{
    /* om_sw_combine_row for one pixel: mean of the top-kk source scores, streaming winner update */
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= HW) return;
    float buf[16];
    float t, s;
    int a, c;
    for (a = 0; a < n; a++)
        buf[a] = ncc[a * HW + p];
    for (a = 1; a < n; a++) {
        t = buf[a];
        c = a - 1;
        while (c >= 0 && buf[c] < t) {
            buf[c + 1] = buf[c];
            c -= 1;
        }
        buf[c + 1] = t;
    }
    s = 0;
    for (a = 0; a < kk; a++)
        s = (s + buf[a]);
    s = (s / ((float)kk));
    if (idx[p] == (i - 1))
        s_next_best[p] = s;
    if (s > best[p]) {
        best[p] = s;
        s_prev_best[p] = prev[p];
        s_next_best[p] = -2.0f;
        idx[p] = i;
    }
    prev[p] = s;
}
"""

# CuPy adds -ftz=true itself (float32 denormals, below 1.2e-38, flush to zero: never reached here);
# NVRTC divides and takes square roots IEEE-rounded by default
OPTIONS = ("-std=c++11", "--fmad=false")
KERNELS = ("om_dw_warp", "om_dw_hsum", "om_dw_vsum", "om_dw_combine")


def device_bytes(H: int, W: int, src_pixels: int, n: int) -> int:
    """Device memory of one sweep: ref, mu, sd, 3 ray planes, 2 hypothesis planes, 5 state planes,
    and per source the image, J, V, 4 window-sum planes and its NCC."""
    return 4 * (H * W * (13 + 7 * n) + src_pixels) + 4096


class GPUDepthSweep:
    """The plane sweep of ``densify._sweep`` on an NVIDIA GPU, with the CPU's result.

    ``_dense.sweep_hypothesis`` (``_sweep.h``) runs one hypothesis row by row on one CPU core. Here
    the same work is four kernels per hypothesis (`SRC`): warp (a thread per source and pixel),
    horizontal window sums (per source and row), vertical window sums + NCC (per source and
    column) and the top-k / winner update (per pixel). The running sums keep their order inside a
    thread, every expression is written as in ``_sweep.h`` (operand order, parenthesisation,
    float32/float64 rounding points) and the module is compiled with ``--fmad=false``, so no
    multiply-add is fused: the depth maps are bit-identical to the x86-64 CPU build, where nothing
    is fused either. `self_test` checks that on the device before it is used. `SRC` has no CuPy
    dependency, so tests/test_gpu_depth.py also compiles it with a C++ compiler and runs it on the CPU.

    ``cp`` is CuPy (or a stand-in with the same calls). Sweeps from several threads run
    concurrently, each on its own stream, within a budget of the device memory that was free at
    start; a sweep that does not fit (or meets an out-of-memory error) returns None and the caller
    runs it on the CPU, with the same result."""

    def __init__(self, cp, budget_fraction: float = 0.6):
        self.cp = cp
        mod = cp.RawModule(code=SRC, options=OPTIONS)
        self._k = {name: mod.get_function(name) for name in KERNELS}
        free, _ = cp.cuda.runtime.memGetInfo()
        self.budget = self._free = int(budget_fraction * free)
        self._cv = threading.Condition()
        self._local = threading.local()
        self.sweeps = 0
        self.cpu_fallbacks = 0

    def _stream(self):
        s = getattr(self._local, "stream", None)
        if s is None:
            s = self._local.stream = self.cp.cuda.Stream(non_blocking=True)
        return s

    def sweep(self, ref, mu, sd, srcs, ray, G, hyps, r, top_k):
        """-> (best, s_prev_best, s_next_best, idx) as numpy arrays, exactly what the CPU loop of
        ``_dense.sweep_hypothesis`` over `hyps` leaves in them; None to run it on the CPU."""
        H, W = ref.shape
        n = len(srcs)
        if not 1 <= n <= 16 or H == 0 or W == 0:
            return None
        need = device_bytes(H, W, sum(int(s.size) for s in srcs), n)
        if need > self.budget:
            self.cpu_fallbacks += 1
            return None
        with self._cv:
            while self._free < need:
                self._cv.wait()
            self._free -= need
        try:
            with self._stream():
                out = self._run(ref, mu, sd, srcs, ray, G, hyps, r, top_k)
            self.sweeps += 1
            return out
        except self.cp.cuda.memory.OutOfMemoryError:
            self.cpu_fallbacks += 1
            return None
        finally:
            with self._cv:
                self._free += need
                self._cv.notify_all()

    def _run(self, ref, mu, sd, srcs, ray, G, hyps, r, top_k):
        cp, k = self.cp, self._k
        f32, i32 = np.float32, np.int32
        H, W = ref.shape
        n, HW = len(srcs), H * W
        dev = lambda a, dt=f32: cp.asarray(np.ascontiguousarray(a, dt))     # noqa: E731
        d_ref, d_mu, d_sd, d_ray = dev(ref), dev(mu), dev(sd), dev(ray)
        d_src = dev(np.concatenate([np.ascontiguousarray(s, f32).ravel() for s in srcs]))
        d_off = dev(np.cumsum([0] + [int(s.size) for s in srcs[:-1]]), np.int64)
        d_h, d_w = dev([s.shape[0] for s in srcs], i32), dev([s.shape[1] for s in srcs], i32)
        d_geo = dev(G, np.float64)
        J, V = cp.empty((n, H, W), f32), cp.empty((n, H, W), f32)
        hs, nc = cp.empty((n, 4, H, W), f32), cp.empty((n, H, W), f32)
        best, prev = cp.full((H, W), -2.0, f32), cp.full((H, W), -2.0, f32)
        spb, snb = cp.full((H, W), -2.0, f32), cp.full((H, W), -2.0, f32)
        idx = cp.full((H, W), -1, i32)
        spec = getattr(hyps, "spec", None)
        kind = {"linear": 1, "refine": 2}.get(spec[0], 0) if spec else 0
        if kind:
            a, b = dev(spec[1]), dev(spec[2])
        kk = top_k if top_k < n else n
        kkd = 1.0 / ((2 * r + 1) * (2 * r + 1))
        lo = f32(1e-6)
        g_warp = ((W + 31) // 32, (H + 7) // 8, n)
        for i in range(len(hyps)):
            if kind == 0:
                a = b = dev(hyps[i])
            t = f32(0 if kind == 0 else i if kind == 1 else i - spec[3])
            k["om_dw_warp"](g_warp, (32, 8, 1), (a, b, i32(kind), t, lo, d_ray, i32(H), i32(W), d_src, d_off,
                                                   d_h, d_w, d_geo, J, V))
            k["om_dw_hsum"](((H + 63) // 64, n), (64,), (J, V, d_ref, i32(H), i32(W), i32(r), hs))
            k["om_dw_vsum"](((W + 63) // 64, n), (64,), (hs, d_mu, d_sd, i32(H), i32(W), i32(r), np.float64(kkd), nc))
            k["om_dw_combine"](((HW + 255) // 256,), (256,), (nc, i32(n), np.int64(HW), i32(kk), i32(i), best, prev,
                                                              spb, snb, idx))
        return cp.asnumpy(best), cp.asnumpy(spb), cp.asnumpy(snb), cp.asnumpy(idx)

    def self_test(self):
        """Raise unless the device reproduces the CPU kernel bit for bit on a small sweep."""
        case = sweep_case(np.random.default_rng(11), 37, 53, 4, 5, 9)
        got = self.sweep(*case, 5, 2)
        if got is None:
            raise RuntimeError("GPU depth sweep did not run")
        for g, w in zip(got, cpu_sweep(*case, 5, 2)):
            if not same_bits(g, w):
                bad = np.abs(g.astype(np.float64) - w.astype(np.float64))
                raise RuntimeError(f"GPU depth sweep differs from the CPU (max |diff| {np.nanmax(bad):.3g})")


def same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    """Equal shape, dtype and bits (+0 and -0 differ; any NaN equals any NaN)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind == "f":
        na, nb = np.isnan(a), np.isnan(b)
        if not np.array_equal(na, nb):
            return False
        a, b = np.where(na, 0, a), np.where(nb, 0, b)
    return np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()


def cpu_sweep(ref, mu, sd, srcs, ray, G, hyps, r, top_k):
    """The CPU loop of densify._sweep (``_dense.sweep_hypothesis`` per hypothesis)."""
    from . import _dense
    H, W = ref.shape
    best, prev = np.full((H, W), -2.0, np.float32), np.full((H, W), -2.0, np.float32)
    spb, snb = np.full((H, W), -2.0, np.float32), np.full((H, W), -2.0, np.float32)
    idx = np.full((H, W), -1, np.int32)
    for i in range(len(hyps)):
        _dense.sweep_hypothesis(ref, mu, sd, srcs, ray, G, np.ascontiguousarray(hyps[i], np.float32), r, top_k, i,
                                best, prev, spb, snb, idx)
    return best, spb, snb, idx


def sweep_case(rng, H, W, n, r, nhyp):
    """A random sweep (reference, window stats, sources, rays, camera table, hypotheses) whose
    projections land inside the sources, with an infinite and a negative hypothesis value."""
    ref = rng.uniform(-2, 2, (H, W)).astype(np.float32)
    mu = rng.uniform(-1, 1, (H, W)).astype(np.float32)
    sd = rng.uniform(0.05, 1.0, (H, W)).astype(np.float32)
    srcs, rows = [], []
    for _ in range(n):
        h, w = max(2, H + int(rng.integers(-3, 4))), max(2, W + int(rng.integers(-3, 4)))
        img = rng.uniform(-2, 2, (h, w)).astype(np.float32)
        img[rng.random((h, w)) < 0.01] = np.nan                 # masked pixels
        srcs.append(img)
        a = rng.normal(0, 0.03, 3)
        Rs = _rodrigues(a)
        b = rng.normal(0, 0.3, 3)
        f = float(rng.uniform(0.8, 1.2) * w)
        rows.append(np.r_[Rs.ravel(), b, [f, rng.normal(0, 0.05), rng.normal(0, 0.01), rng.normal(0, 0.003),
                                          (w - 1) / 2 + rng.normal(0, 2), (h - 1) / 2 + rng.normal(0, 2)]])
    yy, xx = np.mgrid[0:H, 0:W]
    fr = float(W)
    ray = np.stack([(xx - W / 2) / fr, (yy - H / 2) / fr, np.ones((H, W))]).astype(np.float32)
    base = rng.uniform(0.02, 0.2, (H, W)).astype(np.float32)
    step = rng.uniform(1e-4, 5e-3, (H, W)).astype(np.float32)
    hyps = [(base + i * step).astype(np.float32) for i in range(nhyp)]
    if nhyp > 2:
        hyps[1][0, 0] = np.inf
        hyps[2][-1, -1] = -0.01
    return ref, mu, sd, srcs, np.ascontiguousarray(ray), np.array(rows, np.float64), hyps


def _rodrigues(a):
    th = float(np.linalg.norm(a))
    if th < 1e-12:
        return np.eye(3)
    k = a / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
