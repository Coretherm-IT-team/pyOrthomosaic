"""Bit-exact parity of the optimised native kernels against the implementations they replace.

Every optimised kernel must reproduce the original results exactly (same float32/float64
rounding, NaN positions, tie-breaking and update order), not approximately: comparisons below
are on the raw bits (``_same``), never ``allclose``.
"""
import os
import sys

import numpy as np

try:
    import orthomosaic._dense  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orthomosaic import _dense  # noqa: E402


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind == "f":
        return bool((a.view(f"u{a.itemsize}") == b.view(f"u{b.itemsize}")).all())
    return bool((a == b).all())


def _rot(rng, tilt=0.15):
    w = rng.normal(0, tilt, 3) + np.array([np.pi, 0, 0])
    th = np.linalg.norm(w)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _sweep_case(rng, H, W, n, r, top_k, nhyp, nan_frac=0.02):
    ref = rng.normal(0, 1, (H, W)).astype(np.float32)
    srcs = []
    for _ in range(n):
        h, w = H + int(rng.integers(-3, 4)), W + int(rng.integers(-3, 4))
        s = rng.normal(0, 1, (max(h, 2), max(w, 2))).astype(np.float32)
        s[rng.random(s.shape) < nan_frac] = np.nan            # masked pixels
        srcs.append(np.ascontiguousarray(s))
    k = 2 * r + 1
    pad = np.pad(ref, r)
    mu = np.zeros_like(ref)
    m2 = np.zeros_like(ref)
    for dy in range(k):
        for dx in range(k):
            mu += pad[dy:dy + H, dx:dx + W]
            m2 += pad[dy:dy + H, dx:dx + W] ** 2
    mu /= k * k
    sd = np.sqrt(np.maximum(m2 / (k * k) - mu * mu, 1e-6)).astype(np.float32)
    ray = rng.normal(0, 0.3, (3, H, W)).astype(np.float32)
    ray[2] = 1.0 + rng.normal(0, 0.01, (H, W)).astype(np.float32)
    geo, cams = [], []
    for _ in range(n):
        Rs = _rot(rng, 0.05) @ _rot(rng, 0.0).T
        b = rng.normal(0, 0.5, 3)
        f = float(rng.uniform(0.8, 1.2) * W)
        par = [f, float(rng.normal(0, 0.05)), float(rng.normal(0, 0.01)), float(rng.normal(0, 0.003)),
               (W - 1) / 2 + float(rng.normal(0, 2)), (H - 1) / 2 + float(rng.normal(0, 2))]
        geo.append((np.ascontiguousarray(Rs), np.ascontiguousarray(b)))
        cams.append(par)
    base = rng.uniform(0.02, 0.2, (H, W)).astype(np.float32)
    step = rng.uniform(1e-4, 5e-3, (H, W)).astype(np.float32)
    hyps = [(base + i * step).astype(np.float32) for i in range(nhyp)]
    if nhyp > 2:
        hyps[1][0, 0] = np.inf                                 # 1/inv = 0 depth
        hyps[2][-1, -1] = -0.01                                # behind the camera
    return ref, mu.astype(np.float32), sd, srcs, ray, geo, cams, hyps


def _run_old(ref, mu, sd, srcs, ray, geo, cams, hyps, r, top_k):
    H, W = ref.shape
    S = np.empty((len(srcs), H, W), np.float32)
    J, V, hs = np.empty((H, W), np.float32), np.empty((H, W), np.float32), np.empty((4, H, W), np.float32)
    st = [np.full((H, W), -2.0, np.float32) for _ in range(4)]
    idx = np.full((H, W), -1, np.int32)
    for i, inv in enumerate(hyps):
        for j, (src, (Rs, b), c) in enumerate(zip(srcs, geo, cams)):
            _dense.ncc_warp(ref, mu, sd, src, ray, Rs, b, *c[:1], *c[1:4], *c[4:], inv, r, S[j], J, V, hs)
        _dense.combine(S, top_k, i, st[0], st[1], st[2], st[3], idx)
    return st, idx


def _run_new(ref, mu, sd, srcs, ray, geo, cams, hyps, r, top_k):
    H, W = ref.shape
    G = np.array([np.r_[Rs.ravel(), b, c] for (Rs, b), c in zip(geo, cams)], np.float64)
    st = [np.full((H, W), -2.0, np.float32) for _ in range(4)]
    idx = np.full((H, W), -1, np.int32)
    for i, inv in enumerate(hyps):
        _dense.sweep_hypothesis(ref, mu, sd, srcs, ray, G, inv, r, top_k, i, st[0], st[1], st[2], st[3], idx)
    return st, idx


def test_sweep_hypothesis_matches_ncc_warp_combine():
    rng = np.random.default_rng(7)
    cases = [(37, 53, 4, 5, 2, 9), (64, 48, 1, 3, 2, 5), (11, 9, 3, 5, 2, 4), (5, 4, 2, 5, 1, 3),
             (30, 31, 5, 2, 3, 6), (2, 40, 2, 1, 2, 3), (40, 3, 4, 4, 4, 3)]
    for H, W, n, r, top_k, nh in cases:
        case = _sweep_case(rng, H, W, n, r, top_k, nh)
        (a, ia), (b, ib) = _run_old(*case, r, top_k), _run_new(*case, r, top_k)
        assert _same(ia, ib), (H, W, n, r)
        for x, y in zip(a, b):
            assert _same(x, y), (H, W, n, r)
    # equal scores everywhere (flat images): ties keep the first hypothesis, as before
    ref, mu, sd, srcs, ray, geo, cams, hyps = _sweep_case(rng, 20, 24, 3, 2, 2, 5, 0.0)
    flat = [np.ones_like(s) for s in srcs]
    (a, ia), (b, ib) = (_run_old(ref, mu, sd, flat, ray, geo, cams, hyps, 2, 2),
                        _run_new(ref, mu, sd, flat, ray, geo, cams, hyps, 2, 2))
    assert _same(ia, ib) and all(_same(x, y) for x, y in zip(a, b))


def test_sweep_end_to_end_matches_reference_loop():
    """densify._sweep (lazy hypotheses + fused kernel) == the original list + two-kernel loop."""
    from orthomosaic import densify
    rng = np.random.default_rng(3)
    H, W = 45, 61
    ref, mu, sd, srcs, ray, geo, cams, hyps = _sweep_case(rng, H, W, 4, 5, 2, 19)

    class Cam:
        def __init__(self, R, C, par):
            self.R, self.C = R, C
            self.f, self.k1, self.k2, self.k3, self.cx, self.cy = par

    cam_r = Cam(np.eye(3), np.zeros(3), [W, 0.0, 0.0, 0.0, W / 2, H / 2])
    cams_s = []
    for (Rs, b), c in zip(geo, cams):
        Cs = -Rs.T @ b                       # so that Rs @ (C_ref - C_src) == b
        cams_s.append(Cam(Rs, Cs, c))
    opt = densify.DepthOptions()
    opt.window = 5
    lazy = densify._Hypotheses(lambda i: hyps[i].copy(), len(hyps))
    d1, s1, sd1 = densify._sweep(ref, cam_r, srcs, cams_s, lazy, opt, ray)
    # reference: the pre-optimisation body of _sweep
    r = opt.window
    mu0 = np.ascontiguousarray(densify._box(ref, r), np.float32)
    sd0 = np.ascontiguousarray(np.sqrt(np.maximum(densify._box(ref * ref, r) - mu0 * mu0, 1e-6)), np.float32)
    st, idx = _run_old(ref, mu0, sd0, srcs, ray,
                       [(np.ascontiguousarray(cs.R), np.ascontiguousarray(cs.R @ (cam_r.C - cs.C))) for cs in cams_s],
                       cams, hyps, r, opt.top_k)
    best, prev, spb, snb = st
    n = len(hyps)
    inv0, dinv = hyps[0], hyps[1] - hyps[0]
    den = spb - 2 * best + snb
    interior = (idx > 0) & (idx < n - 1) & (den < -1e-6)
    off = np.where(interior, 0.5 * (spb - snb) / np.where(interior, den, -1.0), 0.0)
    inv = inv0 + (idx + np.clip(off, -0.5, 0.5)) * dinv
    d0 = (1.0 / np.maximum(inv, 1e-9)).astype(np.float32)
    assert _same(d0, d1) and _same(best, s1) and _same(sd0, sd1)


def _ba_problem(seed=0, N=24, P=3000, G=2):
    from orthomosaic.camera import rodrigues
    rng = np.random.default_rng(seed)
    R = rodrigues(rng.normal(0, 0.05, (N, 3)) + np.array([np.pi, 0, 0]))
    C = np.column_stack([rng.uniform(-30, 30, (N, 2)), np.full(N, 60.0)])
    X = np.column_stack([rng.uniform(-40, 40, (P, 2)), rng.uniform(0, 12, P)])
    intr = np.array([[1000.0, -0.05, 0.01, 0.02, 600.0, 450.0], [900.0, 0.02, 0.0, -0.01, 640.0, 512.0]])[:G]
    pp = intr[:, 4:6].copy()
    cg = (np.arange(N) % G).astype(np.int32)
    oc, op = [], []
    for p in range(P):
        m = int(rng.integers(1, 9))
        cams = np.sort(rng.choice(N, m, replace=False))
        oc += cams.tolist()
        op += [p] * m
    oc, op = np.array(oc, np.int32), np.array(op, np.int32)
    uv = np.column_stack([rng.uniform(0, 1200, len(oc)), rng.uniform(0, 900, len(oc))])
    behind = rng.random(len(oc)) < 0.01
    C2 = C.copy()
    X[op[behind], 2] = 80.0                                  # above the camera -> z <= 0
    ptr = np.searchsorted(op, np.arange(P + 1)).astype(np.int64)
    pw = np.zeros((P, 3))
    sel = rng.random(P) < 0.01
    pw[sel] = 400.0
    pt = X + rng.normal(0, 0.1, X.shape)
    return R, C2, X, intr, pp, cg, oc, op, uv, ptr, pw, pt


def test_bundle_kernels_threads_bit_identical():
    from orthomosaic import _ba
    R, C, X, intr, pp, cg, oc, op, uv, ptr, pw, pt = _ba_problem()
    for huber in (1e9, 40.0, 3.0):
        ref = _ba.reduced_system(R, C, X, intr, pp, cg, oc, uv, ptr, huber, 1e-3, pw, pt)
        e0, c0 = _ba.residuals(R, C, X, intr, pp, cg, oc, op, uv, huber)
        dc = np.random.default_rng(1).normal(0, 1e-3, len(ref[1]))
        dp0 = _ba.back_substitute(R, C, X, intr, pp, cg, oc, uv, ptr, huber, ref[2], ref[3], dc)
        for workers in (2, 3, 5, 8, 13):
            out = _ba._reduced_system_mt(R, C, X, intr, pp, cg, oc, uv, ptr, huber, 1e-3, pw, pt, workers)
            for a, b in zip(ref, out):
                assert _same(np.asarray(a), np.asarray(b)), (huber, workers)
        for workers in (2, 8):
            e1, c1 = _ba.residuals(R, C, X, intr, pp, cg, oc, op, uv, huber, workers=workers)
            assert _same(e0, e1) and _same(np.float64(c0), np.float64(c1))
            dp1 = _ba.back_substitute(R, C, X, intr, pp, cg, oc, uv, ptr, huber, ref[2], ref[3], dc, workers=workers)
            assert _same(dp0, dp1)
            out = _ba.reduced_system(R, C, X, intr, pp, cg, oc, uv, ptr, huber, 1e-3, pw, pt, workers=workers)
            assert all(_same(np.asarray(a), np.asarray(b)) for a, b in zip(ref, out))


def test_tracks_flat_matches_union_find():
    from orthomosaic import sfm
    rng = np.random.default_rng(11)
    for trial in range(6):
        n_img = int(rng.integers(3, 12))
        n_feats = [int(rng.integers(20, 300)) for _ in range(n_img)]
        results = []
        for i in range(n_img):
            for j in range(i + 1, n_img):
                if rng.random() < 0.6:
                    m = int(rng.integers(8, min(120, n_feats[i])))
                    a = rng.choice(n_feats[i], m, replace=False)
                    b = rng.integers(0, n_feats[j], m)          # repeated b -> conflicting merges
                    results.append(((i, j), (a, b.astype(np.int32))))
        rng.shuffle(results)
        old = sfm._build_tracks(n_feats, results, 2)
        ti, tf, lens = sfm._build_tracks_flat(n_feats, results, 2)
        assert len(old) == len(lens) and all(len(t[0]) == n for t, n in zip(old, lens))
        assert _same(np.concatenate([t[0] for t in old]), ti) and _same(np.concatenate([t[1] for t in old]), tf)
        old3 = sfm._build_tracks(n_feats, results, 3)
        ti3, tf3, l3 = sfm._build_tracks_flat(n_feats, results, 3)
        assert len(old3) == len(l3) and _same(np.concatenate([t[0] for t in old3]), ti3)


def _cams(rng, n=4, W=640, H=480):
    from orthomosaic.densify import _Cam
    out = []
    for _ in range(n):
        R = _rot(rng, 0.1)
        C = np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), 50.0])
        out.append(_Cam(R, C, float(rng.uniform(500, 700)), float(rng.normal(0, 0.08)), float(rng.normal(0, 0.03)),
                        float(W / 2 + rng.normal(0, 3)), float(H / 2 + rng.normal(0, 3)), W, H,
                        float(rng.normal(0, 0.01))))
    return out


def test_camera_kernels_match_numpy():
    from orthomosaic import _fast
    rng = np.random.default_rng(5)
    cams = _cams(rng)
    cr = cams[0]
    H, W = 37, 53
    x, y = _fast.rays_grid(H, W, cr.cx, cr.cy, cr.f, cr.k1, cr.k2, cr.k3)
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
    x0, y0 = cr.rays(uu, vv)
    assert _same(x, x0) and _same(y, y0)
    # fusion / consistency steps on a depth map with holes, borders and behind-camera points
    Hd, Wd = 480, 640
    D = rng.uniform(40, 60, (Hd, Wd)).astype(np.float32)
    D[rng.random(D.shape) < 0.1] = np.nan
    Dj = rng.uniform(40, 60, (Hd, Wd)).astype(np.float32)
    Dj[rng.random(Dj.shape) < 0.1] = np.nan
    vv, uu = (np.ascontiguousarray(a) for a in np.nonzero(np.isfinite(D)))
    d = D[vv, uu].astype(np.float64)
    xr, yr = cr.rays(uu.astype(np.float64), vv.astype(np.float64))
    M0 = np.stack([xr * d, yr * d, d], 1)
    M1 = _fast.ref_points(D, vv, uu, cr.cx, cr.cy, cr.f, cr.k1, cr.k2, cr.k3)
    assert _same(M0, M1)
    X = cr.C + (M0 @ cr.R)
    X[::97, 2] = 80.0                                  # above camera j: behind it
    for cj in cams[1:]:
        xc = (X - cj.C) @ cj.R.T
        u, v = cj.project_cam(xc[:, 0], xc[:, 1], xc[:, 2])
        ui, vi = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
        ins = (xc[:, 2] > 0) & (ui >= 0) & (vi >= 0) & (ui < cj.w) & (vi < cj.h)
        dj = np.full(len(vv), np.nan)
        dj[ins] = Dj[vi[ins], ui[ins]]
        xj, yj = cj.rays(ui.astype(np.float64), vi.astype(np.float64))
        Mj = np.stack([xj * dj, yj * dj, dj], 1)
        a, b, c, e = _fast.neighbour_sample(xc, Dj, cj.f, cj.k1, cj.k2, cj.k3, cj.cx, cj.cy, cj.w, cj.h)
        assert _same(a, ui) and _same(b, vi) and _same(c, dj) and _same(e, Mj)
        Xj = cj.C + (Mj @ cj.R)
        xk = (Xj - cr.C) @ cr.R.T
        ub, vb = cr.project_cam(xk[:, 0], xk[:, 1], xk[:, 2])
        for tol_px, rel in ((1.5, 0.03), (0.5, 0.01), (3.0, 0.05)):
            ok = np.isfinite(dj) & (np.hypot(ub - uu, vb - vv) <= tol_px) & (np.abs(dj - xc[:, 2]) <= rel * dj)
            ok1 = _fast.agreement(xk, xc, dj, uu, vv, cr.f, cr.k1, cr.k2, cr.k3, cr.cx, cr.cy, tol_px, rel)
            assert _same(ok, ok1)
        acc0, cnt0 = X.copy(), np.ones(len(vv), np.int32)
        acc0[ok] += Xj[ok]
        cnt0 += ok
        acc1, cnt1 = X.copy(), np.ones(len(vv), np.int32)
        _fast.accumulate(acc1, cnt1, Xj, ok1.view(np.uint8))
        assert _same(acc0, acc1) and _same(cnt0, cnt1)


def _ply_reference(path, xyz, rgb, offset=(0.0, 0.0, 0.0), comment=""):
    n = len(xyz)
    rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ["ply", "format binary_little_endian 1.0", f"comment offset {offset[0]:.3f} {offset[1]:.3f} {offset[2]:.3f}"]
    if comment:
        header.append(f"comment {comment}")
    header += [f"element vertex {n}", "property float x", "property float y", "property float z",
               "property uchar red", "property uchar green", "property uchar blue", "end_header"]
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode())
        fh.write(rec.tobytes())


def test_chunked_writers_same_bytes():
    import tempfile
    from orthomosaic import export
    rng = np.random.default_rng(9)
    n = 2500
    xyz = np.column_stack([rng.uniform(-50, 50, n), rng.uniform(-50, 50, n), rng.uniform(-5, 30, n)])
    rgb = rng.integers(0, 256, (n, 3)).astype(np.uint8)
    cls = rng.integers(1, 7, n).astype(np.uint8)
    off = np.array([592264.0, 4525733.0, 0.0])
    old_chunk = export._CHUNK
    try:
        with tempfile.TemporaryDirectory() as d:
            _ply_reference(os.path.join(d, "a.ply"), xyz, rgb, off, "EPSG:32618")
            export._CHUNK = 333                       # many blocks, last one partial
            export.write_ply(os.path.join(d, "b.ply"), xyz, rgb, off, "EPSG:32618")
            assert open(os.path.join(d, "a.ply"), "rb").read() == open(os.path.join(d, "b.ply"), "rb").read()
            for c in (None, cls):
                export._CHUNK = 10 ** 9               # one block == the original single-array writer
                export.write_las(os.path.join(d, "a.las"), xyz + off, rgb, 32618, scale=0.001, classification=c)
                export._CHUNK = 333
                export.write_las(os.path.join(d, "b.las"), xyz, rgb, 32618, scale=0.001, classification=c, add=off)
                a, b = open(os.path.join(d, "a.las"), "rb").read(), open(os.path.join(d, "b.las"), "rb").read()
                assert a == b
            z0 = xyz.copy()
            z0[:, 2] -= z0[:, 2].min()                # a zero bound takes the whole-array path
            export.write_las(os.path.join(d, "c.las"), z0, rgb, None, add=off)
            export._CHUNK = 10 ** 9
            export.write_las(os.path.join(d, "d.las"), z0 + off, rgb, None)
            assert open(os.path.join(d, "c.las"), "rb").read() == open(os.path.join(d, "d.las"), "rb").read()
    finally:
        export._CHUNK = old_chunk


def test_streamed_cloud_export_same_bytes():
    """The dense-cloud export makes the warped coordinates and classes block by block
    (RowBlocks, GpsWarp.xy_rows); the PLY / LAS must equal the full-array writers bit for bit."""
    import tempfile
    from orthomosaic import export
    from orthomosaic.reconstruct import GpsWarp
    rng = np.random.default_rng(11)
    n = 2500
    xyz = np.column_stack([rng.uniform(-50, 50, n), rng.uniform(-50, 50, n), rng.uniform(-5, 30, n)])
    rgb = rng.integers(0, 256, (n, 3)).astype(np.uint8)
    cls = rng.integers(1, 7, n).astype(np.uint8)
    off = np.array([592264.0, 4525733.0, 0.0])
    warp = GpsWarp([[1.0013, 0.0021], [-0.0017, 0.9981]], [0.12, -0.07], [3.0, -4.0])
    old_chunk, old_block = export._CHUNK, GpsWarp.BLOCK
    try:
        GpsWarp.BLOCK = 256                           # warp products on 256-row blocks, like 2**21
        full = warp.xy(xyz)
        for a, b in ((0, n), (100, 900), (255, 257), (2400, n), (7, 7)):
            assert _same(warp.xy_rows(xyz, a, b), full[a:b])
        lazy = export.RowBlocks(n, lambda a, b: warp.xy_rows(xyz, a, b), 3)
        lazy_cls = export.RowBlocks(n, lambda a, b: cls[a:b])
        with tempfile.TemporaryDirectory() as d:
            p = lambda name: os.path.join(d, name)  # noqa: E731
            export._CHUNK = 333                       # writer blocks not aligned to the warp blocks
            export.write_ply(p("a.ply"), full, rgb, off, "EPSG:32618")
            export.write_ply(p("b.ply"), lazy, rgb, off, "EPSG:32618")
            assert open(p("a.ply"), "rb").read() == open(p("b.ply"), "rb").read()
            export.write_las(p("a.las"), full, rgb, 32618, scale=0.001, classification=cls, add=off)
            export.write_las(p("b.las"), lazy, rgb, 32618, scale=0.001, classification=lazy_cls, add=off)
            assert open(p("a.las"), "rb").read() == open(p("b.las"), "rb").read()
    finally:
        export._CHUNK, GpsWarp.BLOCK = old_chunk, old_block


def test_box_edge_mean_matches_cumsum_box():
    from orthomosaic import _fast
    from orthomosaic.mvs import _box_mean
    from orthomosaic.ortho import _box
    rng = np.random.default_rng(4)
    for H, W, r in ((37, 53, 3), (5, 4, 7), (1, 9, 2), (64, 64, 20), (9, 1, 1), (300, 211, 20)):
        a = (rng.normal(0, 100, (H, W)) * (rng.random((H, W)) < 0.7)).astype(np.float32)
        b = rng.random((H, W)) < 0.5
        for x in (a, b.astype(np.float32), (a * b).astype(np.float32)):
            ref = _box.__wrapped__(x, r) if hasattr(_box, "__wrapped__") else None
            p = np.pad(x.astype(np.float64), r + 1, mode="edge")
            c = p.cumsum(0).cumsum(1)
            k = 2 * r + 1
            s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
            ref = (s[:H, :W] / (k * k)).astype(np.float32)
            assert _same(ref, _fast.box_edge_mean(np.ascontiguousarray(x), r)), (H, W, r)
            assert _same(ref, _box(x, r)) and _same(ref, _box_mean(x, r))
        assert _same(_box(b, r), _fast.box_edge_mean(b.astype(np.float32), r))


def test_ortho_rgbw_matches_numpy():
    from orthomosaic import _fast
    rng = np.random.default_rng(12)
    for h, w in ((31, 47), (40, 40), (3, 90), (1, 1)):
        rgb = rng.integers(0, 256, (h, w, 3)).astype(np.uint8)
        for gg, bb, use_m in ((None, None, False), ((1.07, 0.93, 1.21), None, True), ((0.8, 1.3, 1.0), (-4.2, 7.9, 0.4), True)):
            m = rng.random((h, w)) < 0.8 if use_m else None
            ref = rgb.astype(np.float32)
            if gg is not None:
                ref *= np.asarray(gg, np.float32)
            if bb is not None:
                ref += np.asarray(bb, np.float32)
            fy = np.minimum(np.arange(h) + 0.5, h - 0.5 - np.arange(h)) / (0.5 * min(h, w))
            fx = np.minimum(np.arange(w) + 0.5, w - 0.5 - np.arange(w)) / (0.5 * min(h, w))
            feather = np.clip(np.minimum(fy[:, None], fx[None, :]), 0, 1).astype(np.float32)
            if m is not None:
                feather[~m] = 0.0
            ref = np.dstack([ref, feather])
            assert _same(ref, _fast.ortho_rgbw(rgb, gg, bb, m, False))
            ref8 = np.clip(ref * np.array([1, 1, 1, 255], np.float32) + 0.5, 0, 255).astype(np.uint8)
            assert _same(ref8, _fast.ortho_rgbw(rgb, gg, bb, m, True))


def test_depth_view_matches_numpy():
    from orthomosaic import _fast
    from orthomosaic.densify import _norm
    rng = np.random.default_rng(13)
    for h, w in ((29, 41), (2, 2)):
        rgb = rng.integers(0, 256, (h, w, 3)).astype(np.uint8)
        for gg, bb, use_m in ((None, None, False), ((1.07, 0.93, 1.21), (-3.5, 2.25, 0.0), True)):
            m = rng.random((h, w)) < 0.7 if use_m else None
            ref = rgb.astype(np.float32)
            if gg is not None:
                ref *= np.asarray(gg, np.float32)
            if bb is not None:
                ref += np.asarray(bb, np.float32)
            grey = np.ascontiguousarray(_norm(ref.mean(-1)))
            if m is not None:
                grey[~m] = np.nan
            rgb8 = np.clip(ref + 0.5, 0, 255).astype(np.uint8)
            g1, r1 = _fast.depth_view(rgb, gg, bb, m)
            assert _same(grey, g1) and _same(rgb8, r1)


def test_depth_helpers_match_numpy():
    from orthomosaic import _fast
    rng = np.random.default_rng(14)
    for H, W in ((37, 53), (3, 2), (120, 90)):
        base = rng.uniform(0.01, 0.05, (H, W)).astype(np.float32)
        step = rng.uniform(1e-5, 1e-3, (H, W)).astype(np.float32)
        for i in (0, 1, 7, 191):
            assert _same((base + i * step).astype(np.float32), _fast.hyp_linear(base, step, i))
        iu = rng.uniform(-1e-3, 0.05, (H, W)).astype(np.float32)
        iu[0, 0] = np.nan
        for t in (-3, -1, 0, 2, 3):
            assert _same(np.maximum(iu + t * step, 1e-6).astype(np.float32), _fast.hyp_refine(iu, step, t))
        a = rng.normal(0, 1, (H, W)).astype(np.float32)
        a[a > 1.5] = -0.0
        for x in (a, a * a):
            for r in (0, 1, 5, 7):
                k = 2 * r + 1
                c = np.cumsum(np.pad(x, ((r + 1, r), (0, 0))), 0)
                a1 = c[k:] - c[:-k]
                c = np.cumsum(np.pad(a1, ((0, 0), (r + 1, r))), 1)
                ref = (c[:, k:] - c[:, :-k]) * np.float32(1.0 / (k * k))
                assert _same(ref, _fast.box_zero_mean(np.ascontiguousarray(x), r)), (H, W, r)


def _sgm_reference(cost, P1, P2):
    """The original (D, H, W) SGM loop, in NumPy-free Python over a small volume."""
    D, H, W = cost.shape
    agg = np.zeros((D, H, W), np.float32)
    L = np.zeros((D, H, W), np.float32)
    P1, P2 = np.float32(P1), np.float32(P2)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        for yy in range(H):
            y = yy if dy >= 0 else H - 1 - yy
            for xx in range(W):
                x = xx if dx >= 0 else W - 1 - xx
                py, px = y - dy, x - dx
                if py < 0 or py >= H or px < 0 or px >= W:
                    L[:, y, x] = cost[:, y, x]
                else:
                    prv = L[:, py, px].copy()
                    best = prv[0]
                    for d in range(1, D):
                        if prv[d] < best:
                            best = prv[d]
                    for d in range(D):
                        v = prv[d]
                        if d > 0 and prv[d - 1] + P1 < v:
                            v = prv[d - 1] + P1
                        if d < D - 1 and prv[d + 1] + P1 < v:
                            v = prv[d + 1] + P1
                        if best + P2 < v:
                            v = best + P2
                        L[d, y, x] = np.float32(np.float32(cost[d, y, x] + v) - best)
                agg[:, y, x] += L[:, y, x]
    return agg


def test_sgm_layout_and_threads_bit_identical():
    from orthomosaic import _mvs
    rng = np.random.default_rng(15)
    for D, H, W in ((7, 9, 11), (1, 4, 3), (12, 1, 6), (5, 6, 1)):
        cost = (1.0 + rng.uniform(0, 0.5, (D, H, W))).astype(np.float32)
        cost[rng.random(cost.shape) < 0.05] = 0.0
        cost[0, 0, 0] = np.nan                          # NaN takes the same comparison paths
        ref = _sgm_reference(cost, 0.05, 0.6)
        for w in (1, 3, 8):
            assert _same(ref, _mvs.sgm(cost, 0.05, 0.6, w)), (D, H, W, w)
    big = (1.0 + rng.uniform(0, 0.5, (40, 48, 48))).astype(np.float32)
    assert _same(_mvs.sgm(big, 0.05, 0.6, 1), _mvs.sgm(big, 0.05, 0.6, 8))


def test_knn_radius_matches_ckdtree():
    """sfm's track extension without SciPy: same neighbours as cKDTree (compared when installed)."""
    from orthomosaic import _fast
    rng = np.random.default_rng(16)
    pts = np.column_stack([rng.uniform(0, 4000, 20000), rng.uniform(0, 3000, 20000)])
    pts[:50] = pts[50:100]                                    # duplicate keypoints: index tie-break
    q = np.column_stack([rng.uniform(-20, 4020, 5000), rng.uniform(-20, 3020, 5000)])
    q[:20] = pts[:20]
    d, i = _fast.knn_radius(pts, q, 4, 7.5)
    assert d.shape == (5000, 4) and i.dtype == np.int64
    assert np.all(np.diff(np.where(np.isfinite(d), d, 1e300), axis=1) >= 0)
    assert ((i == len(pts)) == ~np.isfinite(d)).all()
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return
    d0, i0 = cKDTree(pts).query(q, k=4, distance_upper_bound=7.5)
    assert np.array_equal(np.isfinite(d0), np.isfinite(d)) and np.allclose(d0[np.isfinite(d0)], d[np.isfinite(d)])
    tie = (np.isin(i0, np.arange(100)) | np.isin(i, np.arange(100))).any(1)   # duplicated keypoints
    assert (i0 == i)[~tie].all()                    # exact distance ties may be listed in another order


def test_reconstruct_without_scipy():
    """3D track extension must not require SciPy (it is not a dependency)."""
    import builtins
    from orthomosaic import sfm
    real = builtins.__import__

    def no_scipy(name, *a, **k):
        if name.startswith("scipy"):
            raise ImportError("blocked")
        return real(name, *a, **k)
    R, C, X, intr, pp, cg, oc, op, uv, ptr, pw, pt = _ba_problem(seed=3, N=10, P=300, G=1)
    from orthomosaic.camera import Intrinsics
    from orthomosaic.features import Features
    rec = sfm.Reconstruction(list(range(10)), R, np.column_stack([C[:, :2], np.full(10, 60.0)]),
                             [Intrinsics(1200, 900, 1000.0)], cg, X, np.zeros((len(X), 3), np.uint8), oc, op, uv,
                             obs_feat=np.zeros(len(oc), np.int64))
    feats = [Features(np.random.default_rng(k).uniform(0, 1000, (400, 2)), np.zeros((400, 32), np.uint8),
                      np.zeros((400, 3), np.float32)) for k in range(10)]
    builtins.__import__ = no_scipy
    try:
        n = sfm._extend_tracks(rec, feats, 6.0)
    finally:
        builtins.__import__ = real
    assert n >= 0


def test_box_sat_matches_numpy():
    from orthomosaic.backend import _box_cumsum
    rng = np.random.default_rng(73)
    for shape in [(7, 11), (2, 3, 19, 13), (1, 1), (0, 5), (2, 0, 5), (0, 3, 5)]:
        for dtype in (np.float32, np.float64):
            a = rng.normal(0, 100, shape).astype(dtype)
            if a.size > 10:
                a.flat[0] = -0.0
                a.flat[-2] = np.nan
            for r in (0, 1, 5, 23):
                k = 2 * r + 1
                padded = np.pad(a.astype(np.float64), [(0, 0)] * (a.ndim - 2) + [(r + 1, r)] * 2)
                c = padded.cumsum(axis=-2).cumsum(axis=-1)
                ref = ((c[..., k:, k:] - c[..., :-k, k:] - c[..., k:, :-k] + c[..., :-k, :-k]) /
                       (k * k)).astype(np.float32)
                got = _box_cumsum(np, a, r)
                assert _same(ref, got), (shape, dtype, r)
    a = rng.normal(size=(3, 40, 50)).astype(np.float32)[:, ::2, ::-2]
    a.flags.writeable = False
    assert _same(_box_cumsum(np, a, 3), _box_cumsum(np, a.copy(), 3))


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
