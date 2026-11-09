#changed here: new module -- Pix4D-style occlusion-aware true orthophoto.
"""Pix4D-style true orthophoto: occlusion-aware orthorectification of the final DSM.

Pix4D does not render the orthomosaic during the dense matching; it first builds the DSM and
only then orthorectifies, and while doing so it

  * keeps, for every DSM cell, only the views that actually *see* that cell (occlusion
    handling), so a building no longer smears its wall over the ground behind it;
  * favours the most nadir view (view-angle weighting), so the texture is sampled with the
    least off-nadir stretch;
  * feathers across the view seams (distance to the image border).

The colours produced inside :func:`orthomosaic.mvs.dense_reconstruct` already sample the
surface at its own height, but they blend views that are geometrically hidden and weight
oblique views the same as nadir ones.  This module reproduces the Pix4D render stage on an
already computed DSM and can therefore test occlusion against the *whole* surface, not just
the current matching tile.
"""
from __future__ import annotations

import logging
import math
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .imageio import load_rgb, load_thermal
try:
    from . import _dense
except ImportError:                       # source tree without the compiled kernel
    _dense = None
from .masks import load_mask
from .mvs import _View
from .render import ImageCache

log = logging.getLogger(__name__)


def _visibility(dsm, minX, maxY, gsd, C, X, Y, Z, max_steps, tol, zmax=None):
    """Visibility mask (True = visible) of surface points (X, Y, Z) from camera centre `C`.

    Height-field line of sight: walk from the point towards the camera, but only over the
    stretch where the ray is still below the highest surface (`zmax`) -- above that nothing can
    block it. Samples are one DSM cell apart (coarser only if a ray needs more than `max_steps`),
    so a wall or roof edge right next to the point is not skipped. The point is hidden when the
    DSM rises more than `tol` metres above the ray. Heights, not depths, are compared, so there is
    no camera-axis-depth vs range ambiguity.

    Limits (2.5D): overhangs, bridges and multi-layer structures are not representable; facades
    are the steep cells between roof and ground, so views grazing a facade are judged coarsely.
    """
    H, W = dsm.shape
    X, Y, Z = (np.asarray(a, np.float64) for a in (X, Y, Z))
    vis = np.ones(np.shape(Z), bool)
    if zmax is None:
        zmax = float(np.nanmax(dsm)) if np.isfinite(dsm).any() else None
    if zmax is None:
        return vis
    if _dense is not None:                       # C kernel: per-ray early exit, releases the GIL
        out = np.empty(vis.size, np.uint8)
        _dense.visibility(np.ascontiguousarray(dsm, np.float32), float(minX), float(maxY), float(gsd),
                          float(C[0]), float(C[1]), float(C[2]),
                          np.ascontiguousarray(np.ravel(X), np.float64), np.ascontiguousarray(np.ravel(Y), np.float64),
                          np.ascontiguousarray(np.ravel(Z), np.float64), int(max_steps), float(tol), float(zmax), out)
        return out.reshape(vis.shape).astype(bool)
    ux, uy = C[0] - X, C[1] - Y
    dh = np.hypot(ux, uy)
    rise = (C[2] - Z) / np.maximum(dh, 1e-6)          # ray height gain per horizontal metre
    ux, uy = ux / np.maximum(dh, 1e-6), uy / np.maximum(dh, 1e-6)
    dmax = np.minimum(np.maximum(zmax + tol - Z, 0.0) / np.maximum(rise, 1e-6), dh)
    step = np.maximum(gsd, dmax / max(1, int(max_steps)))
    n = int(np.ceil(np.max(dmax / step))) if dmax.size else 0
    for k in range(1, n + 1):
        d = k * step
        act = d <= dmax
        if not act.any():
            break
        col = ((X + ux * d - minX) / gsd).astype(np.int64)
        row = ((maxY - (Y + uy * d)) / gsd).astype(np.int64)
        inb = act & (col >= 0) & (col < W) & (row >= 0) & (row < H)
        zz = dsm[np.clip(row, 0, H - 1), np.clip(col, 0, W - 1)]
        vis &= ~(inb & np.isfinite(zz) & (zz > Z + rise * d + tol))
    return vis


try:
    from . import _fast
except ImportError:                       # source tree without the compiled kernel
    _fast = None


def lens_r2max(it) -> float:
    """Largest squared undistorted radius (normalised) a point may have to be in the photo.

    The radial polynomial r(1 + k1 r^2 + k2 r^4 + k3 r^6) only describes the lens inside the
    frame; past its maximum it bends back, so ground far outside the frame (30 m off on the M4T
    thermal, k3 = -0.38) projects *into* the image and is painted as long streaks. Limit: the
    radius of the farthest image corner, on the rising branch of the polynomial."""
    rd = math.hypot(max(it.cx, it.width - 1 - it.cx), max(it.cy, it.height - 1 - it.cy)) / it.f
    r = np.linspace(0.0, 4.0 * rd + 1.0, 8001)
    r2 = r * r
    d = r * (1 + r2 * (it.k1 + r2 * (it.k2 + r2 * it.k3)))
    rising = np.flatnonzero(np.diff(d) <= 0)
    stop = rising[0] if rising.size else len(r) - 1        # first fold of the polynomial
    reach = np.flatnonzero(d[:stop + 1] >= rd)
    lim = r[reach[0]] * 1.01 if reach.size else r[stop]
    return float(min(lim, r[stop]) ** 2)


def _outside_lens(R, C, r2max, X0, Y0, gsd, Z):
    """(h, w) bool: cells of a tile whose undistorted radius in camera (R, C) exceeds `r2max`
    (or that lie behind it)."""
    h, w = Z.shape
    x = X0 + (np.arange(w) + 0.5) * gsd - C[0]
    y = Y0 - (np.arange(h) + 0.5) * gsd - C[1]
    z = Z.astype(np.float64) - C[2]
    a = [R[i, 0] * x[None, :] + R[i, 1] * y[:, None] + R[i, 2] * z for i in range(3)]
    return (a[2] <= 1e-6) | (a[0] * a[0] + a[1] * a[1] > r2max * a[2] * a[2])


def _box(a, r):
    """Mean over a (2r+1)^2 window (edge-clamped), via cumulative sums."""
    if r <= 0:
        return a.astype(np.float32)
    if _fast is not None and a.ndim == 2 and a.dtype in (np.float32, np.bool_):
        # same float64 summed-area table, compiled (exact for these input types)
        return _fast.box_edge_mean(np.ascontiguousarray(a, np.float32), int(r))
    p = np.pad(a.astype(np.float64), r + 1, mode="edge")
    c = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return (s[:a.shape[0], :a.shape[1]] / (k * k)).astype(np.float32)


def _seam_labels(cost, cols, smooth, iters):
    """Spatially coherent source labels by ICM on a 4-connected MRF.

    E(l) = sum_p D_p(l_p) + smooth * sum_{p~q, l_p != l_q} (0.1 + |I_{l_p}(p) - I_{l_q}(p)| / 64)

    The data term D is the (normalised) unsuitability of a source; the pair term charges a seam
    in proportion to how much the two sources *disagree* there, so seams avoid buildings, cars
    and misaligned edges and run through places where neighbouring images look alike.
    ICM starts from the best-score labelling and only lowers the energy (a local optimum --
    a graph-cut solver would find a stronger one, at the cost of a new dependency).
    """
    L = cost.shape[0]
    lab = np.argmin(cost, 0)
    if L < 2 or smooth <= 0:
        return lab
    if _dense is not None:                       # C kernel, same energy and update rule
        lab = np.ascontiguousarray(lab, np.int32)
        _dense.seam_icm(np.ascontiguousarray(cost, np.float32), np.ascontiguousarray(cols, np.float32),
                        lab, float(smooth), int(iters))
        return lab
    for _ in range(iters):
        tot = cost.copy()
        for axis, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
            nb = np.roll(lab, sh, axis)
            edge = np.ones(lab.shape, bool)       # np.roll wraps: ignore the wrapped border line
            if axis == 0:
                edge[0 if sh == 1 else -1, :] = False
            else:
                edge[:, 0 if sh == 1 else -1] = False
            cnb = np.take_along_axis(cols, nb[None, :, :, None], 0)[0]   # neighbour's source at p
            for l in range(L):
                diff = np.abs(cols[l] - cnb).mean(-1)
                tot[l] += smooth * edge * (nb != l) * (0.1 + diff / 64.0)
        new = np.argmin(tot, 0)
        if (new == lab).all():
            break
        lab = new
    return lab


def _global_sources(rec, dsm, minX, maxY, gsd, use_occ, occ_steps, occ_tol, zmax, workers,
                    cell_m=0.25, K=6, lam=0.25, iters=12):
    """One source photo per ~`cell_m` region for the whole mosaic (ODM/Pix4D select views
    globally, not per tile, so neighbouring tiles never disagree).

    Every camera is scored on a coarse grid of DSM points: near-vertical rays first (tall objects
    lean least, so seams between photos do not cut them), then distance to the image border.
    Points the camera cannot see (line of sight over the DSM) cost +10, so they are used only when
    no camera sees them. Each cell keeps its K best cameras; Potts smoothing over the whole grid
    (energy cost + lam * #differing 4-neighbours, a few sweeps) turns them into large coherent
    regions. Returns (labels (gh, gw) int32 index into rec.used, -1 = none; step in cells)."""
    H, W = dsm.shape
    s = max(1, int(round(cell_m / gsd)))
    ys = np.minimum(np.arange(-(-H // s)) * s + s // 2, H - 1)
    xs = np.minimum(np.arange(-(-W // s)) * s + s // 2, W - 1)
    gh, gw = len(ys), len(xs)
    Zg = dsm[ys][:, xs]
    ok = np.isfinite(Zg)
    gy, gx = np.nonzero(ok)
    P = np.column_stack([minX + (xs[gx] + 0.5) * gsd, maxY - (ys[gy] + 0.5) * gsd, Zg[gy, gx]]).astype(np.float64)
    n = len(P)
    labs = np.full((gh, gw), -1, np.int32)
    if n == 0:
        return labs, s
    dsm_c = np.ascontiguousarray(dsm, np.float32)

    def score_cam(k):
        it = rec.intr[rec.cam_group[k]]
        xc = (P - rec.C[k]) @ rec.R[k].T
        z = xc[:, 2]
        front = z > 1e-6
        nrm = xc[:, :2] / np.where(front, z, 1.0)[:, None]
        r2 = (nrm * nrm).sum(1)
        front &= r2 < lens_r2max(it)
        uv = (it.f * (1 + r2 * (it.k1 + r2 * (it.k2 + r2 * it.k3))))[:, None] * nrm + (it.cx, it.cy)
        bu = np.minimum(uv[:, 0], it.width - 1 - uv[:, 0]) / (0.5 * it.width)
        bv = np.minimum(uv[:, 1], it.height - 1 - uv[:, 1]) / (0.5 * it.height)
        border = np.minimum(bu, bv)
        idx = np.flatnonzero(front & (border > 0.01))
        if idx.size == 0:
            return k, idx, None
        ray = rec.C[k] - P[idx]
        vert = ray[:, 2] / np.linalg.norm(ray, axis=1)          # cos of the ray's angle to vertical
        cost = (1.0 - vert ** 4) + 0.3 * (1.0 - np.clip(border[idx] / 0.3, 0, 1))
        if use_occ:
            vis = _visibility(dsm_c, minX, maxY, gsd, rec.C[k], P[idx, 0], P[idx, 1], P[idx, 2],
                              occ_steps, occ_tol, zmax)
            cost = np.where(vis, cost, cost + 10.0)
        return k, idx, cost.astype(np.float32)

    best_c = np.full((K, n), np.inf, np.float32)
    best_l = np.full((K, n), -1, np.int32)
    with ThreadPoolExecutor(max(1, workers)) as ex:
        for k, idx, cost in ex.map(score_cam, range(len(rec.used))):
            if cost is None:
                continue
            c = np.concatenate([best_c[:, idx], cost[None]])
            l = np.concatenate([best_l[:, idx], np.full((1, idx.size), k, np.int32)])
            o = np.argsort(c, 0)[:K]
            best_c[:, idx] = np.take_along_axis(c, o, 0)
            best_l[:, idx] = np.take_along_axis(l, o, 0)
    C = np.full((K, gh, gw), np.inf, np.float32)
    Lb = np.full((K, gh, gw), -1, np.int32)
    C[:, gy, gx] = best_c
    Lb[:, gy, gx] = best_l
    lab = Lb[0].copy()
    for _ in range(iters):
        pad = np.pad(lab, 1, constant_values=-2)
        nb = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
        same = (Lb[:, None] == nb[None]).sum(1)
        E = np.where(Lb >= 0, C + lam * (4 - same), np.inf)
        new = np.take_along_axis(Lb, np.argmin(E, 0)[None], 0)[0]
        if (new == lab).all():
            break
        lab = new
    return lab.astype(np.int32), s


class _Defaults:
    #changed here: fallback option object so true_orthophoto works without an Options3D.
    ortho_tile = 512
    max_views = 6
    view_angle_power = 1.5
    occlusion = True
    occlusion_tol = 0.20
    occlusion_steps = 96
    occlusion_stride = 2
    cache_mb = 1024
    workers = 0
    ortho_blend = "seam"
    ortho_views = 8          # max photos per tile (taken from the global source map)
    fill_hidden = True       # colour cells hidden in every chosen view from the best view anyway
    source_weights = (1.0, 0.5, 0.7, 0.3)   # angle, resolution, border, exposure
    seam_smoothness = 0.6
    seam_iters = 4
    seam_band = 3


def true_orthophoto(ar, rec, dsm, minX, maxY, gsd, gains=None, biases=None, opt=None):
    #changed here: whole function -- the Pix4D "Orthomosaic" render stage on the final DSM.
    """Render `dsm` as an occlusion-aware true orthophoto.

    `minX`/`maxY` are the world coordinates of the top-left DSM cell *corner*; `dsm` is the
    final surface (NaN = no data). Inverse mapping: every cell centre (X, Y, DSM(X, Y)) is
    projected into the candidate views with the full camera model (distortion applied once, on
    the original images), sampled bilinearly, and kept only where the view is valid and the
    point is visible (height-field line of sight).

    Source selection (``ortho_blend="seam"``): each visible view gets a normalised score
    ``w_angle*cos(ray, normal)^p + w_res*res + w_border*border + w_exposure*unclipped`` (weights in
    ``source_weights``); labels are made spatially coherent by :func:`_seam_labels`, then blended
    only within ``seam_band`` cells of a seam. ``ortho_blend="feather"`` keeps the old weighted
    average of all visible views (more ghosting on tall objects).

    Returns ``(rgba (H, W, 4) uint8, source (H, W) int32, count (H, W) uint8, tmp_dir)``, all
    memory-mapped from files in `tmp_dir` (the caller removes it): alpha is 255 where a photo
    coloured the cell, `source` the index into ``rec.used`` of the view chosen per cell (-1 =
    unobserved) and `count` the number of views that see the cell.

    Radiometric thermal (``ar.thermal_range`` set): frames are sampled and blended as float32 gray
    (load_thermal), and the unquantised result is also written to ``<tmp_dir>/value.npy``
    ((H, W) float32 gray levels, NaN = no data) for the °C raster.
    """
    opt = opt if opt is not None else _Defaults()
    g = lambda name: getattr(opt, name, getattr(_Defaults, name))
    backend = ar.backend
    if not backend.parallel_blocks:
        # On a GPU the per-tile work is still mostly CPU (scoring, visibility, seams, blending),
        # and GPU backends process tiles one at a time, so the render ran on one core (CUDA: the
        # GPU ~3% busy for 22 min on a 1117-photo flight). Use the C sampler with all cores instead.
        from .backend import CPUBackend
        backend = CPUBackend()
    frames = ar.frames
    dsm = np.ascontiguousarray(dsm, np.float32)
    H, W = dsm.shape
    covered = np.isfinite(dsm)
    # outputs live on disk (memory-mapped) so the orthophoto can be rendered at the native GSD on
    # an 8 GB machine; the caller writes them out and removes the folder (returned last)
    tmp = tempfile.mkdtemp(prefix="ortho_", dir=getattr(opt, "work_dir", None))
    rgba_out = np.lib.format.open_memmap(os.path.join(tmp, "rgba.npy"), "w+", np.uint8, (H, W, 4))
    SRC = np.lib.format.open_memmap(os.path.join(tmp, "src.npy"), "w+", np.int32, (H, W))
    SRC[:] = -1                                               # index into rec.used, -1 = none
    CNT = np.lib.format.open_memmap(os.path.join(tmp, "cnt.npy"), "w+", np.uint8, (H, W))
    thermal = getattr(ar, "thermal_range", None) is not None
    VAL = None
    if thermal:
        VAL = np.lib.format.open_memmap(os.path.join(tmp, "value.npy"), "w+", np.float32, (H, W))
        VAL[:] = np.nan
    if not covered.any() or len(rec.used) == 0:
        return rgba_out, SRC, CNT, tmp

    gains = gains or {}
    biases = biases or {}
    zfill = float(np.nanmedian(dsm))

    # Load images at (about) the orthophoto resolution, not full resolution: sampling a 4032 px
    # image for a 6 cm ortho oversamples ~5x and, at 36 MB/image, blows past the cache so images
    # are re-decoded hundreds of times. Matching the load scale to the output GSD keeps every
    # image in cache (one decode each) with no loss of ortho detail.
    native_gsd = float(getattr(getattr(ar, "alignment", None), "gsd", gsd) or gsd)
    img_scale = float(np.clip(native_gsd / gsd, 0.1, 1.0))

    def load(k):
        i = rec.used[k]
        fr = frames[i]
        if _fast is not None and not thermal:
            # one compiled pass (same rounding as the NumPy steps below)
            rgb = load_rgb(fr, img_scale)
            h, w = rgb.shape[:2]
            rgb = _fast.ortho_rgbw(np.ascontiguousarray(rgb, np.uint8), gains.get(i), biases.get(i),
                                   load_mask(fr.path, w, h), u8)
            it = rec.intr[rec.cam_group[k]]
            sx = rgb.shape[1] / fr.width
            return _View([], rgb if u8 else backend.upload(rgb),
                         np.ascontiguousarray(rec.R[k]), np.ascontiguousarray(rec.C[k]),
                         it.f * sx, it.k1, it.k2, (it.cx + 0.5) * sx - 0.5, (it.cy + 0.5) * sx - 0.5, it.k3)
        rgb = (np.repeat(load_thermal(fr, img_scale)[..., None], 3, axis=2) if thermal
               else load_rgb(fr, img_scale).astype(np.float32))
        gg = gains.get(i)
        if gg is not None:
            rgb *= np.asarray(gg, np.float32)
        bb = biases.get(i)
        if bb is not None:
            rgb += np.asarray(bb, np.float32)
        h, w = rgb.shape[:2]
        # 4th channel: feather weight (distance to the image border) -> smooth view seams
        fy = np.minimum(np.arange(h) + 0.5, h - 0.5 - np.arange(h)) / (0.5 * min(h, w))
        fx = np.minimum(np.arange(w) + 0.5, w - 0.5 - np.arange(w)) / (0.5 * min(h, w))
        feather = np.clip(np.minimum(fy[:, None], fx[None, :]), 0, 1).astype(np.float32)
        m = load_mask(fr.path, w, h)
        if m is not None:
            feather[~m] = 0.0                    # masked pixels: weight 0 = not a valid sample
        rgb = np.dstack([rgb, feather])
        if u8:
            # 8-bit cache (colour 0-255, feather 0-255): 4x more photos fit in the same RAM
            rgb = np.clip(rgb * np.array([1, 1, 1, 255], np.float32) + 0.5, 0, 255).astype(np.uint8)
        it = rec.intr[rec.cam_group[k]]
        sx = rgb.shape[1] / fr.width
        return _View([], np.ascontiguousarray(rgb) if u8 else backend.upload(np.ascontiguousarray(rgb)),
                     np.ascontiguousarray(rec.R[k]), np.ascontiguousarray(rec.C[k]),
                     it.f * sx, it.k1, it.k2, (it.cx + 0.5) * sx - 0.5, (it.cy + 0.5) * sx - 0.5, it.k3)

    # thermal stays float32 end to end (an 8-bit cache would quantise the measured values)
    u8 = backend.name == "cpu" and _dense is not None and hasattr(_dense, "sample_view_u8") and not thermal
    cache = ImageCache(load, lambda v: v.nbytes, int(g("cache_mb")) * 1024 * 1024)
    r2max = [lens_r2max(it) for it in rec.intr]          # past this radius the lens model folds back

    def sample(v, X0, Y0, Ztile):
        if not u8:
            rv, vv = backend.sample_view(v.rgb, v.cam(0), X0, Y0, gsd, Ztile)
            return backend.to_numpy(rv), backend.to_numpy(vv)
        R, C, f, k1, k2, k3, cx, cy = v.cam(0)
        out = np.zeros(Ztile.shape + (4,), np.float32)     # cells a photo misses stay 0, never garbage
        val = np.zeros(Ztile.shape, np.uint8)
        _dense.sample_view_u8(v.rgb, np.ascontiguousarray(R, np.float64), np.ascontiguousarray(C, np.float64),
                              float(f), float(k1), float(k2), float(k3), float(cx), float(cy), float(X0), float(Y0), float(gsd),
                              np.ascontiguousarray(Ztile, np.float32), out, val)
        out[..., 3] *= 1.0 / 255.0
        return out, val

    T = int(g("ortho_tile"))
    max_views = int(g("max_views"))
    va_power = float(g("view_angle_power"))
    use_occ = bool(g("occlusion"))
    occ_tol = float(g("occlusion_tol"))
    occ_steps = int(g("occlusion_steps"))
    occ_stride = max(1, int(g("occlusion_stride")))
    blend = str(g("ortho_blend"))
    n_views = int(g("ortho_views"))
    fill_hidden = bool(g("fill_hidden"))
    w_ang, w_res, w_bord, w_exp = (float(x) for x in g("source_weights"))
    smooth = float(g("seam_smoothness"))
    s_iters = int(g("seam_iters"))
    band = max(0, int(g("seam_band")))
    # two-band blending (Burt & Adelson): colour/exposure (low band) is blended over ~0.25 m across
    # a seam, detail (high band) over `band` cells, so lighting steps between photos vanish while
    # edges stay sharp. The tile margin covers the wide band, so tiles blend identically.
    lo_r = max(band, int(round(0.25 / gsd)))
    pad = 2 * lo_r + 8 if blend == "seam" else 0
    zmax = float(np.nanmax(dsm))

    dsm_c = np.ascontiguousarray(dsm, np.float32)       # one contiguous copy for the C visibility kernel

    def normals(y0, y1, x0, x1):
        """Surface normals of a DSM window (computed per tile: a full-raster normal map costs
        12 bytes/cell). Steep cells are facades/edge ramps of the 2.5D surface: scoring them
        against their tilted normal would favour oblique views that look *at* the wall, so the
        vertical is used there."""
        a0, a1 = max(0, y0 - 1), min(H, y1 + 1)
        b0, b1 = max(0, x0 - 1), min(W, x1 + 1)
        win = np.where(covered[a0:a1, b0:b1], dsm[a0:a1, b0:b1], zfill).astype(np.float32)
        gy_, gx_ = np.gradient(win, gsd) if min(win.shape) > 1 else (np.zeros_like(win), np.zeros_like(win))
        n = np.dstack([gx_, -gy_, -np.ones_like(gx_)])
        n /= -np.linalg.norm(n, axis=2, keepdims=True)
        n[n[..., 2] < np.cos(np.radians(30))] = (0.0, 0.0, 1.0)
        return n[y0 - a0:y0 - a0 + (y1 - y0), x0 - b0:x0 - b0 + (x1 - x0)]

    # global source map: which photo colours which region of the whole mosaic
    t_g = time.time()
    workers = int(g("workers")) or getattr(ar, "workers", 1)
    glab, gstep = _global_sources(rec, dsm_c, minX, maxY, gsd, use_occ, occ_steps, occ_tol, zmax, workers)
    gl = glab[glab >= 0]
    log.info("True orthophoto: global source selection, %d photos over %d grid cells of %.2f m (%.1fs)",
             len(np.unique(gl)), gl.size, gstep * gsd, time.time() - t_g)

    def _vis_map(C, sub, X0, Y0, py0, px0, vis_h, vis_w):
        """Occlusion test on every `occ_stride`-th cell of the tile; a coarse cell is visible only
        if it and its 4-neighbours are."""
        sr, scc = np.nonzero(sub[::occ_stride, ::occ_stride])
        vis_map = np.ones((vis_h, vis_w), bool)
        if sr.size:
            sr_, sc_ = sr * occ_stride, scc * occ_stride
            vis_map[sr, scc] = _visibility(dsm_c, minX, maxY, gsd, C, X0 + (sc_ + 0.5) * gsd,
                                           Y0 - (sr_ + 0.5) * gsd, dsm[py0 + sr_, px0 + sc_],
                                           occ_steps, occ_tol, zmax)
            if occ_stride > 1:   # a coarse cell is visible only if it and its 4-neighbours are
                vm = vis_map.copy()
                vm[1:] &= vis_map[:-1]; vm[:-1] &= vis_map[1:]
                vm[:, 1:] &= vis_map[:, :-1]; vm[:, :-1] &= vis_map[:, 1:]
                vis_map = vm
        return vis_map

    def process(tile):
        tx, ty = tile
        tw, th = min(T, W - tx), min(T, H - ty)
        if not covered[ty:ty + th, tx:tx + tw].any():
            return None
        px0, py0 = max(0, tx - pad), max(0, ty - pad)
        px1, py1 = min(W, tx + tw + pad), min(H, ty + th + pad)
        pw, ph = px1 - px0, py1 - py0
        sub = covered[py0:py1, px0:px1]
        X0 = minX + px0 * gsd
        Y0 = maxY - py0 * gsd
        Ztile = np.where(sub, dsm[py0:py1, px0:px1], zfill).astype(np.float32)

        # candidate views: the photos the global source map assigns to this tile (and its context
        # margin), most frequent first; tiles therefore agree with their neighbours
        ys_g = np.arange(py0, py1) // gstep
        xs_g = np.arange(px0, px1) // gstep
        g_tile = glab[ys_g][:, xs_g]
        lv, lc = np.unique(g_tile[(g_tile >= 0) & sub], return_counts=True)
        if lv.size == 0:
            return None
        cand = [int(k) for k in lv[np.argsort(-lc)][:max(n_views, 1)]]
        views = [cache.get(k) for k in cand]

        rr, cc = np.nonzero(sub)
        L = len(cand)
        cols = np.zeros((L, ph, pw, 3), np.float32)
        score = np.full((L, ph, pw), -1.0, np.float32)            # -1 = not usable
        score_nv = np.full((L, ph, pw), -1.0, np.float32)         # same, ignoring occlusion (hidden fill)
        inv_depth = np.zeros((L, rr.size), np.float32)
        vis_h, vis_w = -(-ph // occ_stride), -(-pw // occ_stride)
        if _fast is not None:
            # compiled per-view scoring (same arithmetic as the NumPy branch below)
            sub8 = np.ascontiguousarray(sub).view(np.uint8)
            nwin = np.ascontiguousarray(normals(py0, py1, px0, px1), np.float32)
            for j, (k, v) in enumerate(zip(cand, views)):
                rgbw, vv = sample(v, X0, Y0, Ztile)
                vv = np.where(_outside_lens(rec.R[k], rec.C[k], r2max[rec.cam_group[k]], X0, Y0, gsd, Ztile), 0, vv)
                C = rec.C[k].astype(np.float64)
                vis_map = _vis_map(C, sub, X0, Y0, py0, px0, vis_h, vis_w) if use_occ else None
                cos_n = _fast.ortho_view_cos(sub8, dsm, py0, px0, float(X0), float(Y0), float(gsd), nwin,
                                             float(C[0]), float(C[1]), float(C[2]))
                Rz = np.asarray(rec.R[k][2], np.float64)
                _fast.ortho_view_fill(sub8, dsm, py0, px0, float(X0), float(Y0), float(gsd),
                                      np.ascontiguousarray(rgbw, np.float32), np.ascontiguousarray(vv, np.uint8),
                                      (vis_map if use_occ else np.ones((1, 1), bool)).view(np.uint8), occ_stride,
                                      use_occ, float(C[0]), float(C[1]), float(C[2]), float(Rz[0]), float(Rz[1]),
                                      float(Rz[2]), np.ascontiguousarray(cos_n ** va_power, np.float64), w_ang,
                                      w_bord, w_exp, cols[j], score[j], score_nv[j], inv_depth[j])
            _fast.ortho_resolution(sub8, inv_depth, score, w_res)
        else:
            wx = X0 + (cc + 0.5) * gsd                 # map coordinates of cell centres
            wy = Y0 - (rr + 0.5) * gsd
            wz = dsm[py0 + rr, px0 + cc].astype(np.float64)
            n_c = normals(py0, py1, px0, px1)[rr, cc]
            for j, (k, v) in enumerate(zip(cand, views)):
                rgbw, vv = sample(v, X0, Y0, Ztile)
                vv = np.where(_outside_lens(rec.R[k], rec.C[k], r2max[rec.cam_group[k]], X0, Y0, gsd, Ztile), 0, vv)
                rgbw = np.nan_to_num(rgbw, nan=0.0, posinf=0.0, neginf=0.0)
                valid = (vv[rr, cc] > 0) & (rgbw[rr, cc, 3] > 0)
                C = rec.C[k].astype(np.float64)
                ray = np.column_stack([C[0] - wx, C[1] - wy, C[2] - wz])
                rng_ = np.linalg.norm(ray, axis=1) + 1e-9
                cos_n = np.clip(np.sum(ray * n_c, 1) / rng_, 0.0, 1.0)     # vs. surface normal
                if use_occ:
                    vis = _vis_map(C, sub, X0, Y0, py0, px0, vis_h, vis_w)[rr // occ_stride, cc // occ_stride]
                else:
                    vis = np.ones(rr.size, bool)
                ok = valid & vis
                px = rgbw[rr, cc, :3]
                lum = px.mean(1)
                expo = np.where((lum > 250) | (lum < 5), 0.0, 1.0)            # clipped samples
                depth = np.sum((np.column_stack([wx, wy, wz]) - C) * rec.R[k][2], 1)   # camera-axis depth
                inv_depth[j] = np.where(ok, 1.0 / np.maximum(depth, 1e-3), 0)
                s = (w_ang * cos_n ** va_power + w_bord * np.clip(rgbw[rr, cc, 3], 0, 1) + w_exp * expo)
                cols[j, rr, cc] = px
                score[j, rr, cc] = np.where(ok, s, -1.0)
                score_nv[j, rr, cc] = np.where(valid, s, -1.0)
            # resolution term: projected pixel size relative to the finest view at this cell
            best = inv_depth.max(0)
            res_t = np.where(best > 0, inv_depth / np.maximum(best, 1e-9), 0)
            for j in range(L):
                score[j, rr, cc] = np.where(score[j, rr, cc] >= 0, score[j, rr, cc] + w_res * res_t[j], -1.0)
        wsum_w = max(w_ang + w_res + w_bord + w_exp, 1e-6)
        usable = score >= 0
        count = usable.sum(0)                                     # views that *see* the cell
        if fill_hidden:
            # cells inside photos but occluded in all of them (e.g. ground at the foot of a wall):
            # colour them from the best photo anyway instead of leaving a hole (as 0.3.3 did);
            # coverage_count stays 0 there, so they remain identifiable
            fb = (count == 0) & (score_nv >= 0).any(0)
            if fb.any():
                score = np.where(fb[None], score_nv, score)
                usable = score >= 0

        if blend == "seam":
            # follow the global map: a source is preferred where the map assigns it, blurred over
            # one region so that, near a region border, the colour-aware seam step decides where
            # exactly to cut (around cars and roof units, through flat areas)
            pref = np.stack([_box(g_tile == k, gstep) for k in cand])
            cost = np.where(usable, 1.0 - score / wsum_w + 0.6 * (1.0 - pref), 1e3).astype(np.float32)
            lab = _seam_labels(cost, cols, smooth, s_iters)
            has = usable.any(0)
            # feather only inside a band around the seams; elsewhere one source per cell
            rgb = np.zeros((ph, pw, 3), np.float32)
            ws = np.zeros((ph, pw), np.float32)
            wl_sum = np.zeros((ph, pw), np.float32)
            lo_acc = np.zeros((ph, pw, 3), np.float32)
            for j in range(L):
                if not usable[j].any():
                    continue
                u = usable[j].astype(np.float32)
                own = ((lab == j) & has).astype(np.float32)
                # low band of photo j: normalised blur over the cells it covers
                den = _box(u, lo_r)
                low = np.stack([_box(cols[j, ..., c] * u, lo_r) for c in range(3)], -1) / np.maximum(den, 1e-6)[..., None]
                w_hi = _box(own, band) * u
                w_lo = _box(own, lo_r) * u
                rgb += w_hi[..., None] * (cols[j] - low)
                ws += w_hi
                lo_acc += w_lo[..., None] * low
                wl_sum += w_lo
            rgb = rgb / np.maximum(ws, 1e-6)[..., None] + lo_acc / np.maximum(wl_sum, 1e-6)[..., None]
            src = np.where(has, np.asarray(cand)[lab], -1)
        else:   # legacy weighted average of all visible views
            w = np.where(usable, np.maximum(score, 0) ** 2, 0)
            ws = w.sum(0)
            rgb = (w[..., None] * cols).sum(0) / np.maximum(ws, 1e-6)[..., None]
            src = np.where(usable.any(0), np.asarray(cand)[np.argmax(score, 0)], -1)
        oy, ox = ty - py0, tx - px0
        crop = (slice(oy, oy + th), slice(ox, ox + tw))
        cov = usable.any(0)
        if blend == "seam":
            src = np.where(cov, np.asarray(cand)[lab], -1)
        return tx, ty, th, tw, rgb[crop], (ws[crop] > 1e-6) & cov[crop], src[crop], count[crop]

    def consume(res):
        if res is None:
            return
        tx, ty, th, tw, rgb, m, src, count = res
        if not m.any():
            return
        block = rgba_out[ty:ty + th, tx:tx + tw]
        block[m, :3] = np.clip(rgb[m] + 0.5, 0, 255).astype(np.uint8)
        block[m, 3] = 255
        if VAL is not None:
            VAL[ty:ty + th, tx:tx + tw][m] = rgb[m, 0]
        SRC[ty:ty + th, tx:tx + tw][m] = src[m]
        CNT[ty:ty + th, tx:tx + tw] = np.minimum(count, 255)

    tiles = [(tx, ty) for ty in range(0, H, T) for tx in range(0, W, T)]
    tw_n = max(1, (W + T - 1) // T)
    ordered = []
    for r in range(0, len(tiles), tw_n):     # serpentine keeps neighbouring images cached
        row = tiles[r:r + tw_n]
        ordered.extend(row if (r // tw_n) % 2 == 0 else row[::-1])

    log.info("True orthophoto: rendering %d tiles", len(ordered))
    t0 = time.time()
    done = 0
    step = max(1, len(ordered) // 10)
    if backend.parallel_blocks and workers > 1:
        with ThreadPoolExecutor(workers) as ex:
            for res in ex.map(process, ordered):
                consume(res)
                done += 1
                if done % step == 0:
                    log.info("  ortho: %d/%d tiles", done, len(ordered))
    else:
        for t in ordered:
            consume(process(t))
            done += 1
            if done % step == 0:
                log.info("  ortho: %d/%d tiles", done, len(ordered))
    log.info("True orthophoto: %d/%d tiles in %.1fs (%d image loads)",
             len(ordered), len(ordered), time.time() - t0, cache.loads)
    return rgba_out, SRC, CNT, tmp
