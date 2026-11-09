"""RGB-driven thermal binding.

A DJI dual sensor (M4T, M3T, H20T, ...) carries an RGB and a thermal camera on the same gimbal,
firing together. The RGB block has strong texture and self-calibrates well; thermal is low-texture,
low-resolution and its focal length is weakly observable, so a thermal-only reconstruction drifts.

Binding solves the RGB block by aerial triangulation, then places every thermal image using its
RGB twin's pose plus a single **rig offset** (a fixed rotation + translation from the RGB camera to
the thermal camera, shared by all frames). Thermal geometry then inherits RGB-grade accuracy and,
crucially, the thermal products land in the same coordinates as the RGB ones -- exactly what a
radiometric-overlay workflow needs.

Camera model (as elsewhere):  x_cam = R (X - C).  For a rigid pair,
    x_T = R_rel x_V + t_rel   =>   R_T = R_rel R_V ,   C_T = C_V - R_V^T R_rel^T t_rel .
The rig (R_rel, t_rel) is estimated robustly from the two independently-solved pose sets.

The thermal-only SfM is too weak to measure the lever arm: its altitude trades off against its
focal length, and on flight 050 it put the thermal camera 0.83 m below the RGB one (the real
M4T lenses are a few cm apart). So by default the thermal cameras sit *at* their RGB twins'
centres, and one thermal-only adjustment then refines the thermal lens and each thermal attitude
on the thermal tie points. Pinned centres alone do not fix the focal length of a nadir block (a
focal error is absorbed by moving every tie point up or down), so the tie points are also held
to the RGB DSM's height: the scene depth is then known and the focal is measured.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

from .camera import orthonormalize

log = logging.getLogger(__name__)

# DJI dual-sensor filename: PREFIX_<seq>_<suffix>.ext, e.g. DJI_20260915182922_0001_V.JPG
_DJI = re.compile(r"^(?P<stem>.+)_(?P<suffix>[A-Za-z]+)\.(?P<ext>[^.]+)$")


def pair_key(basename: str) -> str:
    """Capture key shared by an RGB/thermal pair: the filename without the sensor suffix."""
    m = _DJI.match(basename)
    return m.group("stem") if m else basename


_SEQ = re.compile(r"^.+_(?P<seq>\d{3,6})_[A-Za-z]+\.[^.]+$")


def frame_number(basename: str):
    """DJI capture counter shared by an RGB/thermal pair (DJI_20260915182922_0001_V.JPG -> '0001'),
    or None. Unlike the full stem it does not carry the second, so a pair taken across a second
    boundary (..._091851_0010_V / ..._091852_0010_T) still matches."""
    m = _SEQ.match(basename)
    return m.group("seq") if m else None


def pair_frames(rgb_names, thermal_names, by_frame: bool = False, times=None, max_dt: float = 1.0) -> list:
    """Return [(rgb_basename, thermal_basename)] paired by capture (shared filename stem).

    `by_frame` also pairs the thermal photos left over by their DJI frame number (a pair straddling
    a second gets two different stems: 98 of 517 pairs on Lake Grove). With `times` (basename ->
    exposure seconds) a frame-number pair must be within `max_dt` s, which also guards against a
    counter repeated in the folder; without times the frame number must be unique."""
    rgb = {pair_key(n): n for n in rgb_names}
    pairs, taken, left = [], set(), []
    for tn in thermal_names:
        r = rgb.get(pair_key(tn))
        if r is not None:
            pairs.append((r, tn))
            taken.add(r)
        else:
            left.append(tn)
    if not (by_frame and left):
        return pairs
    by_seq = {}
    for n in rgb_names:
        s = frame_number(n)
        if s is not None and n not in taken:
            by_seq.setdefault(s, []).append(n)
    for tn in left:
        cands = by_seq.get(frame_number(tn) or "", [])
        if times is not None:
            tt = times.get(tn)
            cands = [] if tt is None else sorted(
                (abs(times[n] - tt), n) for n in cands if times.get(n) is not None and abs(times[n] - tt) <= max_dt)
            r = cands[0][1] if cands else None
        else:
            r = cands[0] if len(cands) == 1 else None
        if r is not None:
            pairs.append((r, tn))
            by_seq[frame_number(tn)].remove(r)
    return pairs


def exposure_offsets(pairs, rgb_pose, times, max_gap: float = 5.0, max_shift: float = 2.0):
    """Where each thermal camera was when *it* fired, relative to its RGB twin's centre.

    The thermal exposure comes 16-182 ms after the wide one on the M4T (UTCAtExposure), so at
    2.75 m/s a thermal photo placed at its twin's centre sits ~14 cm short along the flight line,
    the other way on the next line: no single shift of the map removes that. The RGB trajectory
    (SfM centres ordered by exposure time) is interpolated linearly at the thermal exposure between
    the two RGB shots around it. Pairs without times, at either end of the block, across a gap of
    more than `max_gap` s or with an implausible result (> `max_shift` m) keep a zero offset.
    -> (offsets (n, 3), dt seconds (n,), NaN where untimed)."""
    n = len(pairs)
    off, dts = np.zeros((n, 3)), np.full(n, np.nan)
    track = sorted((times[k], k) for k in rgb_pose if times.get(k) is not None)
    if len(track) < 2:
        return off, dts
    ts = np.array([t for t, _ in track])
    Cs = np.stack([rgb_pose[k][1] for _, k in track])
    for i, (r, t) in enumerate(pairs):
        tv, tt = times.get(r), times.get(t)
        if tv is None or tt is None:
            continue
        j = int(np.searchsorted(ts, tt))
        if j <= 0 or j >= len(ts) or not 1e-3 < ts[j] - ts[j - 1] <= max_gap:
            continue
        a = (tt - ts[j - 1]) / (ts[j] - ts[j - 1])
        d = Cs[j - 1] + a * (Cs[j] - Cs[j - 1]) - rgb_pose[r][1]
        if np.linalg.norm(d) <= max_shift:
            off[i], dts[i] = d, tt - tv
    return off, dts


@dataclass
class Rig:
    R_rel: np.ndarray               # (3, 3) RGB-camera -> thermal-camera rotation
    t_rel: np.ndarray               # (3,) translation in the thermal-camera frame
    rot_scatter_deg: float          # consistency of the rig rotation across frames
    trans_scatter_m: float          # consistency of the rig translation across frames
    n_pairs: int
    n_inliers: int


def _rotation_mean(Rs: np.ndarray) -> np.ndarray:
    """Chordal L2 mean rotation: average the matrices and project back to SO(3)."""
    return orthonormalize(Rs.mean(axis=0))


def _geodesic_deg(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Angle (deg) between rotation matrices A[i] and B."""
    tr = np.trace(np.einsum("nij,kj->nik", A, B), axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def estimate_rig(R_V, C_V, R_T, C_T, rot_tol_deg: float = 2.0, trans_tol_m: float = 1.0) -> Rig:
    """Robustly estimate the fixed RGB->thermal rig from paired poses (same world frame).

    R_V, C_V: RGB poses for the paired frames; R_T, C_T: thermal poses (from thermal-only SfM)."""
    R_rel_i = np.einsum("nij,nkj->nik", R_T, R_V)          # R_T R_V^T per frame
    keep = np.ones(len(R_V), bool)
    R_rel = _rotation_mean(R_rel_i)
    for _ in range(5):
        ang = _geodesic_deg(R_rel_i, R_rel)
        new = ang < max(rot_tol_deg, 3 * np.median(ang[keep]) if keep.any() else rot_tol_deg)
        if new.sum() < 3 or np.array_equal(new, keep):
            keep = new
            break
        keep = new
        R_rel = _rotation_mean(R_rel_i[keep])
    # t_rel = -R_rel R_V (C_T - C_V), robust median over inliers
    t_i = -np.einsum("ij,njk,nk->ni", R_rel, R_V[keep], (C_T - C_V)[keep])
    t_rel = np.median(t_i, axis=0)
    rot_scatter = float(np.sqrt(np.mean(_geodesic_deg(R_rel_i[keep], R_rel) ** 2)))
    trans_scatter = float(np.sqrt(np.mean(np.linalg.norm(t_i - t_rel, axis=1) ** 2)))
    return Rig(R_rel, t_rel, rot_scatter, trans_scatter, len(R_V), int(keep.sum()))


def apply_rig(rig: Rig, R_V: np.ndarray, C_V: np.ndarray):
    """Place thermal cameras from RGB poses via the rig. Returns (R_T, C_T)."""
    R_T = np.einsum("ij,njk->nik", rig.R_rel, R_V)
    C_T = C_V - np.einsum("nji,jk,k->ni", R_V, rig.R_rel.T, rig.t_rel)
    return R_T, C_T


def rebase(rec, positions, origin_from, origin_to):
    """Move a reconstruction (and its GPS positions) from one local origin to another, in place.
    Each block's local origin is its own rounded mean GPS, so RGB and thermal can differ by a
    metre when their image sets differ; binding compares poses and must share one frame."""
    d = np.array([origin_from[0] - origin_to[0], origin_from[1] - origin_to[1], 0.0])
    if not d.any():
        return
    rec.C = rec.C + d
    rec.X = rec.X + d
    if positions is not None:
        positions[:, :2] += d[:2]


def _ground_prior(X, ground, sigma: float):
    """(weight, target) point priors holding each tie point's height to `ground(xy)`; points off
    the surface (walls, roof edges, a DSM hole, a bad match) get none. The current heights may
    share a common offset (focal error), so the gate is around the median difference."""
    w, tgt = np.zeros_like(X), X.copy()
    z = ground(X[:, :2])
    dz = X[:, 2] - z
    ok = np.isfinite(dz)
    if ok.sum() < 10:
        return w, tgt
    med = np.median(dz[ok])
    mad = 1.4826 * np.median(np.abs(dz[ok] - med))
    ok &= np.abs(dz - med) < max(0.5, 3 * mad)
    w[ok, 2] = 1.0 / sigma ** 2
    tgt[ok, 2] = z[ok]
    return w, tgt


def _drop_off_geometry(rec, workers: int = 0) -> int:
    """Drop marks the bound geometry cannot explain: residual above max(30 px, 10 x median), or
    the point behind or within a tenth of the median depth of the camera. Re-triangulated on the
    bound poses, a few tracks land there (050: 1328 of 334k marks, residuals up to 1e30); they
    dominate the cost, and the first LM steps then wreck every pose to reduce them (focal
    halved, cameras moved 18 m). Returns the number of marks dropped."""
    import os
    from . import _ba
    from .sfm import _keep_observations
    if not len(rec.obs_pt):
        return 0
    c = np.ascontiguousarray
    e, _ = _ba.residuals(c(rec.R), c(rec.C), c(rec.X), rec.intr_array(), rec.pp_array(), c(rec.cam_group, np.int32),
                         c(rec.obs_cam, np.int32), c(rec.obs_pt, np.int32), c(rec.obs_uv, np.float64), 1e9,
                         workers=workers or os.cpu_count() or 1)
    depth = np.einsum("ni,ni->n", rec.X[rec.obs_pt] - rec.C[rec.obs_cam], rec.R[rec.obs_cam][:, 2, :])
    good = np.isfinite(e) & (e < max(30.0, 10.0 * float(np.median(e)))) & (depth > 0.1 * np.median(depth))
    n = int((~good).sum())
    if n:
        log.info("Bound thermal refinement: dropping %d of %d marks off the bound geometry before adjusting",
                 n, len(good))
        _keep_observations(rec, good)
    return n


def refine_bound(rec, C_target, R_target, ground=None, pos_sigma: float = 0.05, tilt_sigma_deg: float = 1.0,
                 ground_sigma: float = 0.15, max_err_px: float = 2.0, workers: int = 0) -> dict:
    """Thermal-only bundle adjustment with every camera pinned to its RGB-derived centre.

    `ground(xy) -> z` (local frame, NaN = unknown) is the RGB surface: with the tie points held to
    it the thermal focal length is observable and is refined with the distortion and principal
    point. Attitudes stay within `tilt_sigma_deg` of `R_target` (yaw is held by the tie points).
    Overlapping thermal photos then agree with each other (no doubled edges at seams) and with the
    RGB DSM they are rendered on. If the adjustment does not converge (median error above
    `max_err_px`), the poses, lens and points are restored and `reverted` is set."""
    import copy
    from .sfm import Priors, _keep_observations, bundle_adjust
    N = len(rec.R)
    saved = copy.deepcopy((rec.R, rec.C, rec.X, rec.color, rec.intr, rec.obs_cam, rec.obs_pt, rec.obs_uv, rec.stats))
    _drop_off_geometry(rec, workers)
    f0 = [it.f for it in rec.intr]
    I_t = rec.intr_array()
    I_s = np.full_like(I_t, np.inf)
    for k, it in enumerate(rec.intr):
        I_s[k, 0] = 0.03 * it.f
        I_s[k, 1:4] = 0.3
        I_s[k, 4:6] = 0.01 * it.width
    pri = Priors(np.array(C_target, np.float64), np.full((N, 3), pos_sigma), I_t, I_s,
                 axis_target=np.array(R_target[:, 2, :]), axis_weight=1.0 / np.radians(tilt_sigma_deg) ** 2)
    err0 = None
    held = 0
    for huber, thr in ((4.0, 12.0), (2.0, 4.0), (2.0, None)):
        if ground is not None:
            pri.X_weight, pri.X_target = _ground_prior(rec.X, ground, ground_sigma)
            held = int((pri.X_weight[:, 2] > 0).sum())
        err = bundle_adjust(rec, pri, huber, workers=workers)
        err0 = err if err0 is None else err0
        if thr is not None:
            _keep_observations(rec, err < thr)
    if not len(err) or np.median(err) > max_err_px or len(rec.X) < 50:
        log.warning("Bound thermal refinement did not converge (median error %.1f px, %d points): thermal "
                    "poses and lens kept as bound, unrefined", float(np.median(err)) if len(err) else -1, len(rec.X))
        (rec.R, rec.C, rec.X, rec.color, rec.intr, rec.obs_cam, rec.obs_pt, rec.obs_uv, rec.stats) = saved
        return dict(reverted=True, median_err_px=float(np.median(err)) if len(err) else None,
                    points=int(len(saved[2])), focal_px_before=[it.f for it in rec.intr],
                    focal_px=[it.f for it in rec.intr])
    rec.rms_px = float(np.sqrt(np.mean(np.square(err)))) if len(err) else 0.0
    rec.stats = dict(rec.stats or {}, rms_px=rec.rms_px, points=int(len(rec.X)), observations=int(len(err)),
                     focal_px=[it.f for it in rec.intr], k1=[it.k1 for it in rec.intr],
                     k2=[it.k2 for it in rec.intr], k3=[it.k3 for it in rec.intr],
                     principal_point=[[it.cx, it.cy] for it in rec.intr],
                     reproj_mean_px=float(np.mean(err)) if len(err) else 0.0)
    tilt = _geodesic_deg(np.einsum("nij,nkj->nik", rec.R, R_target), np.eye(3))
    shift = np.linalg.norm(rec.C - C_target, axis=1)
    out = dict(rms_px=rec.rms_px, median_err_before_px=float(np.median(err0)) if len(err0) else None,
               median_err_px=float(np.median(err)) if len(err) else None, points=int(len(rec.X)),
               points_on_ground=held,
               focal_px_before=f0, focal_px=[it.f for it in rec.intr],
               attitude_change_deg_median=float(np.median(tilt)), attitude_change_deg_max=float(tilt.max()),
               centre_shift_m_median=float(np.median(shift)), centre_shift_m_max=float(shift.max()), reverted=False)
    log.info("Bound thermal refinement: %d points, median error %.2f -> %.2f px, focal %s -> %s px, "
             "attitude change median %.2f deg, centres moved median %.3f m",
             out["points"], out["median_err_before_px"] or 0, out["median_err_px"] or 0,
             ", ".join(f"{v:.1f}" for v in f0), ", ".join(f"{v:.1f}" for v in out["focal_px"]),
             out["attitude_change_deg_median"], out["centre_shift_m_median"])
    return out


def bind_thermal(rec_rgb, frames_rgb, rec_thermal, frames_thermal, refine: bool = True,
                 lever_arm: bool = False, ground=None, attitude: str = "thermal", workers: int = 0,
                 timing: bool = False, by_frame: bool = False):
    """Rebind the thermal reconstruction's camera poses to the RGB block through a shared rig.

    Both reconstructions must be in the same local frame (see `rebase`). Returns the thermal
    Reconstruction with poses overwritten by rig-of-RGB, restricted to frames that have an RGB
    twin, plus a dict of rig diagnostics. `lever_arm=False` places each thermal centre at its RGB
    twin's (the thermal SfM cannot measure a cm-level baseline). Attitudes start from the thermal
    SfM's own (`attitude="thermal"`): they already fit the thermal tie points, while RGB x mean rig
    carries the rig's per-frame scatter (1.4 deg on 050, ~50 px). `refine` then adjusts the thermal
    lens and attitudes with the centres pinned and the tie points on `ground` (`refine_bound`).

    `timing` moves each thermal centre along the RGB trajectory to the thermal photo's own exposure
    time (`exposure_offsets`); `attitude="rig"` takes every thermal attitude from its RGB twin
    through the rig. With `refine=False` that is the "timed rig" placement: on Lake Grove (2.75 m/s)
    it cut the tiles over 20 cm from 14 % to 4 %, where the thermal-only refinement re-fits the
    attitudes to the noisier thermal tie points. `by_frame` also pairs photos by DJI frame number;
    those extra pairs are placed but the rig is still estimated from the name-paired ones (adding
    them to the estimate moved the rig ~0.3 deg on Lake Grove)."""
    rgb_pose = {frames_rgb[g].name: (rec_rgb.R[k], rec_rgb.C[k]) for k, g in enumerate(rec_rgb.used)}
    th_local = {frames_thermal[g].name: k for k, g in enumerate(rec_thermal.used)}
    times = {f.name: getattr(f, "exposure_time", None) for f in list(frames_rgb) + list(frames_thermal)}
    pairs = pair_frames(list(rgb_pose), list(th_local), by_frame=by_frame,
                        times=times if any(v is not None for v in times.values()) else None)
    pairs = [(r, t) for r, t in pairs if t in th_local]
    if len(pairs) < 3:
        raise RuntimeError(f"Only {len(pairs)} RGB/thermal pairs found; cannot estimate the rig")
    by_name = np.array([pair_key(r) == pair_key(t) for r, t in pairs])
    if by_name.sum() < 3:
        by_name[:] = True

    R_V = np.stack([rgb_pose[r][0] for r, t in pairs])
    C_V = np.stack([rgb_pose[r][1] for r, t in pairs])
    R_T = np.stack([rec_thermal.R[th_local[t]] for r, t in pairs])
    C_T = np.stack([rec_thermal.C[th_local[t]] for r, t in pairs])
    rig = estimate_rig(R_V[by_name], C_V[by_name], R_T[by_name], C_T[by_name])
    log.info("Rig: %d/%d consistent pairs, rotation scatter %.3f deg, translation scatter %.3f m, "
             "baseline %.3f m", rig.n_inliers, rig.n_pairs, rig.rot_scatter_deg, rig.trans_scatter_m,
             float(np.linalg.norm(rig.t_rel)))

    # rebuild the thermal reconstruction keeping only twinned frames, poses = rig(RGB)
    keep_local = [th_local[t] for r, t in pairs]
    used_rig = rig if lever_arm else Rig(rig.R_rel, np.zeros(3), rig.rot_scatter_deg, rig.trans_scatter_m,
                                         rig.n_pairs, rig.n_inliers)
    R_new, C_new = apply_rig(used_rig, R_V, C_V)
    if attitude == "thermal":
        R_new = R_T.copy()
    placement = dict(attitude=attitude, refined=bool(refine), timing=bool(timing), pairs=len(pairs),
                     pairs_by_name=int(by_name.sum()), pairs_by_frame_number=int((~by_name).sum()))
    if timing:
        off, dts = exposure_offsets(pairs, rgb_pose, times)
        C_new = C_new + off
        ok = np.isfinite(dts)
        if ok.any():
            sh = np.linalg.norm(off[ok, :2], axis=1)
            placement.update(timed=int(ok.sum()), dt_ms_median=float(np.median(dts[ok]) * 1e3),
                             dt_ms_range=[float(dts[ok].min() * 1e3), float(dts[ok].max() * 1e3)],
                             shift_m_median=float(np.median(sh)), shift_m_p90=float(np.percentile(sh, 90)),
                             shift_m_max=float(sh.max()))
            log.info("Thermal exposure timing: %d/%d photos moved to their own exposure (thermal %.0f ms after "
                     "the wide, %.0f..%.0f), shift median %.3f m, p90 %.3f m, max %.3f m", ok.sum(), len(pairs),
                     placement["dt_ms_median"], *placement["dt_ms_range"], placement["shift_m_median"],
                     placement["shift_m_p90"], placement["shift_m_max"])
        else:
            placement["timed"] = 0
            log.warning("Thermal exposure timing: no UTCAtExposure in the photos, thermal kept at the RGB centres")
    remap = {old: new for new, old in enumerate(keep_local)}
    rec = rec_thermal
    rec.R = np.ascontiguousarray(R_new)
    rec.C = np.ascontiguousarray(C_new)
    rec.used = [rec_thermal.used[i] for i in keep_local]
    rec.cam_group = np.ascontiguousarray(rec_thermal.cam_group[keep_local], np.int32)
    # drop observations of dropped cameras and re-index cameras
    obs_keep = np.array([c in remap for c in rec_thermal.obs_cam])
    rec.obs_cam = np.array([remap[c] for c in rec_thermal.obs_cam[obs_keep]], np.int32)
    rec.obs_pt = np.ascontiguousarray(rec_thermal.obs_pt[obs_keep], np.int32)
    rec.obs_uv = np.ascontiguousarray(rec_thermal.obs_uv[obs_keep])
    order = np.argsort(rec.obs_pt, kind="stable")
    rec.obs_cam, rec.obs_pt, rec.obs_uv = rec.obs_cam[order], rec.obs_pt[order], rec.obs_uv[order]
    rec.obs_feat = None
    # re-triangulate points with the bound poses so the sparse cloud matches the new geometry
    from .sfm import triangulate
    if len(rec.obs_pt):
        counts = np.bincount(rec.obs_pt, minlength=len(rec.X))
        rec.X, _ = triangulate(rec.R, rec.C, rec.intr_array(), rec.pp_array(), rec.cam_group,
                               rec.obs_cam, rec.obs_pt, rec.obs_uv, len(rec.X))
    diag = dict(rig_rotation_deg=[float(v) for v in _rig_euler(rig.R_rel)],
                rig_translation_m=[float(v) for v in rig.t_rel],
                rig_baseline_m=float(np.linalg.norm(rig.t_rel)),
                rotation_scatter_deg=rig.rot_scatter_deg, translation_scatter_m=rig.trans_scatter_m,
                pairs=rig.n_pairs, inliers=rig.n_inliers, lever_arm_applied=bool(lever_arm), attitude_seed=attitude,
                placement=placement)
    if refine and len(rec.obs_pt):
        diag["refinement"] = refine_bound(rec, C_new, R_new, ground=ground, workers=workers)
    return rec, diag


def _rig_euler(R):
    """RGB->thermal rotation as small yaw/pitch/roll degrees, for the report."""
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    pitch = np.degrees(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2])))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return roll, pitch, yaw
