"""Thermal binding: pairing by filename, and recovering a known rig from paired poses."""
import math
import os
import sys

import numpy as np

try:
    import orthomosaic._core  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orthomosaic import binding  # noqa: E402
from orthomosaic.camera import rodrigues  # noqa: E402


def test_pairing():
    rgb = ["DJI_20260915182922_0001_V.JPG", "DJI_20260915182924_0002_V.JPG", "DJI_20260915182926_0003_V.JPG"]
    thr = ["DJI_20260915182922_0001_T.JPG", "DJI_20260915182926_0003_T.JPG", "DJI_20260915182930_0009_T.JPG"]
    pairs = binding.pair_frames(rgb, thr)
    assert pairs == [("DJI_20260915182922_0001_V.JPG", "DJI_20260915182922_0001_T.JPG"),
                     ("DJI_20260915182926_0003_V.JPG", "DJI_20260915182926_0003_T.JPG")]


def test_pairing_by_frame_number_across_a_second():
    # the 0010 pair straddles a second: different stems, same frame number and ~10 ms apart
    rgb = ["DJI_20261029091851_0010_V.JPG", "DJI_20261029091852_0011_V.JPG", "DJI_20261029091854_0012_V.JPG"]
    thr = ["DJI_20261029091852_0010_T.JPG", "DJI_20261029091852_0011_T.JPG", "DJI_20261029091859_0012_T.JPG"]
    assert len(binding.pair_frames(rgb, thr)) == 1
    times = {rgb[0]: 10.72, thr[0]: 10.73, rgb[1]: 12.23, thr[1]: 12.26, rgb[2]: 14.0, thr[2]: 19.0}
    pairs = binding.pair_frames(rgb, thr, by_frame=True, times=times)
    # 0012 is the same counter but 5 s apart (counter reused): not a pair
    assert sorted(pairs) == [(rgb[0], thr[0]), (rgb[1], thr[1])]
    assert binding.frame_number(thr[0]) == "0010" and binding.frame_number("IMG.JPG") is None


def test_exposure_offsets_follow_the_flight_line():
    # two lines flown in opposite directions at 3 m/s, one shot every 2 s; thermal fires 50 ms late
    names, pose, times, pairs = [], {}, {}, []
    for k in range(10):
        line, step = divmod(k, 5)
        x = 3.0 * 2 * step if line == 0 else 3.0 * 2 * (4 - step)
        t = 2.0 * k
        v = f"DJI_{k:04d}_V.JPG"
        pose[v] = (np.eye(3), np.array([x, 20.0 * line, 30.0]))
        times[v], times[v.replace("_V", "_T")] = t, t + 0.05
        pairs.append((v, v.replace("_V", "_T")))
    off, dts = binding.exposure_offsets(pairs, pose, times)
    np.testing.assert_allclose(dts[np.isfinite(dts)], 0.05)
    # inside each line: 15 cm along the direction of flight, opposite on the return line
    np.testing.assert_allclose(off[1], [0.15, 0, 0], atol=1e-9)
    np.testing.assert_allclose(off[6], [-0.15, 0, 0], atol=1e-9)
    # the very last shot has nothing after it: left at its twin
    assert np.isnan(dts[-1]) and not off[-1].any()


def test_estimate_rig_recovers_known_offset():
    rng = np.random.default_rng(0)
    n = 40
    R_V = rodrigues(np.c_[np.full(n, np.pi), rng.normal(0, 0.05, (n, 2))])
    C_V = np.c_[np.linspace(-30, 30, n), rng.uniform(-5, 5, n), 50 + rng.normal(0, 0.5, n)]
    # a true rig: thermal rotated ~1.5 deg from RGB, 4 cm baseline
    R_rel_true = rodrigues(np.radians([0.8, -1.2, 0.5]))
    t_rel_true = np.array([0.03, -0.02, 0.01])
    R_T = np.einsum("ij,njk->nik", R_rel_true, R_V)
    C_T = C_V - np.einsum("nji,jk,k->ni", R_V, R_rel_true.T, t_rel_true)
    # add pose noise (thermal SfM is noisier)
    R_T = np.einsum("nij,njk->nik", rodrigues(rng.normal(0, np.radians(0.2), (n, 3))), R_T)
    C_T = C_T + rng.normal(0, 0.03, C_T.shape)
    rig = binding.estimate_rig(R_V, C_V, R_T, C_T)
    assert float(binding._geodesic_deg(R_rel_true[None], rig.R_rel)[0]) < 0.3
    assert np.linalg.norm(rig.t_rel - t_rel_true) < 0.05
    # apply_rig with the recovered rig reproduces the noise-free thermal poses
    R_clean = np.einsum("ij,njk->nik", R_rel_true, R_V)
    C_clean = C_V - np.einsum("nji,jk,k->ni", R_V, R_rel_true.T, t_rel_true)
    R_back, C_back = binding.apply_rig(rig, R_V, C_V)
    assert binding._geodesic_deg(R_back, np.eye(3)).shape == (n,)   # shape sanity
    assert np.abs(C_back - C_clean).max() < 0.05
    assert binding._geodesic_deg(np.einsum("nij,nkj->nik", R_back, R_clean), np.eye(3)).max() < 0.3


def test_rig_outlier_rejection():
    rng = np.random.default_rng(2)
    n = 30
    R_V = rodrigues(np.c_[np.full(n, np.pi), rng.normal(0, 0.05, (n, 2))])
    C_V = np.c_[np.linspace(-20, 20, n), rng.uniform(-3, 3, n), np.full(n, 45.0)]
    R_rel = rodrigues(np.radians([0.5, 0.5, 0.0]))
    t_rel = np.array([0.05, 0.0, 0.0])
    R_T = np.einsum("ij,njk->nik", R_rel, R_V)
    C_T = C_V - np.einsum("nji,jk,k->ni", R_V, R_rel.T, t_rel)
    # 5 gross outliers (bad thermal poses)
    bad = rng.choice(n, 5, replace=False)
    R_T[bad] = rodrigues(rng.normal(0, 1.0, (5, 3))) @ R_T[bad]
    C_T[bad] += rng.normal(0, 5, (5, 3))
    rig = binding.estimate_rig(R_V, C_V, R_T, C_T)
    assert rig.n_inliers >= n - 8
    assert np.linalg.norm(binding._geodesic_deg(R_rel[None], rig.R_rel)) < 0.5


def test_rebase_moves_block_to_other_origin():
    from orthomosaic.sfm import Reconstruction
    rec = Reconstruction([0], np.eye(3)[None], np.array([[1.0, 2.0, 30.0]]), [], np.zeros(1, np.int32),
                         np.array([[5.0, 5.0, 0.0]]), np.zeros((1, 3), np.uint8), np.zeros(0, np.int32),
                         np.zeros(0, np.int32), np.zeros((0, 2)))
    pos = np.array([[1.0, 2.0]])
    binding.rebase(rec, pos, (500100.0, 4000050.0), (500101.0, 4000049.0))
    # same UTM point: 500100 + 1 == 500101 + 0, 4000050 + 2 == 4000049 + 3
    assert np.allclose(rec.C[0], [0.0, 3.0, 30.0]) and np.allclose(rec.X[0], [4.0, 6.0, 0.0])
    assert np.allclose(pos[0], [0.0, 3.0])


def _terrain(xy):
    return 0.5 * np.sin(xy[:, 0] / 10.0) + 0.3 * np.cos(xy[:, 1] / 7.0)


def _bound_block(rng, f_true=1000.0, f_start=1030.0, n_side=6):
    """Nadir thermal block over a flat textured plane: true poses, observations from the true lens,
    and a reconstruction that starts from a wrong focal and slightly rotated attitudes."""
    from orthomosaic.camera import Intrinsics
    from orthomosaic.sfm import Reconstruction
    xs, ys = np.meshgrid(np.arange(n_side) * 8.0, np.arange(n_side) * 8.0)
    C = np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, 40.0)]
    N = len(C)
    R = rodrigues(np.c_[np.full(N, np.pi), np.zeros((N, 2))])          # looking straight down
    X = np.c_[rng.uniform(-10, 50, 1500), rng.uniform(-10, 50, 1500), np.zeros(1500)]
    X[:, 2] = _terrain(X[:, :2])
    W, H = 640, 512
    oc, op, uv = [], [], []
    for p, P in enumerate(X):
        for k in range(N):
            q = R[k] @ (P - C[k])
            u, v = f_true * q[0] / q[2] + W / 2, f_true * q[1] / q[2] + H / 2
            if 0 <= u < W and 0 <= v < H:
                oc.append(k); op.append(p); uv.append((u, v))
    oc, op, uv = np.array(oc, np.int32), np.array(op, np.int32), np.array(uv) + rng.normal(0, 0.3, (len(oc), 2))
    it = Intrinsics(width=W, height=H, f=f_start, ppx=W / 2, ppy=H / 2)
    R0 = np.einsum("nij,njk->nik", rodrigues(rng.normal(0, np.radians(0.3), (N, 3))), R)
    rec = Reconstruction(list(range(N)), R0, C.copy(), [it], np.zeros(N, np.int32), X.copy(),
                         np.zeros((len(X), 3), np.uint8), oc, op, uv)
    from orthomosaic.sfm import triangulate
    rec.X, _ = triangulate(rec.R, rec.C, rec.intr_array(), rec.pp_array(), rec.cam_group,
                           rec.obs_cam, rec.obs_pt, rec.obs_uv, len(rec.X))
    return rec, R, C, X


def test_refine_bound_recovers_focal_with_pinned_centres():
    """Pinned (RGB) centres and tie points held to the (RGB) ground fix the scene depth, so a 3%
    focal error and 0.3 deg attitude noise are removed and the centres stay put."""
    rng = np.random.default_rng(4)
    rec, _, C_true, _ = _bound_block(rng)
    R_prior = rec.R.copy()
    out = binding.refine_bound(rec, C_true, R_prior, ground=_terrain, workers=1)
    assert abs(rec.intr[0].f - 1000.0) < 3.0, rec.intr[0].f
    assert out["centre_shift_m_max"] < 0.05
    assert out["median_err_px"] < 0.6
    assert out["points_on_ground"] > 0.9 * out["points"]


def test_refine_bound_without_ground_cannot_see_focal():
    """The nadir ambiguity this guards against: without the ground, the focal error stays."""
    rng = np.random.default_rng(4)
    rec, _, C_true, _ = _bound_block(rng)
    binding.refine_bound(rec, C_true, rec.R.copy(), workers=1)
    assert abs(rec.intr[0].f - 1000.0) > 10.0


def test_refine_bound_reverts_when_it_cannot_converge():
    """Observations that fit no geometry (shuffled marks): the bound poses and lens are kept."""
    rng = np.random.default_rng(5)
    rec, _, C_true, _ = _bound_block(rng)
    rec.obs_uv = rec.obs_uv[rng.permutation(len(rec.obs_uv))]
    R0, C0, f0 = rec.R.copy(), rec.C.copy(), rec.intr[0].f
    out = binding.refine_bound(rec, C_true, R0, ground=_terrain, workers=1)
    assert out["reverted"]
    assert np.array_equal(rec.R, R0) and np.array_equal(rec.C, C0) and rec.intr[0].f == f0


def test_marks_off_the_bound_geometry_are_dropped():
    """Tracks re-triangulated behind a camera or far off it are removed before the adjustment
    (on flight 050 their 1e30 residuals made the first LM steps wreck the whole block)."""
    rng = np.random.default_rng(6)
    rec, _, _, _ = _bound_block(rng)
    n_obs = len(rec.obs_pt)
    bad = np.zeros(len(rec.X), bool)
    bad[:20] = True
    rec.X[:10, 2] = 200.0                                         # behind the 40 m cameras
    rec.X[10:20, :2] += 5.0                                       # 5 m off: hundreds of px
    n_bad = int(bad[rec.obs_pt].sum())
    dropped = binding._drop_off_geometry(rec, workers=1)
    assert dropped >= n_bad and dropped < n_bad + 0.02 * n_obs
    assert len(rec.X) <= len(bad) - 20 + 1
    assert binding._drop_off_geometry(rec, workers=1) == 0      # what is left fits

if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
