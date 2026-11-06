import os
import sys
import tempfile

import numpy as np

try:
    import orthomosaic._dense  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orthomosaic import _dense, densify  # noqa: E402
from orthomosaic.densify import DenseCloud, _CloudWriter, _row_bands, rasterize  # noqa: E402
from orthomosaic._gpu_depth import same_bits  # noqa: E402


def _cloud(rng, n, W, H, gsd):
    """Points over a W x H grid with roofs, walls (several layers per cell) and empty patches."""
    x = rng.uniform(-3 * gsd, (W + 3) * gsd, n)
    y = rng.uniform(-3 * gsd, (H + 3) * gsd, n)
    z = np.where((x > W * gsd * 0.3) & (y > H * gsd * 0.4), 6.0, 0.0) + rng.normal(0, 0.05, n)
    low = rng.random(n) < 0.1                                  # a lower layer under the top one
    z[low] -= rng.uniform(1, 5, int(low.sum()))
    wall = rng.random(n) < 0.08
    z[wall] = rng.uniform(0, 6, int(wall.sum()))
    gap = (x > W * gsd * 0.1) & (x < W * gsd * 0.15)
    xyz = np.column_stack([x, y, z])[~gap]
    m = len(xyz)
    xyz[:50, 2] = xyz[0, 2]                                   # equal heights (stable order matters)
    rgb = rng.integers(0, 256, (m, 3), dtype=np.uint8)
    return DenseCloud(xyz, rgb, rng.integers(2, 6, m).astype(np.uint8), rng.integers(0, 40, m).astype(np.int32),
                      2 * gsd)


def _rasterize_unique(cloud, minX, maxY, W, H, gsd, layer_gap=1.0):
    """The 0.9.3 top-layer step (np.unique over occupied cells), as the reference."""
    col = np.floor((cloud.xyz[:, 0] - minX) / gsd).astype(np.int64)
    row = np.floor((maxY - cloud.xyz[:, 1]) / gsd).astype(np.int64)
    ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    uc, inv = np.unique((row * W + col)[ok], return_inverse=True)
    Zc, RGBc, _ = _dense.top_layer(np.ascontiguousarray(inv, np.int64), np.ascontiguousarray(cloud.xyz[ok, 2]),
                                   np.ascontiguousarray(cloud.rgb[ok]), len(uc), float(layer_gap))
    Z = np.full((H, W), np.nan, np.float32)
    Z.ravel()[uc] = Zc
    RGB = np.zeros((H, W, 3), np.uint8)
    RGB.reshape(-1, 3)[uc] = RGBc
    return Z, RGB


def test_row_bands():
    assert _row_bands(np.array([5, 5, 5, 5]), 10, 10, 1000) == [(0, 2), (2, 4)]
    assert _row_bands(np.array([50, 1, 1]), 10, 10, 1000) == [(0, 1), (1, 3)]      # a full row is never split
    assert _row_bands(np.array([0, 0, 0, 0, 0]), 10, 10, 20) == [(0, 2), (2, 4), (4, 5)]
    assert _row_bands(np.zeros(3, np.int64), 10, 10, 1000) == [(0, 3)]


def test_banded_rasterize_matches_unique_reference():
    rng = np.random.default_rng(2)
    W, H, gsd = 120, 90, 0.05
    cloud = _cloud(rng, 200_000, W, H, gsd)
    Zr, Cr = _rasterize_unique(cloud, 0.0, H * gsd, W, H, gsd)
    old_chunk = densify.RASTER_CHUNK
    try:
        for chunk, bp, bc in [(1 << 23, None, None), (7_777, None, None), (10_000, 20_000, None),
                              (50_000, 1_000_000, 5 * W), (3_000, 1, 1)]:
            densify.RASTER_CHUNK = chunk
            with tempfile.TemporaryDirectory() as tmp:
                d = rasterize(cloud, 0.0, H * gsd, W, H, gsd, radii=[0.0], band_points=bp, band_cells=bc,
                              scratch_dir=tmp)
                assert os.listdir(tmp) == []                  # band files removed
            assert same_bits(d.Z, Zr) and same_bits(d.rgb, Cr), (chunk, bp, bc)
    finally:
        densify.RASTER_CHUNK = old_chunk
    print("ok banded rasterize == np.unique reference")


def test_cloud_writer_disk_equals_memory():
    rng = np.random.default_rng(4)
    parts = [(rng.normal(0, 10, (n, 3)), rng.integers(0, 256, (n, 3), dtype=np.uint8),
              rng.integers(2, 9, n).astype(np.uint8), np.full(n, k, np.int32))
             for k, n in enumerate([0, 5, 1000, 1, 37])]
    with tempfile.TemporaryDirectory() as tmp:
        mem, disk = _CloudWriter(None), _CloudWriter(os.path.join(tmp, "dense_cloud"))
        for p in parts:
            if len(p[0]):
                mem.add(*p)
                disk.add(*p)
        a, b = mem.finish(), disk.finish()
        assert not os.path.exists(os.path.join(tmp, "dense_cloud")) or os.name == "nt"
        for x, y in zip(a, b):
            assert isinstance(y, np.memmap) and same_bits(np.asarray(y), x)
        assert _CloudWriter(None).finish() is None
        assert _CloudWriter(os.path.join(tmp, "empty")).finish() is None


if __name__ == "__main__":
    for t in (test_row_bands, test_banded_rasterize_matches_unique_reference, test_cloud_writer_disk_equals_memory):
        t()
        print("ok", t.__name__)
