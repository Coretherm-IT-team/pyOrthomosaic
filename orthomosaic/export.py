"""Writers for 3D outputs: PLY / LAS / LAZ point clouds, OBJ and GLB meshes.

PLY, LAS, OBJ and GLB are pure Python + numpy. Two outputs need optional packages:
LAZ compression (`pip install "laspy[lazrs]"`) and the Poisson mesh from the dense cloud
(`pip install open3d`); without them the LAS file / the DSM mesh is kept instead."""
from __future__ import annotations

import datetime
import io
import json
import math
import struct
from typing import Optional

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------
# point clouds
# --------------------------------------------------------------------------

_CHUNK = 4_000_000          # points per block when writing large clouds (bounded memory, same bytes)


def write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray, offset=(0.0, 0.0, 0.0), comment: str = ""):
    """Binary little-endian PLY. Coordinates are stored relative to `offset` (float32)."""
    n = len(xyz)
    header = ["ply", "format binary_little_endian 1.0",
              f"comment offset {offset[0]:.3f} {offset[1]:.3f} {offset[2]:.3f}"]
    if comment:
        header.append(f"comment {comment}")
    header += [f"element vertex {n}", "property float x", "property float y", "property float z",
               "property uchar red", "property uchar green", "property uchar blue", "end_header"]
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode())
        for c0 in range(0, n, _CHUNK):
            c1 = min(n, c0 + _CHUNK)
            rec = np.empty(c1 - c0, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                           ("red", "u1"), ("green", "u1"), ("blue", "u1")])
            rec["x"], rec["y"], rec["z"] = xyz[c0:c1, 0], xyz[c0:c1, 1], xyz[c0:c1, 2]
            rec["red"], rec["green"], rec["blue"] = rgb[c0:c1, 0], rgb[c0:c1, 1], rgb[c0:c1, 2]
            fh.write(memoryview(rec).cast("B"))


def write_las(path: str, xyz_abs: np.ndarray, rgb: np.ndarray, epsg: Optional[int] = None,
              scale: float = 0.001, classification: Optional[np.ndarray] = None, add: Optional[np.ndarray] = None):
    """LAS 1.2, point format 2 (XYZ + RGB), absolute coordinates, optional EPSG GeoKey VLR.
    classification: optional per-point ASPRS class (1 = unclassified, 2 = ground).
    add: optional (3,) offset added to `xyz_abs` block by block (the file is the same as for
    ``write_las(path, xyz_abs + add, ...)`` without the full-size sum in memory)."""
    n = len(xyz_abs)

    def coords(c0, c1):
        blk = xyz_abs[c0:c1]
        return blk if add is None else blk + add
    if n > _CHUNK:
        parts = [coords(c0, min(n, c0 + _CHUNK)) for c0 in range(0, n, _CHUNK)]
        mins = np.min([p.min(0) for p in parts], 0)
        maxs = np.max([p.max(0) for p in parts], 0)
        del parts
        if not (np.all(np.isfinite(mins) & (mins != 0)) and np.all(np.isfinite(maxs) & (maxs != 0))):
            full = coords(0, n)              # +-0 / NaN ties: the original whole-array reduction
            mins, maxs = full.min(0), full.max(0)
            del full
    else:
        mins, maxs = coords(0, n).min(0), coords(0, n).max(0)
    offset = np.floor(mins)
    vlrs = b""
    if epsg:
        keys = [1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, int(epsg)]
        data = struct.pack("<%dH" % len(keys), *keys)
        vlrs = (struct.pack("<H", 0) + b"LASF_Projection".ljust(16, b"\0") + struct.pack("<HH", 34735, len(data))
                + b"GeoKeyDirectoryTag".ljust(32, b"\0") + data)
    header_size = 227
    offset_to_points = header_size + len(vlrs)
    today = datetime.date.today()
    hdr = bytearray()
    hdr += b"LASF" + struct.pack("<HH", 0, 0) + b"\0" * 16 + struct.pack("<BB", 1, 2)
    hdr += b"pyOrthomosaic".ljust(32, b"\0") + b"pyOrthomosaic".ljust(32, b"\0")
    hdr += struct.pack("<HHHIIBHI", today.timetuple().tm_yday, today.year, header_size, offset_to_points,
                       1 if epsg else 0, 2, 26, n)
    hdr += struct.pack("<5I", n, 0, 0, 0, 0)
    hdr += struct.pack("<3d", scale, scale, scale)
    hdr += struct.pack("<3d", *offset)
    hdr += struct.pack("<6d", maxs[0], mins[0], maxs[1], mins[1], maxs[2], mins[2])
    assert len(hdr) == header_size
    with open(path, "wb") as fh:
        fh.write(bytes(hdr))
        fh.write(vlrs)
        for c0 in range(0, n, _CHUNK):
            c1 = min(n, c0 + _CHUNK)
            pts = np.zeros(c1 - c0, dtype=[("X", "<i4"), ("Y", "<i4"), ("Z", "<i4"), ("I", "<u2"), ("ret", "u1"),
                                           ("cls", "u1"), ("ang", "i1"), ("ud", "u1"), ("src", "<u2"),
                                           ("R", "<u2"), ("G", "<u2"), ("B", "<u2")])
            q = np.round((coords(c0, c1) - offset) / scale).astype(np.int64)
            pts["X"], pts["Y"], pts["Z"] = q[:, 0], q[:, 1], q[:, 2]
            pts["ret"] = 0b00001001          # return 1 of 1
            # ASPRS: 1 unclassified, 2 ground
            pts["cls"] = 1 if classification is None else (
                classification[c0:c1] if np.ndim(classification) else classification)
            rgb16 = rgb[c0:c1].astype(np.uint16) * 257
            pts["R"], pts["G"], pts["B"] = rgb16[:, 0], rgb16[:, 1], rgb16[:, 2]
            fh.write(memoryview(pts).cast("B"))


def las_to_laz(las_path: str) -> Optional[str]:
    """Compress a LAS file to LAZ (same header, VLRs and points). Returns the LAZ path, or None
    when no LAZ backend is installed (the LAS file is then kept)."""
    try:
        import laspy
    except ImportError:
        return None
    if not laspy.LazBackend.detect_available():
        return None
    import os
    laz = os.path.splitext(las_path)[0] + ".laz"
    # stream in chunks (memory bounded) with the parallel lazrs compressor when available
    backend = laspy.LazBackend.detect_available()[0]
    with laspy.open(las_path) as r, laspy.open(laz, mode="w", header=r.header, do_compress=True,
                                                laz_backend=backend) as w:
        for chunk in r.chunk_iterator(5_000_000):
            w.write_points(chunk)
    return laz


# --------------------------------------------------------------------------
# vector / vectorised elevation products
# --------------------------------------------------------------------------

def write_geojson_contours(path: str, contours, offset=(0.0, 0.0), epsg: Optional[int] = None):
    #changed here: Pix4D-style elevation-mapping contour deliverable (GeoJSON).
    """Write contour segments from `terrain.contour_lines` as a GeoJSON FeatureCollection.

    `contours` is a list of ``(level, segs)``; every level becomes one MultiLineString feature
    with an ``elevation`` property.  Coordinates are shifted by `offset` (the block origin).
    """
    feats = []
    for level, segs in contours:
        coords = [[[float(x) + offset[0], float(y) + offset[1]] for x, y in seg] for seg in segs]
        feats.append({"type": "Feature", "properties": {"elevation": float(level)},
                      "geometry": {"type": "MultiLineString", "coordinates": coords}})
    fc = {"type": "FeatureCollection", "features": feats}
    if epsg:
        fc["crs"] = {"type": "name", "properties": {"name": f"EPSG:{epsg}"}}
    with open(path, "w") as fh:
        json.dump(fc, fh)


# --------------------------------------------------------------------------
# meshes
# --------------------------------------------------------------------------

def grid_mesh(Z: np.ndarray, valid: np.ndarray, minX: float, maxY: float, cell: float):
    """Triangulate a height grid. Each quad is split along the diagonal with the smaller
    height difference (keeps roof ridges and walls sharp). Returns (V (n,3), F (m,3), UV (n,2))."""
    H, W = Z.shape
    idx = -np.ones((H, W), np.int64)
    ys, xs = np.nonzero(valid)
    idx[ys, xs] = np.arange(len(ys))
    V = np.column_stack([minX + (xs + 0.5) * cell, maxY - (ys + 0.5) * cell, Z[ys, xs]]).astype(np.float64)
    UV = np.column_stack([(xs + 0.5) / W, 1.0 - (ys + 0.5) / H])
    a, b = idx[:-1, :-1], idx[:-1, 1:]
    c, d = idx[1:, :-1], idx[1:, 1:]
    za, zb, zc, zd = Z[:-1, :-1], Z[:-1, 1:], Z[1:, :-1], Z[1:, 1:]
    diag_ad = np.abs(za - zd) <= np.abs(zb - zc)
    tris = []
    for m, t in [(diag_ad, (a, c, d)), (diag_ad, (a, d, b)), (~diag_ad, (a, c, b)), (~diag_ad, (b, c, d))]:
        f = np.stack([t[0][m], t[1][m], t[2][m]], 1)
        tris.append(f[(f >= 0).all(1)])
    F = np.concatenate(tris)
    return V, F, UV


def mesh_input(xyz: np.ndarray, rgb: np.ndarray, cam: np.ndarray, centers: np.ndarray, spacing: float,
               max_points: int, chunk: int = 2_000_000):
    """Reduce a dense cloud of any size to the Poisson input, in bounded memory.

    Same reduction as the worker's Open3D ``voxel_down_sample`` (voxel size
    ``max(spacing, sqrt(area / max_points))``, grid starting half a voxel below the cloud's
    minimum, per-voxel mean of position, colour and viewing direction), but streamed over
    `chunk` points at a time, so a memory-mapped cloud is never copied whole. Peak memory is
    ~300 bytes per chunk point plus ~90 bytes per occupied voxel, whatever the cloud size.

    `cam` indexes `centers` (camera centres): the viewing direction of each point.
    Returns ``(xyz float64, rgb uint8, view_dirs float32, extent (3,))``; `extent` is the full
    cloud's, for the worker's octree depth."""
    n = len(xyz)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for s in range(0, n, chunk):
        a = np.asarray(xyz[s:s + chunk], np.float64)
        lo, hi = np.minimum(lo, a.min(0)), np.maximum(hi, a.max(0))
    ext = hi - lo
    centers = np.asarray(centers, np.float64)

    def vdirs(a, c):
        d = centers[np.asarray(c)] - a
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return d

    if n <= max_points:                      # small cloud: the worker takes it as is
        a = np.asarray(xyz, np.float64)
        return a, np.asarray(rgb, np.uint8), vdirs(a, cam).astype(np.float32), ext
    area = max(float(ext[0] * ext[1]), 1e-6)
    v = max(spacing, math.sqrt(area / max_points))
    origin = lo - 0.5 * v
    dims = np.floor((hi - origin) / v).astype(np.int64) + 1
    keys = np.empty(0, np.int64)
    sums = np.empty((0, 10), np.float64)     # x y z, r g b, view x y z, count

    def reduce(k, vals):
        u, inv = np.unique(k, return_inverse=True)
        out = np.empty((len(u), vals.shape[1]), np.float64)
        for j in range(vals.shape[1]):
            out[:, j] = np.bincount(inv, weights=vals[:, j], minlength=len(u))
        return u, out

    for s in range(0, n, chunk):
        a = np.asarray(xyz[s:s + chunk], np.float64)
        idx = np.minimum(np.floor((a - origin) / v).astype(np.int64), dims - 1)
        k = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
        vals = np.column_stack([a, np.asarray(rgb[s:s + chunk], np.float64),
                                vdirs(a, cam[s:s + chunk]), np.ones(len(a))])
        del a, idx
        k, vals = reduce(k, vals)
        keys, sums = reduce(np.concatenate([keys, k]), np.vstack([sums, vals]))
        del k, vals
    cnt = sums[:, 9:10]
    return (sums[:, 0:3] / cnt, np.clip(np.rint(sums[:, 3:6] / cnt), 0, 255).astype(np.uint8),
            (sums[:, 6:9] / cnt).astype(np.float32), ext)


def poisson_mesh(xyz: np.ndarray, rgb: np.ndarray, view_dirs: np.ndarray, spacing: float,
                 max_points: int = 3_000_000, max_vertices: int = 1_500_000, trim_quantile: float = 0.08,
                 extent: Optional[np.ndarray] = None):
    """Screened Poisson surface from the dense cloud (Open3D). Returns (V, F, vertex_rgb uint8),
    or None when Open3D is missing or fails.

    Runs in a child process: Open3D's Poisson iso-surface extraction can abort the whole process
    from C++ ("Failed to close loop"), which Python cannot catch; it is also forced single-threaded,
    which avoids that abort in practice. Normals come from local PCA and are flipped towards the
    camera that saw each point. The lowest-density vertices (`trim_quantile`) and triangles longer
    than 4x the point spacing are removed: that is surface Poisson invents where there are no
    points (the "bubble" effect over gaps and edges).

    Pass a large cloud through `mesh_input` first (with ``max_points=len(reduced)`` and its
    `extent`): the worker then gets ~max_points points instead of the whole cloud.
    """
    try:
        import open3d  # noqa: F401
    except ImportError:
        return None
    import os
    import subprocess
    import sys
    import tempfile
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        src, dst = os.path.join(td, "in.npz"), os.path.join(td, "out.npz")
        ext = np.full(3, np.nan) if extent is None else np.asarray(extent, np.float64)
        np.savez(src, xyz=np.asarray(xyz, np.float64), rgb=np.asarray(rgb, np.uint8),
                 vdir=np.asarray(view_dirs, np.float32),
                 args=np.array([spacing, max_points, max_vertices, trim_quantile, *ext], np.float64))
        r = subprocess.run([sys.executable, "-c", "from orthomosaic.export import _poisson_worker as w; "
                            "import sys; w(sys.argv[1], sys.argv[2])", src, dst],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(dst):
            import logging
            logging.getLogger(__name__).warning(
                "Poisson meshing failed (exit %s): %s", r.returncode, (r.stderr or "").strip()[-300:])
            return None
        with np.load(dst) as z:              # closed before the folder is removed (Windows)
            return z["V"], z["F"], z["C"]


def _poisson_worker(src: str, dst: str):
    import open3d as o3d
    with np.load(src) as z:                  # each z[...] re-reads the file: read every array once
        args, xyz, rgb, vdir = z["args"], z["xyz"], z["rgb"], z["vdir"]
    spacing, max_points, max_vertices, trim_q = args[:4]
    ext = args[4:7] if len(args) >= 7 and np.all(np.isfinite(args[4:7])) else np.ptp(xyz, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    del xyz
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    del rgb
    pcd.normals = o3d.utility.Vector3dVector(vdir.astype(np.float64))  # carried through the voxel filter
    del vdir
    area = max(float(ext[0] * ext[1]), 1e-6)
    if len(pcd.points) > max_points:
        pcd = pcd.voxel_down_sample(max(spacing, math.sqrt(area / max_points)))
    step = max(spacing, math.sqrt(area / max(len(pcd.points), 1)))
    view = np.asarray(pcd.normals).copy()
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * step, max_nn=30))
    n = np.asarray(pcd.normals)
    n[np.sum(n * view, 1) < 0] *= -1
    pcd.normals = o3d.utility.Vector3dVector(n)
    # octree depth: finest cell ~ max(point spacing, the size that meets the vertex budget)
    cell = max(step, math.sqrt(area / max_vertices))
    depth = int(np.clip(math.floor(math.log2(max(float(ext.max()) * 1.05 / cell, 2.0))), 6, 12))
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=1.05, linear_fit=False, n_threads=1)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, trim_q))
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if len(F):
        e = np.max([np.linalg.norm(V[F[:, a]] - V[F[:, b]], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], 0)
        mesh.remove_triangles_by_mask(e > 4 * max(step, cell))
        mesh.remove_unreferenced_vertices()
    V = np.asarray(mesh.vertices)
    C = ((np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255 + 0.5).astype(np.uint8)
         if mesh.has_vertex_colors() else np.full((len(V), 3), 200, np.uint8))
    np.savez(dst, V=V, F=np.asarray(mesh.triangles), C=C)


def write_obj_colored(path: str, V: np.ndarray, F: np.ndarray, rgb: np.ndarray, offset=(0.0, 0.0, 0.0)):
    """Wavefront OBJ with per-vertex colours (`v x y z r g b`, read by MeshLab, CloudCompare,
    Blender 4.x, Pix4D/Agisoft viewers). Vertices relative to `offset`."""
    Vl = V - np.asarray(offset)
    with open(path, "w") as fh:
        fh.write(f"# pyOrthomosaic mesh from the dense point cloud; add offset {offset[0]:.3f} {offset[1]:.3f} "
                 f"{offset[2]:.3f} for absolute coordinates\n")
        np.savetxt(fh, np.column_stack([Vl, rgb / 255.0]), fmt="v %.3f %.3f %.3f %.4f %.4f %.4f")
        np.savetxt(fh, F + 1, fmt="f %d %d %d")


def write_obj(path: str, V: np.ndarray, F: np.ndarray, UV: np.ndarray, texture: Image.Image,
              offset=(0.0, 0.0, 0.0)):
    """Wavefront OBJ + MTL + JPEG texture (vertices relative to `offset`)."""
    import os
    base = os.path.splitext(path)[0]
    name = os.path.basename(base)
    texture.convert("RGB").save(base + "_texture.jpg", quality=90)
    with open(base + ".mtl", "w") as fh:
        fh.write(f"newmtl ortho\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd {name}_texture.jpg\n")
    Vl = V - np.asarray(offset)
    with open(path, "w") as fh:
        fh.write(f"# pyOrthomosaic mesh; add offset {offset[0]:.3f} {offset[1]:.3f} {offset[2]:.3f} "
                 f"for absolute coordinates\nmtllib {name}.mtl\nusemtl ortho\n")
        np.savetxt(fh, Vl, fmt="v %.3f %.3f %.3f")
        np.savetxt(fh, UV, fmt="vt %.6f %.6f")
        f1 = F + 1
        np.savetxt(fh, np.column_stack([f1[:, 0], f1[:, 0], f1[:, 1], f1[:, 1], f1[:, 2], f1[:, 2]]),
                   fmt="f %d/%d %d/%d %d/%d")


def write_glb(path: str, V: np.ndarray, F: np.ndarray, UV: Optional[np.ndarray], texture: Optional[Image.Image],
              offset=(0.0, 0.0, 0.0), vertex_rgb: Optional[np.ndarray] = None):
    """Binary glTF 2.0 with an embedded JPEG texture, or per-vertex colours (`vertex_rgb`).
    glTF is Y-up: (E, N, Z) -> (E, Z, -N)."""
    Vl = (V - np.asarray(offset)).astype(np.float32)
    pos = np.column_stack([Vl[:, 0], Vl[:, 2], -Vl[:, 1]]).astype(np.float32)
    ind = F.astype(np.uint32).ravel()
    if vertex_rgb is not None:
        return _write_glb_colored(path, pos, ind, vertex_rgb, offset)
    # glTF UV origin is top-left
    uv = np.column_stack([UV[:, 0], 1.0 - UV[:, 1]]).astype(np.float32)
    buf = io.BytesIO()
    texture.convert("RGB").save(buf, "JPEG", quality=90)
    jpg = buf.getvalue()

    chunks, views = [], []

    def add(data: bytes, target=None):
        off = sum(len(c) for c in chunks)
        chunks.append(data + b"\0" * (-len(data) % 4))
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            v["target"] = target
        views.append(v)
        return len(views) - 1

    v_pos = add(pos.tobytes(), 34962)
    v_uv = add(uv.tobytes(), 34962)
    v_ind = add(ind.tobytes(), 34963)
    v_img = add(jpg)
    gltf = {
        "asset": {"version": "2.0", "generator": "pyOrthomosaic"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0,
                                                 "roughnessFactor": 1.0}, "doubleSided": True}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": v_img, "mimeType": "image/jpeg"}],
        "accessors": [
            {"bufferView": v_pos, "componentType": 5126, "count": len(pos), "type": "VEC3",
             "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
            {"bufferView": v_uv, "componentType": 5126, "count": len(uv), "type": "VEC2"},
            {"bufferView": v_ind, "componentType": 5125, "count": len(ind), "type": "SCALAR"},
        ],
        "bufferViews": views,
        "extras": {"offset": list(map(float, offset)), "axes": "x=East, y=Up, z=-North"},
    }
    binary = b"".join(chunks)
    gltf["buffers"] = [{"byteLength": len(binary)}]
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(binary)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        fh.write(struct.pack("<II", len(binary), 0x004E4942) + binary)


def _write_glb_colored(path, pos, ind, rgb, offset):
    col = np.ascontiguousarray(rgb, np.uint8)
    col = np.column_stack([col, np.full(len(col), 255, np.uint8)])
    blobs = [pos.tobytes(), col.tobytes(), ind.tobytes()]
    views, off = [], 0
    for b, tgt in zip(blobs, (34962, 34962, 34963)):
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(b), "target": tgt})
        off += len(b) + (-len(b) % 4)
    binary = b"".join(b + b"\0" * (-len(b) % 4) for b in blobs)
    gltf = {
        "asset": {"version": "2.0", "generator": "pyOrthomosaic"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0}, "doubleSided": True}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(pos), "type": "VEC3",
             "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
            {"bufferView": 1, "componentType": 5121, "normalized": True, "count": len(col), "type": "VEC4"},
            {"bufferView": 2, "componentType": 5125, "count": len(ind), "type": "SCALAR"},
        ],
        "bufferViews": views, "buffers": [{"byteLength": len(binary)}],
        "extras": {"offset": list(map(float, offset)), "axes": "x=East, y=Up, z=-North"},
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binary)))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        fh.write(struct.pack("<II", len(binary), 0x004E4942) + binary)
