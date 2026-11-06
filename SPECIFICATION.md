# pyOrthomosaic — Library Specification

**Package name (PyPI):** `pyOrthomosaic`
**Import name:** `orthomosaic`
**Version:** 0.3.0
**License:** MIT
**Author:** Soyabul Islam Lincoln
**Python:** ≥ 3.9

A native orthomosaic and 2.5D reconstruction toolkit for **nadir drone imagery**. It performs the
full photogrammetry pipeline — metadata reading, feature extraction, matching, global alignment,
structure-from-motion, dense stereo, and product export — **without Docker, VMs, OpenCV, GDAL or
PROJ**. The compute-heavy kernels are implemented in Cython-compiled C (releasing the GIL), with
optional GPU acceleration on **NVIDIA (CUDA via CuPy)** and **Apple Silicon (Metal via MLX)**.

Runtime dependencies are just `numpy` and `pillow`; a GPU backend adds `cupy` or `mlx`.

---

## Table of contents

1. [Installation](#1-installation)
2. [Architecture overview](#2-architecture-overview)
3. [Public API](#3-public-api)
   - [3.1 `build_orthomosaic` — 2D planar mosaic](#31-build_orthomosaic--2d-planar-mosaic)
   - [3.2 `Options` — 2D options](#32-options--2d-options)
   - [3.3 `build_3d` — 2.5D reconstruction](#33-build_3d--25d-reconstruction)
   - [3.4 `Options3D` — 3D options](#34-options3d--3d-options)
   - [3.5 `build_thermal_bound` — RGB-driven thermal binding](#35-build_thermal_bound--rgb-driven-thermal-binding)
   - [3.6 `dtm_from_dsm` / `TerrainOptions` — bare-ground DTM](#36-dtm_from_dsm--terrainoptions--bare-ground-dtm)
   - [3.7 Thermal palettes: `palette_names`, `apply_palette`, `recolor`](#37-thermal-palettes)
   - [3.8 Ground control points: `load_gcps`](#38-ground-control-points-load_gcps)
   - [3.9 Backend selection: `select_backend`, `cuda_available`](#39-backend-selection)
   - [3.10 `write_quality_report` — processing quality report](#310-write_quality_report--processing-quality-report)
4. [Command-line interfaces](#4-command-line-interfaces)
5. [Output products](#5-output-products)
6. [Internal modules (developer reference)](#6-internal-modules-developer-reference)
7. [Native (Cython) kernels](#7-native-cython-kernels)
8. [Environment variables](#8-environment-variables)
9. [Scope and limitations](#9-scope-and-limitations)

---

## 1. Installation

```bash
pip install pyOrthomosaic            # CPU only
pip install "pyOrthomosaic[cuda]"    # + NVIDIA GPU (CUDA via CuPy; [ctk] pulls the CUDA headers)
pip install "pyOrthomosaic[mps]"     # + Apple-silicon GPU (Metal via MLX)
```

Prebuilt wheels ship for Linux (manylinux_2_28), Windows, and macOS 14+ on CPython 3.9–3.13.

From source:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy cython pillow setuptools
python setup.py build_ext --inplace          # compiles the .pyx kernels
```

`ORTHO_NATIVE=1 python setup.py build_ext --inplace` adds `-march=native`.

---

## 2. Architecture overview

The library is one Python package (`orthomosaic`) with three compiled extension modules:

| Extension | Source | Role |
|---|---|---|
| `_core` | `_core.pyx` | Image ops, ORB-style features, Hamming matching, RANSAC affine, warp/blend accumulator |
| `_ba` | `_ba.pyx` | Bundle-adjustment linear algebra (Schur reduction, back-substitution, residuals) |
| `_mvs` | `_mvs.pyx` | Dense multi-view stereo: view sampling, semi-global matching, bilateral surface filter |

Three interchangeable **compute backends** expose one small interface (matching, warp-accumulate,
dense sampling, box filter): `CPUBackend` (Cython), `CUDABackend` (CuPy raw kernels), `MLXBackend`
(Apple Metal). `select_backend` picks one; every GPU kernel is checked against the CPU version.

**Two pipelines** sit on top:

- **2D** (`pipeline.py`): metadata → features → matching → linear global affine alignment →
  exposure gains → tiled GeoTIFF render. Fast planar mosaic; tall objects lean.
- **3D** (`reconstruct.py`): reuses the 2D alignment as SfM initialization → track building and
  triangulation → Cython bundle adjustment → dense height-sweep stereo → DSM/DTM/nDSM, true
  orthophoto, point clouds, textured mesh.

---

## 3. Public API

Everything below is exported from the top-level package:

```python
from orthomosaic import (
    Options, build_orthomosaic,
    Options3D, build_3d, build_thermal_bound,
    TerrainOptions, dtm_from_dsm,
    palette_names, apply_palette, recolor,
    load_gcps,
    select_backend, cuda_available,
    __version__,
)
```

### 3.1 `build_orthomosaic` — 2D planar mosaic

```python
build_orthomosaic(images, output, options=None) -> dict
```

Builds a planar orthomosaic and writes a tiled, Deflate-compressed **RGBA GeoTIFF** in UTM.

**Parameters**

- `images` (`str | Sequence[str]`): a folder path (all supported images inside it are used) or an
  explicit list of image paths. At least two images are required.
- `output` (`str`): path to the output `.tif`. Parent directories are created.
- `options` (`Options | None`): tuning; defaults used if omitted.

**Returns** a report `dict` (also written to `<output>_report.json`) containing: backend name, image
counts (total / used / dropped), candidate & verified pair counts, georeferencing flag and EPSG,
output GSD, match RMS in pixels, elapsed seconds, thermal block (if radiometric), the full options
used, and a per-camera map of absolute affine transform + exposure gain.

**Side-effect outputs**

- `<output>` — the RGBA GeoTIFF (EPSG:326xx/327xx UTM).
- `<output stem>_preview.jpg` — a preview ≤ 2048 px (unless `preview=False`).
- `<output stem>_report.json` — the report.
- For radiometric thermal surveys: `<stem>_thermal.tif` (float32 raw values) and `<stem>_legend.png`.

**Pipeline stages** (implemented across `pipeline.align_images` + `render.render`):

1. **Metadata** — EXIF/XMP GPS projected to UTM with a built-in transverse Mercator (`geo.py`).
2. **Features** — ORB-style: 4-level pyramid, FAST-9 corners Harris-scored, grid bucketed,
   intensity-centroid orientation, 256-bit steered BRIEF (`_core.pyx`).
3. **Pairs** — k nearest GPS neighbours; without GPS, all pairs (small sets) or a capture-order window.
4. **Matching** — single-pass Hamming (popcount) matcher, ratio test, mutual check, RANSAC affine,
   least-squares refit. GPU path uses a shared-memory kernel.
5. **Global alignment** — linear least-squares bundle of one affine per image; X and Y decouple into
   a 3N×3N system solved relative → robust similarity-to-GPS → joint refinement with GPS priors and
   a scale-lock constraint that removes world-space shrinkage bias.
6. **Exposure** — per-channel gain compensation (Brown & Lowe style); disabled for radiometric thermal.
7. **Render** — 1024-px blocks rendered in parallel with serpentine traversal and an LRU image cache,
   streamed into the GeoTIFF; memory stays bounded regardless of mosaic size.

### 3.2 `Options` — 2D options

`dataclass` (`pipeline.Options`). All fields have defaults:

| Field | Default | Meaning |
|---|---|---|
| `backend` | `"auto"` | `auto` (CUDA → Apple GPU → CPU) \| `cuda` \| `mps` \| `cpu` |
| `workers` | `0` | CPU threads; `0` → `os.cpu_count()` |
| `feature_max_dim` | `2000` | detect features on images downscaled to this max dimension |
| `n_features` | `5000` | target features per image |
| `neighbors` | `8` | candidate pairs per image (GPS kNN) |
| `ratio` | `0.8` | Lowe ratio-test threshold |
| `min_inliers` | `25` | minimum RANSAC inliers to accept a pair |
| `gps_sigma` | `3.0` | expected GPS error, metres (use ~0.05 for RTK) |
| `match_sigma_px` | `4.0` | expected matching residual, pixels |
| `resolution` | `None` | output GSD (m/px); default = native GSD / `render_scale` |
| `render_scale` | `1.0` | decode images at this fraction of full resolution (RAM/speed knob) |
| `blend` | `"feather"` | `feather` (distance-weighted) \| `seam` (most-nadir pixel wins) |
| `feather_power` | `2.0` | feather falloff exponent |
| `exposure_compensation` | `True` | per-channel gain balancing |
| `block` | `1024` | render block size in pixels |
| `cache_mb` | `1024` | decoded-image cache budget (main RAM knob) |
| `preview` | `True` | write the preview JPEG |
| `thermal` | `"auto"` | `auto` (use raw radiometric data when every image has it) \| `off` |
| `palette` | `"rainbow"` | thermal palette name (see §3.7) |

### 3.3 `build_3d` — 2.5D reconstruction

```python
build_3d(images, out_dir, options=None, pre=None) -> dict
```

Full 2.5D reconstruction for **nadir grid flights**: one height per ground cell (roofs, ground,
trees — not walls).

**Parameters**

- `images` (`str | Sequence[str]`): folder or explicit list.
- `out_dir` (`str`): output directory (created).
- `options` (`Options3D | None`).
- `pre` (`tuple | None`): an existing `(AlignResult, Reconstruction)` to reuse an already-solved
  block (used internally by thermal binding) instead of aligning and triangulating again.

**Returns** a report `dict` (also `out_dir/report.json`) with image counts, backend, georeferencing
/ EPSG / origin, the height datum and offset to absolute altitude, SfM statistics, DSM metadata
(GSD, size, bounds, confident fraction, z-range), DTM metadata, dense point count, an `outputs` map,
timing, thermal block, options, and full per-camera pose + intrinsics.

**Reconstruction stages** (`reconstruct._products` + `sfm.py` + `mvs.py`):

1. **Structure from motion** — 2D alignment seeds poses; matches verified with a plane+parallax test
   (robust on flat scenes), linked into multi-view tracks, triangulated. A Cython Levenberg–Marquardt
   bundle adjustment (Schur complement, Huber loss) refines poses, points, and self-calibrates radial
   distortion, using GPS and DJI relative-altitude priors (and GCPs if given).
2. **Dense matching** — coarse-to-fine height sweep over a ground grid; texture-weighted NCC against a
   per-cell reference view over the best half of the other views (occlusion handling); coarse level
   regularised by 8-direction semi-global matching. Runs on numpy (Cython) or CuPy (CUDA kernel).
3. **Products** — progressive blunder removal, push-pull hole filling, edge-preserving (bilateral)
   surface filter, true orthophoto blending only views that agree at each cell's height, point clouds,
   and a grid mesh that keeps ridges/walls sharp. A DTM (bare ground) is derived from the DSM.

**Focal length note:** straight-down imagery cannot separate focal length from depth, so focal is
held at the EXIF 35 mm-equivalent value (heights inherit its ~1–2% accuracy). Set
`refine_focal=True` only for flights with large altitude changes.

### 3.4 `Options3D` — 3D options

`dataclass` (`reconstruct.Options3D`) — **extends `Options`**, so every 2D field above is also valid,
plus:

| Field | Default | Meaning |
|---|---|---|
| `neighbors` | `10` | candidate pairs per image (overrides the 2D default of 8) |
| `alt_sigma` | `0.5` | DJI relative-altitude accuracy prior, metres |
| `refine_focal` | `None` | `None` → refine only when an altitude reference exists; `True`/`False` to force |
| `dsm_resolution` | `None` | DSM cell size (m); default = 2 × native GSD |
| `max_views` | `6` | views used per cell in the dense sweep |
| `min_score` | `0.5` | NCC photo-consistency needed for a confident dense point (0–1) |
| `ncc_window` | `3` | NCC window radius |
| `dsm_smooth` | `0.45` | edge-preserving surface smoothing height scale (m); `0` = off |
| `dsm_smooth_iters` | `2` | bilateral filter iterations |
| `tile` | `160` | dense-processing tile size |
| `cloud_step` | `1` | keep every n-th confident DSM cell in the dense cloud (thinning) |
| `mesh_max_vertices` | `1_500_000` | mesh vertex budget (auto block-reduce to fit) |
| `texture_max` | `8192` | max mesh texture dimension |
| `formats` | `("ply","las","obj","glb")` | which export formats to write |
| `dtm` | `True` | also produce the bare-ground DTM / nDSM |
| `dtm_max_object` | `60.0` | largest building/tree short side (m) removed by the ground filter |
| `dtm_slope` | `0.3` | terrain slope tolerated by the ground filter |
| `gcp` | `None` | path to a WebODM/Pix4D GCP list file |
| `gcp_sigma` | `0.05` | surveyed GCP accuracy prior, metres |

### 3.5 `build_thermal_bound` — RGB-driven thermal binding

```python
build_thermal_bound(rgb_images, thermal_images, out_dir,
                    options=None, thermal_options=None) -> dict
```

For DJI dual sensors (M4T, M3T, H20T, …) that fire RGB and thermal together from one gimbal. The
RGB block triangulates precisely; thermal is low-texture and drifts on its own. This solves the RGB
block, then places every thermal image from its RGB twin's pose plus one shared **rig offset** (fixed
rotation + translation). Thermal DSM/orthophoto inherit RGB-grade geometry and are pixel-registered
to the RGB products.

**Parameters**

- `rgb_images`, `thermal_images` (`str`): the two image folders. Images are paired by their shared
  DJI capture filename (the sequence before the `_V` / `_T` suffix).
- `out_dir` (`str`): RGB products land in `out_dir/rgb`, bound thermal in `out_dir/thermal`.
- `options` (`Options3D | None`): options for the RGB block.
- `thermal_options` (`Options3D | None`): options for the thermal block; a sensible default is
  derived (higher feature counts, more neighbours, no exposure compensation).

**Returns** a report with the rig (roll/pitch/yaw, baseline, per-frame scatter) and per-block
summaries; also `out_dir/binding_report.json`. The rig is estimated robustly with outlier frames
rejected; large rotation/translation scatter is flagged (cameras not truly rigid, or a noisy pose set).

### 3.6 `dtm_from_dsm` / `TerrainOptions` — bare-ground DTM

```python
dtm_from_dsm(dsm, gsd, opt=None) -> (dtm, ground_mask)
```

Derives a **Digital Terrain Model** (bare ground with buildings, vegetation and cars removed) from a
DSM array. Runs automatically inside `build_3d`, and standalone on any existing DSM GeoTIFF.

**Parameters**

- `dsm` (`np.ndarray`, float; NaN = no data): the surface model.
- `gsd` (`float`): DSM cell size in metres.
- `opt` (`TerrainOptions | None`).

**Returns** `(dtm, ground_mask)` at the DSM resolution — `dtm` is a float array (NaN where the DSM
is), `ground_mask` is a boolean array of cells classified as ground.

**Method:** the DSM is resampled to a coarse grid using each block's 25th percentile (robust to
stereo noise); low blunders are dropped; a **progressive morphological filter** (Zhang et al., 2003)
runs openings with windows growing up to `max_object_size`, with a slope-dependent height threshold
separating objects from ground; detected objects are buffered; ground is interpolated underneath,
refined at full resolution, lightly smoothed, and never allowed above the real surface.

**`TerrainOptions` fields** (`terrain.TerrainOptions`):

| Field | Default | Meaning |
|---|---|---|
| `max_object_size` | `60.0` | largest building/tree footprint (short side) to remove, metres |
| `slope` | `0.3` | terrain slope tolerated inside one window (rise/run) |
| `dh0` | `0.3` | height above local ground still counted as ground, metres |
| `dh_max` | `3.0` | cap on the growing height threshold, metres |
| `cell` | `0.0` | working grid cell (m); `0` = automatic (~0.25–1 m) |
| `refine_tolerance` | `0.25` | full-res cells this close to the DTM stay ground, metres |
| `low_outlier` | `1.0` | metres below local median counted as a low blunder |
| `outlier_radius` | `3.0` | neighbourhood radius for the low-outlier test, metres |
| `object_buffer` | `1.0` | grow detected objects by this before interpolating, metres |
| `smooth` | `1.0` | box-smoothing window of the final DTM, metres (`0` = off) |

Also public in `terrain.py`: `classify_ground(Z, cell, opt) -> bool mask` (the morphological filter
alone).

### 3.7 Thermal palettes

DJI radiometric R-JPEGs (M30T, M3T, M4T, H20T, …) carry the raw 16-bit sensor image. The pipeline
mosaics raw values on one survey-wide scale (palette colours are never blended, exposure comp
disabled), applies the palette to the finished product, and writes a float32 raw-value raster so the
palette can be changed later without re-processing.

```python
palette_names() -> list[str]
apply_palette(gray, palette="rainbow") -> np.ndarray          # uint8 index image -> (...,3) RGB
recolor(thermal_tif, out_tif, palette="rainbow",
        value_range=None, legend_png=None) -> str             # re-render a *_thermal.tif raster
```

- `palette_names()` — the built-in palette names.
- `apply_palette(gray, palette)` — maps a uint8 index image to RGB via a 256-entry LUT.
- `recolor(...)` — re-colours a raw-value thermal GeoTIFF (produced by `build_orthomosaic` /
  `build_3d`) with another palette, optionally writing a legend PNG. `palette` may be a name, a list
  of `(R,G,B)` stops, or a full `(256,3)` table.

**Built-in palettes:** `rainbow` (default), `rainbow_hc`, `iron`, `white_hot`, `black_hot`, `arctic`,
`lava`, `hot_metal`, `medical`, `green_hot`. Aliases like `ironbow`, `whitehot`, `grayscale`,
`fusion` map onto these.

> Raw values are the camera's sensor units (monotonic with temperature). Converting to degrees needs
> the DJI Thermal SDK radiometric calibration, which is not performed here.

Related functions in `thermal.py`: `palette_lut`, `colorize_rgba`, `legend`, `raw_thermal_shape`,
`read_raw_thermal`, `raw_to_gray`, `survey_range`.

### 3.8 Ground control points: `load_gcps`

```python
load_gcps(path) -> GCPSet
```

Parses a **WebODM / Pix4D GCP list** (one row per image mark). Passing `gcp=...` and `gcp_sigma=...`
in `Options3D` anchors the bundle block to real-world coordinates and removes doming, taking a
GPS-only block (metre-level) to survey-level accuracy.

**File format**

```
EPSG:32646                                          # or WGS84 / +proj=longlat (lat/lon auto-projected)
geo_x geo_y geo_z im_x im_y image_name [gcp_name]
230012.5 2635008.1 12.3 2456.0 1810.5 DJI_0001_V.JPG GCP1
230012.5 2635008.1 12.3 1203.0  905.0 DJI_0002_V.JPG GCP1
```

Columns are whitespace- or comma-separated; `#` lines are comments; marks are grouped by explicit
GCP name (or shared coordinate). Three or more well-spread GCPs suffice; marks are never rejected as
outliers. `report.json` records per-GCP world error and the block's control RMSE.

**Data types** (`gcp.py`): `GCP(name, world, marks)`, `GCPSet(gcps, epsg, is_latlon)` with
`n_marks`. `to_local(gset, zone, north, origin)` converts to the reconstruction's local frame (used
internally).

### 3.9 Backend selection

```python
select_backend(prefer="auto")  -> backend object
cuda_available()               -> bool
```

- `select_backend(prefer)` — `prefer` is `auto` (CUDA → Apple GPU → CPU), `cuda`, `mps`, or `cpu`.
  In `auto`, GPU backends self-test at startup and fall back to CPU with an install hint on failure;
  when a specific GPU is requested and unavailable, it raises.
- `cuda_available()` — whether an NVIDIA device and a working CuPy are present.

Also in `backend.py`: `mps_available()`, and the backend classes `CPUBackend` / `CUDABackend` /
`MLXBackend`, all exposing the same interface (`match`, `match_mutual`, `new_block`, `upload`,
`warp_accumulate`, `finalize`, `sample_view`, `box`, `to_numpy`, `asarray`, `argmin0`, `argmax0`,
`topk_mean`). Constants `MODE_FEATHER = 0`, `MODE_MAX = 1`.

### 3.10 `write_quality_report` — processing quality report

```python
write_quality_report(report, out_html, product_dir=None, title=None) -> str
```

Renders an **Agisoft Metashape / Pix4D-style HTML quality report** from a `build_3d` result. It is
purely a renderer of the existing `report.json`, so it adds no reconstruction cost.

**Parameters**

- `report` (`dict | str`): the dict returned by `build_3d`, or a path to a `report.json`.
- `out_html` (`str`): output HTML path.
- `product_dir` (`str | None`): folder holding the previews (`orthophoto_preview.jpg`,
  `dsm_preview.png`, `dtm_preview.png`) to embed as base64. Defaults to the `report.json` folder (or
  the `out_html` folder for a dict). Missing previews are skipped gracefully.
- `title` (`str | None`): report heading.

**Report sections:** survey summary (images, CRS, area, GSD, mean camera height, datum, backend,
time), a top-down **camera-location plot** (inline SVG, coloured by height), reconstruction quality
(tie points, projections, mean track length, verified pairs, reprojection RMS), a **camera
calibration** table (sensor groups: resolution, focal, principal point, k1/k2), **ground control**
accuracy (per-GCP ΔX/ΔY/ΔZ and RMSE 3D/H/V, when GCPs were used), **digital models** (DSM/DTM stats
plus embedded previews), an optional thermal section, and the full **processing parameters**.

The output is one self-contained, print-friendly HTML file (no external assets). CLI:

```bash
python -m orthomosaic.qcreport recon/ -o recon/quality_report.html
python -m orthomosaic.qcreport recon/report.json          # -> recon/quality_report.html
```

See [`example_quality_report.py`](example_quality_report.py) for a driver that reconstructs a folder
and writes the report in one step.

---

## 4. Command-line interfaces

Two console scripts are installed, plus module entrypoints:

```bash
orthomosaic <images> -o ortho.tif                 # 2D  (== python -m orthomosaic)
orthomosaic-3d <images> -o recon/                 # 3D  (== python -m orthomosaic.reconstruct)
```

**`orthomosaic` (2D) flags:** `--backend {auto,cuda,mps,cpu}`, `--workers N`, `--resolution M`,
`--render-scale F`, `--feature-max-dim N`, `--features N`, `--neighbors N`, `--gps-sigma M`,
`--blend {feather,seam}`, `--no-exposure`, `--cache-mb N`, `--no-preview`, `--palette NAME`,
`--no-thermal`, `-v/--verbose`.

**`orthomosaic-3d` (3D) flags:** `--backend`, `--workers`, `--dsm-resolution M`, `--gps-sigma M`,
`--max-views N`, `--min-score S`, `--cloud-step N`, `--mesh-max-vertices N`,
`--formats ply,las,obj,glb`, `--no-dtm`, `--dtm-max-object M`, `--cache-mb N`, `--palette NAME`,
`--no-thermal`, `--gcp FILE`, `--gcp-sigma M`, `--bind-thermal THERMAL_IMAGES`, `-v/--verbose`.

**Terrain (standalone DTM from an existing DSM):**

```bash
python -m orthomosaic.terrain dsm.tif -o dtm.tif --max-object 80 [--slope S] [--ndsm ndsm.tif]
```

**Thermal palettes:**

```bash
python -m orthomosaic.thermal list
python -m orthomosaic.thermal recolor ortho_thermal.tif -o ortho_iron.tif --palette iron [--range MIN MAX]
```

---

## 5. Output products

**2D (`build_orthomosaic`):**

| File | Content |
|---|---|
| `ortho.tif` | Tiled, Deflate-compressed RGBA GeoTIFF in UTM (EPSG:326xx/327xx) |
| `ortho_preview.jpg` | Preview ≤ 2048 px |
| `ortho_report.json` | Affines, gains, statistics, options |
| `ortho_thermal.tif`, `ortho_legend.png` | (radiometric thermal only) raw float32 raster + colour bar |

**3D (`build_3d`), all in the same local height datum:**

| File | Content |
|---|---|
| `dsm.tif` | Digital Surface Model — float32 elevation incl. buildings/trees (nodata −9999) |
| `dtm.tif` | Digital Terrain Model — bare ground |
| `ndsm.tif` | Height above ground (DSM − DTM) |
| `orthophoto.tif` | **True** orthophoto rendered on the DSM (buildings don't lean) |
| `dense.las`, `dense.ply` | Coloured dense point cloud (LAS 1.2 w/ EPSG, classified ground=2 / unclassified=1; binary PLY) |
| `mesh.glb`, `mesh.obj` | Textured surface mesh (glTF 2.0; OBJ+MTL+JPEG) |
| `sparse.ply` | SfM points |
| `report.json` | Poses, intrinsics, statistics |
| `dsm_preview.png`, `orthophoto_preview.jpg`, `dtm_preview.png` | Quick looks |

---

## 6. Internal modules (developer reference)

These are not part of the public `__all__` but define the pipeline internals.

- **`pipeline.py`** — `Options`, `AlignResult`, `align_images(images, opt)` (metadata → features →
  matching → global alignment), `build_orthomosaic`.
- **`reconstruct.py`** — `Options3D`, `sparse_block(images, opt)` (aerial triangulation → returns
  `(AlignResult, Reconstruction)`), `build_3d`, `build_thermal_bound`, `_products` (product export).
- **`align.py`** — `PairMatch`, `candidate_pairs`, `match_pair`, `largest_component`, the `_Normal`
  linear system, `Alignment`, `solve_alignment`, `pair_residuals_px`, `solve_gains` (exposure).
- **`features.py`** — `Features`, `extract(frame, max_dim, n_features, levels)` (ORB-style), grid
  bucketing, steered BRIEF pattern.
- **`imageio.py`** — `Frame` (EXIF/XMP GPS, altitude, raw-thermal shape), `list_images`, `read_frame`,
  `load_rgb`. Reads DJI XMP for relative/absolute altitude.
- **`geo.py`** — built-in UTM: `utm_zone`, `utm_epsg`, `latlon_to_utm`, `utm_to_latlon` (transverse
  Mercator, no PROJ).
- **`geotiff.py`** — `GeoTIFFWriter` (tiled, Deflate, GeoTIFF tags) and `read_geotiff`. No GDAL.
- **`render.py`** — `ImageCache` (LRU by byte budget) and `render(...)` (parallel blocked warp/blend,
  serpentine order, streamed to GeoTIFF).
- **`camera.py`** — `Intrinsics`, `focal_from_exif`, `group_intrinsics`, `rodrigues`,
  `orthonormalize`, `project`, `undistort_normalized`, `poses_from_alignment`.
- **`sfm.py`** — `Reconstruction`, track building, `triangulate`, `Priors`, `bundle_adjust`
  (LM + Schur + Huber), `_add_gcps`, `gcp_report`, `reconstruct(ar, ...)`.
- **`mvs.py`** — `DenseOptions`, `DenseResult`, `dense_reconstruct`, `postprocess`, `fill_holes`,
  `_nanmedian_filter` (dense stereo + surface cleanup).
- **`export.py`** — `write_ply`, `write_las`, `grid_mesh`, `write_obj`, `write_glb`.
- **`binding.py`** — `pair_key`, `pair_frames`, `Rig`, `estimate_rig`, `apply_rig`, `bind_thermal`
  (RGB→thermal rig binding).
- **`gcp.py`** — GCP parsing and local-frame conversion (see §3.8).
- **`terrain.py`** — DTM (see §3.6).
- **`thermal.py`** — palettes and radiometric R-JPEG decoding (see §3.7).
- **`qcreport.py`** — `write_quality_report` and the `python -m orthomosaic.qcreport` CLI (see §3.10).
- **`backend.py`** — compute backends (see §3.9).

---

## 7. Native (Cython) kernels

Compiled from `.pyx`; GIL-releasing, called through the backends.

- **`_core.pyx`** — `rgb_to_gray`, `gaussian_blur`, `resize_bilinear`, `detect_corners` (FAST+Harris),
  `orientations` (intensity centroid), `brief_describe` (256-bit steered BRIEF), `match_hamming`,
  `match_hamming_mutual`, `ransac_affine`, `warp_accumulate`, `finalize` (blend).
- **`_ba.pyx`** — `reduced_system` (Schur complement), `back_substitute`, `residuals` — the bundle
  adjustment's hot loops.
- **`_mvs.pyx`** — `sample_view` (multi-view resampling at height hypotheses), `sgm` (8-direction
  semi-global matching), `bilateral` (edge-preserving surface filter).

Equivalent CUDA and MLX kernels live in `_cuda.py` and `_mlx.py`; `tests/test_backends.py` checks
every GPU kernel against the CPU implementation. The CUDA plane sweep of the depth maps lives in
`_gpu_depth.py` (plain CUDA C, compiled with `--fmad=false`): it reproduces
`_dense.sweep_hypothesis` bit for bit, which `tests/test_gpu_depth.py` checks on the device and,
by compiling the same source with a C++ compiler, on any machine.

---

## 8. Environment variables

| Variable | Effect |
|---|---|
| `ORTHO_NATIVE=1` | build with `-march=native` (build time) |
| `ORTHO_DISABLE_CUDA=1` | force `cuda_available()` to report `False` |
| `ORTHO_DISABLE_MPS=1` | force `mps_available()` to report `False` |
| `CUDA_PATH` / `CUDA_HOME` | point CuPy at a CUDA toolkit for runtime kernel compilation (auto-detected if unset) |

---

## 9. Scope and limitations

- **2D mode** is a planar mosaic — fast, but tall objects lean. Use the 3D true orthophoto for relief.
- **3D mode** is 2.5D (one height per ground cell). Walls, overhangs, and oblique/orbit flights are
  not modelled.
- **Thermal** imagery has little texture, so its DSM is less reliable than RGB — prefer the RGB DSM
  for geometry (hence thermal binding).
- **Gimbal pitch** need not be nadir (tilts are solved in bundle adjustment), but strongly oblique
  shots should be excluded.
- **Absolute accuracy** without GCPs or RTK is limited by the onboard GPS (metre-level). Add GCPs for
  survey-grade results.
- **Radiometric-to-temperature** conversion is not performed; raw sensor units are stored.
