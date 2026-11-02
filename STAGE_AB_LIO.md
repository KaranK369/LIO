# Stage A + B — LiDAR-Inertial Front-End (implemented)

> **Update:** for degenerate scenes (runway / tunnel / bridge), the per-platform parameter
> files, FAST-LIO degeneracy + attitude aiding and the reworked loop/router/backend, see
> [`DEGENERATE_ENVIRONMENTS.md`](DEGENERATE_ENVIRONMENTS.md). Node parameters now live in
> `config/platform_<name>.yaml`, not in the launch file.

This implements **Stage A (point-cloud preprocessing)** and **Stage B (FAST-LIO2
front-end)** of the GNSS-denied forest localization methodology
(`../../GNSS_DENIED_LOCALIZATION_METHODOLOGY.md`). Stages C–E (trunk landmarks,
loop closure, GTSAM backend) are not implemented yet.

Platform: ROS 2 Humble, Ubuntu 22.04. Sensors: Velodyne VLP-32 (`/velodyne_points`),
Xsens IMU @ ~400 Hz (`/imu/data`), wheel odom (`/copernicus_base_controller/odom`).

```
/velodyne_points ─▶ cloud_preprocess (Stage A) ─▶ /velodyne_points_processed ─┐
                                                                /imu/data ─────┤
                                                                               ▼
                                                          FAST-LIO2 (Stage B)
                                                  /Odometry, /cloud_registered,
                                                  /path, TF camera_init→body
```

## Packages

| Package | Role |
|---|---|
| `fast_lio` | Vendored **FAST-LIO2** (hku-mars, ROS2 branch). Uses **ikd-Tree** (`KD_TREE`, incremental add/delete/NN) + **iterated error-state EKF** (`esekfom::esekf`, `update_iterated_dyn_share_modified`), direct method (`feature_extract_enable: false`). |
| `livox_ros_driver2` | Interface-only **stub** (CustomMsg/CustomPoint). FAST-LIO2 hard-depends on these msgs for its (unused) Livox path; the stub satisfies build/link without Livox-SDK2. The robot is Velodyne-only. |
| `copernicus_localization` | Our integration: Stage A node (`cloud_preprocess_node`), FAST-LIO config, launch. |

## Stage A — `cloud_preprocess_node` (`src/cloud_preprocess_node.cpp`)

Subscribes `/velodyne_points` (and `/Odometry`, for dynamic removal only),
publishes `/velodyne_points_processed` preserving all fields by name
(`x y z intensity time ring`) so FAST-LIO2 consumes it exactly as the raw topic.
This now covers **all five** methodology Stage-A steps. Algorithm detail for the
heavier steps lives in headers under `include/copernicus_localization/`:
`ground_segmentation.hpp`, `dynamic_removal.hpp`.

0. **Time fix (always on, REQUIRED).** This Velodyne driver stamps each cloud at
   *scan end* and writes per-point `time` as a **negative** offset (≈ −0.1…0 s,
   seconds). FAST-LIO2 assumes a **positive** offset from *scan start*
   (`lidar_end_time = header.stamp + last_point_time`). Fed the raw cloud it
   collapses the deskew window to ~0.6 ms and mis-aligns LiDAR↔IMU by a full scan.
   We renormalise `time' = time − min(time)` and shift `stamp' = stamp + min(time)`,
   preserving absolute per-point times → correct deskew + IMU sync. Hence
   `timestamp_unit: 0` (seconds) in the FAST-LIO config.

**Spatial filtering (toggle `enable_filtering`), in data-flow order:**

1. **Range crop** — NaN drop + Euclidean gate (`min_range` 0.8, `max_range` 80 m):
   removes ego/mount returns and far-sparse noise.
2. **Multi-return / intensity** (`enable_intensity_filter`, default **off**) — drops
   weak returns below `min_intensity` (foliage/dust/edge points). If the driver runs
   dual-return AND exposes an echo field (`return_type`/`echo`/…), the weak echoes are
   the low-intensity duplicates removed by the same gate (detected & logged once).
   Zero extra compute; conservative default keeps everything until tuned.
3. **Ground segmentation** (`enable_ground_seg`, default **on**) —
   **Patchwork++-style** (in-house: Concentric Zone Model + region-wise plane fit +
   uprightness/flatness/elevation gating). Labels rigid ground vs non-ground and
   yields a slope-aware `groundHeightAt(x,y)`. Replaces the old min-z-grid heuristic,
   which biased the ground to a cell's downhill corner on off-road slopes.
4. **Lower-trunk band** — keep ground (`keep_ground`) + non-ground within
   `[band_min, band_max]` above the **fitted** ground. Default `band_min=0.0,
   band_max=3.0` keeps ground + trunks, drops swaying canopy — the cheap primary
   wind defense; ground retained as a hedge against clearing degeneracy. For a pure
   Stage-C trunk band: `keep_ground=false, band_min=0.5, band_max=2.0`.
5. **Rigidity / foliage rejection** (`enable_rigidity`, default **off**) — local
   structure-tensor PCA over a 0.4 m neighbourhood **on the non-ground band**; keep
   linear (trunks)/planar, drop scattered/volumetric (foliage). Heaviest per-point
   step; enable on Orin after profiling.
6. **Dynamic-object removal** (`enable_dynamic_removal`, default **off**) —
   **Dynablox-style** temporal free-space/occupancy voxel map in the world frame,
   at **keyframe rate** (gated by `dynamic_keyframe_dist`/`_time`). A return landing
   in an "ever-free" voxel (one we repeatedly saw *through*) is a moving-into-free-
   space candidate, confirmed by clustering (`dynamic_min_cluster_points`). Needs
   ego-pose from `/Odometry`; no-op until the front-end has converged. Targets
   *translating* actors only — in-place wind jitter is handled by 3–5, not here.

QoS is **reliable** (matches the bag's offered profile and FAST-LIO2's reliable
lidar subscription — a best-effort publisher would deliver nothing to FAST-LIO).

## Stage B — FAST-LIO2 config (`config/fast_lio_copernicus.yaml`)

- `lid_topic: /velodyne_points_processed`, `imu_topic: /imu/data`
- `lidar_type: 2` (Velodyne), `scan_line: 32`, `scan_rate: 10`, `timestamp_unit: 0`
- **Extrinsic (LiDAR in IMU frame)** from URDF: base_link→velodyne=(−0.05185,0,0.650),
  base_link→imu_link=(−0.05185,0,0.368) ⇒ `extrinsic_T: [0,0,0.282]`, `extrinsic_R: I`.
- Full 3D (no `two_d_mode`), `cube_side_length: 500` (bounded local map),
  `max_iteration: 4`, `pcd_save_en: false`.

## Run

```bash
cd ~/AIRL/copernicus_ws
colcon build
source install/setup.bash

# Terminal 1 — pipeline (RViz fixed frame = map)
ros2 launch copernicus_localization copernicus_localization.launch.py
#   enable_filtering:=true|false   enable_rigidity:=true|false
#   enable_denoise:=true|false     (DROR particulate removal; snow/dust/rain)
#   enable_dynamic:=true|false     (Dynablox dynamic removal; uses /Odometry)
#   rviz:=true|false   use_sim_time:=true

# Terminal 2 — play the recording (publishes /clock)
ros2 bag play ~/AIRL/off_road2 --clock
```

## Verification — `off_road2` bag (first ~104 s, ~30 m traverse)

FAST-LIO2 output `/Odometry` vs wheel odometry:

| Config | 3D path | end (x,y,z) | z span | LIO/wheel path | NaNs | jumps>1 m |
|---|---|---|---|---|---|---|
| Baseline (time-fix only) | 30.18 m | (10.57, 22.56, 0.68) | 0.69 m | 1.023 | 0 | 0 |
| Filtered (ground+canopy band) | 30.20 m | (10.58, 22.52, 0.70) | 0.71 m | 1.024 | 0 | 0 |

- Continuous **10 Hz** odometry, **0 NaNs, 0 jumps**, smooth (max step 0.11 m).
- **Full 3D**: z tracks ~0.7 m of terrain relief (not collapsed to a plane).
- **Scale within ~2–4%** of wheel-odometry path length → translation/deskew correct.
- Baseline vs filtered endpoints agree to **~4 cm over 30 m** → Stage A filtering is
  safe (no degradation); on a non-windy segment it changes little, as expected.
- Wheel-odom yaw span (170°) ≫ LIO (114°): skid-steer wheel yaw is unreliable
  off-road (per methodology); not used as a reference for yaw.

Stage A point reduction (forest frames, ~52.7 k finite pts/scan):

| Stage A config | kept | removed |
|---|---|---|
| NaN + time-fix only | 90.9% | 9.1% (invalid returns) |
| + below-canopy band | 89.7% | 10.3% (canopy >3 m is small for a 0.65 m-high sensor) |
| + rigidity (full) | 69.9% | 30.1% (scattered foliage — the real wind lever) |

## Rating of initial localization

**Solid baseline LiDAR-inertial odometry — works, no divergence over the tested leg.**
On the ~30 m segment it is locally accurate and metrically consistent (scale error
~2–4% vs wheel odom, correct 3D terrain, no glitches). This is a strong Stage-B
result. **Caveat:** this only proves *local* odometry quality. Global drift is
unevaluated — there is no ground truth (RTK/total-station) and no loop closure yet,
so long-horizon x/y/yaw drift is unbounded by design (methodology §6) until Stages
D/E land. To rate global accuracy, capture an out-and-back / revisit bag and ground
truth, then report ATE/RPE.

## Critical invariant — preserve per-point scan order

Stage A **must emit points in their original per-scan order**. FAST-LIO2 derives the
scan-end time from the **last point's timestamp** (`laserMapping.cpp`,
`points.back().curvature`); reordering the cloud feeds it an arbitrary end time →
IMU/LiDAR desync → the front-end gets stuck at the origin (RViz shows scattered raw
ring sweeps, no map). All Stage-A steps therefore filter in place (per-point keep
masks), never regroup points. Do not "optimize" by emitting ground-then-non-ground.

## Performance (measured, this host, full pipeline alongside FAST-LIO2)

| Stage-A config | per-scan cost | rate | FAST-LIO tracking |
|---|---|---|---|
| ground-seg + band | ~7–12 ms | ~10 Hz | ✅ |
| + rigidity (voxel) | ~11–18 ms | ~10 Hz | ✅ |
| + dynamic removal | ~13 ms (≈27 ms on keyframes) | ~10 Hz | ✅ |

Rigidity is **voxel-based** (per-voxel structure tensor over a 3×3×3 neighbourhood,
one eigen-decomposition per voxel) — ~3 ms. The earlier per-point kd-tree radius
search was ~950 ms/scan (~1 Hz) and starved FAST-LIO; do not reintroduce it.
The default launch keeps the real-time core enabled (`enable_filtering`) and leaves
rigidity, denoise, and dynamic removal off unless explicitly needed;
`cloud_preprocess` logs ms/scan and point reduction every 2 s for quick Hz checks.
Build `copernicus_localization` as **Release**. An unoptimized build was measured at
~270–350 ms/scan on `off_road2`, which drops the stack to ~3–4 Hz.

## Known limitations / next

- No ground truth on `off_road2` ⇒ relative (vs wheel) + self-consistency only. (A
  separate RTK run gave **ATE 0.53 m / 435 m**; see `traj_compare.py`.)
- Default Stage A now runs **real-time** alongside FAST-LIO2 (~7–12 ms/scan,
  ~10 Hz); enable denoise/dynamic only for adverse weather or moving-actor tests.
- Ground segmentation is an **in-house Patchwork++-style** implementation (CZM +
  R-GPF + uprightness/flatness/elevation gating), not the upstream library; the
  A-GLE (fully adaptive thresholds) and TGR (temporal ground revert) refinements
  are not implemented — thresholds are exposed as parameters instead.
- Dynamic removal is **scan-pre-filter using delayed odometry** (uses the latest
  `/Odometry`, ~1 scan stale); it is a lightweight take on Dynablox, not the full
  TSDF system.
- Robust point-to-plane kernel (Huber/Cauchy) from the methodology is not exposed
  by stock FAST-LIO2 config; would need a code patch.
- Next per build order: GTSAM backend (Stage E plumbing) → trunk landmarks (C) →
  Scan-Context + constellation loop closure (D).
