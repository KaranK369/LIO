# Degenerate environments (runway, tunnel, bridge): what was wrong and what changed

This page covers the GEODE bags `bridge2` and `urban_tunnel` (VLP-16 + Xsens) and
`runwayD_ros2` (Ouster OS0-128 + internal IMU). Every claim below was checked on
those bags (see *Results*).

## Root causes found

| # | Where | Problem | Effect |
|---|---|---|---|
| 1 | launch → `cloud_preprocess` | `sensor_height: 0.53` was hard-coded (the Copernicus robot). The GEODE sensors sit about 1.7–1.9 m up, so the ground-fit elevation gate rejected every bin, and the band filter (fallback −0.53 m) **deleted all ground**. | FAST-LIO lost the ground. On the runway, almost no points were left, and the cloud swung between full and near-empty |
| 2 | `cloud_preprocess` | `band_max: 3.0` (a forest canopy cut) deleted tunnel ceilings, bridge trusses and hangars. Rigidity, DROR and dynamic removal deleted corners and isolated far returns | the only along-axis features were removed |
| 3 | `cloud_preprocess` | Velodyne `time` runs −0.1…0 s with the stamp at scan END. The time was renormalised but the stamp was never shifted (`shiftStamp()` was dead code) | 100 ms LiDAR↔IMU offset |
| 4 | `loop_closure_node` | Only 30 keyframes (≈30 m) were excluded from the loop search. Scan Context plus GICP on a flat runway or in a tunnel "verifies" kf *i* against kf *i−31* with ~0 translation (log: `trans=0.02 (odom gap 24.7m)`). The "PCM" never used the loop transform, so consecutive aliases always agreed | **false loops snapped the graph** |
| 5 | `backend_node` | Loops were attached to the nearest keyframe *by position* (often an old keyframe, up to 3 m off). Huber never rejects anything, factors were never removable, and POOR scenes inflated odometry 21× but loops only 4× | loops won exactly where they were least reliable |
| 6 | router | Ran on a single, already-filtered scan (a VLP-16 ring arc reads as a "line" everywhere), used x/y only, a condition ratio only, and had no hysteresis | misclassification |
| 7 | FAST-LIO | No degeneracy handling. In along-axis-degenerate scenes **pitch is unobservable**: tilt error leaks into along-axis acceleration, which the LiDAR cannot see, so map and pose tilt together (bridge2: 10° off the AHRS after 130 s) | z ran away (bridge2: 1.9 km of z) |
| 8 | `fast_lio_runwayD.yaml` | `extrinsic_T` z = +0.108 was the total-station **prism**, not the LiDAR | 11.6 cm lever-arm error |
| 9 | runwayD data | The platform spins (−1540° of yaw in total) and the Ouster gyro **clips at ±250°/s** (85 samples) on a flat runway with ground only within ~12 m | unmeasured yaw that no sensor can recover |

## What changed

**FAST-LIO (`src/fast_lio`)**
- Publishes `/localizability` (`fast_lio/Localizability`). This is the normalised 6-DOF
  information of the point-to-plane Jacobian the iEKF actually used, in the world frame.
  The translation eigenvalues sum to 1.
- `degeneracy.*`: degeneracy-aware update. In **soft** mode (the default), a direction whose
  eigenvalue is below the threshold has its information scaled by λ/threshold. A **hard**
  projection (Zhang 2016) diverged on the runway, where degeneracy is persistent, because
  velocity was left unobserved. In rich scenes nothing changes.
- `attitude.*`: gravity-referenced roll/pitch aiding. Its weight is κ × the LiDAR's own
  roll/pitch information, with a floor. Source 1 is the IMU's AHRS orientation (Xsens).
  Source 2 is the level-ground plane normal (Ouster has no orientation).
- `mapping.imu_init_samples`: number of static start-up samples (stock FAST-LIO uses 10).
- `mapping.gyro_saturation`: rotation process noise is inflated while the gyro is clipped.
- `/Odometry` is now published *after* its covariance is filled.

**Stage A (`cloud_preprocess_node`, `ground_segmentation.hpp`)**
- The stamp is shifted to the first point (`shift_stamp_to_first_point`).
- `sensor_height <= 0` → **auto height** (median of upright, flat near bins over 20 scans).
- Fail-safe: if less than `min_ground_bin_frac` of bins are accepted as ground, the band
  filter is skipped. `band_max <= 0` → no upper cap.
- **Scene-adaptive**: while `/environment/state` is SPARSE/POOR, only the range crop runs.
- Dynamic removal uses the full LiDAR→IMU extrinsic (taken from the FAST-LIO config) and a
  pose extrapolated to the scan time.

**Loop closure (`loop_closure_node`)**: the cloud and odometry are synced by stamp, then
each candidate must pass:
- a minimum *travelled path* to the match;
- a descriptor-informativeness gate (flat ground never searches);
- GICP against a submap, from two inits, accepted on overlap + RMSE;
- a **drift gate** (registration vs odometry within `drift_ratio·L + floor`);
- an **observability gate**: point-to-plane information is eigen-analysed, and the
  information sent to the backend is ≈0 along degenerate axes;
- real PCM on loop transforms.

Loops that leave the PCM clique are **retracted**. `LoopConstraint` gained `loop_id`,
`retract`, full poses, `information[36]`, `overlap` and `path_length`.

**Backend (`backend_node`)**
- Loops are matched **by stamp**, with exact offset compensation.
- **χ² gate** against the current estimate plus a drift budget, before insertion.
- Anisotropic loop noise taken from the detector.
- Post-optimisation residual check, and factor removal on retraction.
- Odometry sigma grows with motion. It is inflated (capped at 10×) along the directions
  `/localizability` reports as weak over the keyframe interval, including yaw.
- Without loops or other absolute factors the optimum *is* the odometry. The backend cannot
  remove front-end drift by itself, which is why the drift fixes live in FAST-LIO.

**Router (`environment_router_node`)**
- The primary signal is FAST-LIO `/localizability` (6-DOF, averaged over 1 s).
- Dimensionality runs on a cloud accumulated across the keyframe interval.
- Hysteresis: degrade after 1 keyframe, recover after 3.
- `EnvironmentState` gained `rot_axes`, `rot_eigvals` and `source`. The cloud fallback is kept.

**Configuration.** All node tuning now lives in `config/platform_<name>.yaml`
(`copernicus`, `geode_vlp16`, `runwayD`). Launch-time dicts used to override YAML. The
launch now passes only topics, `use_sim_time`, the LiDAR extrinsic (read from the FAST-LIO
config) and switches you set explicitly. A new `fast_lio_urban_tunnel.yaml` was added.
**Note:** `fast_lio_copernicus.yaml` currently contains the *runwayD Ouster* config. The
robot's original is in `fast_lio_copernicus.yaml.phasma_backup`.

## Run

```bash
ros2 launch copernicus_localization copernicus_localization.launch.py config_file:=lio_bridge2.yaml enable_loop:=true
ros2 launch copernicus_localization copernicus_localization.launch.py config_file:=lio_urban_tunnel.yaml enable_loop:=true
ros2 launch copernicus_localization copernicus_localization.launch.py config_file:=lio_runwayD.yaml enable_loop:=true
#   platform:=<copernicus|geode_vlp16|runwayD>   (default derived from config_file)
#   params_file:=/abs/path.yaml                 (overrides platform)
ros2 bag play /home/karan_k/AIRL/bridge2 --clock
```

Watch `/localizability` (degeneracy), `/environment/state`, and the `loop_debug` log, where
every rejected candidate is printed with its reason.

## Results (1× playback; old = code before this change)

| Bag | Metric | Old | New |
|---|---|---|---|
| runwayD (GT) | ATE, arc-length associated, SE(3) aligned | 58.9 m (LIO) / 48.3 m (backend) | **38.5 m** |
| | start–end closure (GT 0.12 m) | 130.8 m | 107.7 m |
| | backend path length (GT 357 m) | **2390 m** (false-loop teleports, 244 jumps >1 m) | 345.6 m, 0 jumps |
| | false loops applied | 4 | 0 |
| urban_tunnel (no GT) | LIO path / z-span | **diverged**: 183 km / 128 km | 1.98 km / 134 m |
| | loops | 15 aliases applied (e.g. 25 m odometry gap, "0.12 m" by ICP) | 15 cross-section-only loops (`degen_axes=1`), 13 rejected as drift-implausible |
| bridge2 (no GT) | LIO path / z-span | **diverged**: 483 km / 463 km | 6.39 km / 86 m |
| | pitch vs Xsens AHRS | — | within 1–3.5° over 472 s (10.5° after 130 s without attitude aiding) |

**Hilti exp07 long_corridor** (`config_file:=lio_hesai.yaml` → `platform_hilti_handheld.yaml`).
Evaluated against the 6 surveyed, timestamped control points (SE(3) aligned):

| Config | Points | RMSE | Max | Rate |
|---|---|---|---|---|
| old code, user's partial config | 5/6 | 0.11 m | 0.19 m | **~1.4 Hz** (183 poses in 128 s, 49 jumps >1 m) |
| new code, user's partial config | 6/6 | 0.10 m | 0.19 m | 10 Hz |
| new code, `fast_lio_hesai.yaml` | 6/6 | **0.08 m** | **0.12 m** | 10 Hz, scale error 0.1 % |

Properties of this rig, checked on the bag:
- The XT-32 and the IMU are mounted upside down, so ground segmentation is off.
- 33–53 % of every scan is the operator or device within 1 m, in fixed sectors, so the
  range crop and `blind` are set to 1.0 m.
- Hesai stamps equal the first point. `cloud_preprocess` now re-stamps to the first
  *kept* point, because FAST-LIO's Hesai handler measures time offsets from it.
- The Alphasense IMU has no orientation, so attitude aiding uses the floor plane. The IMU
  runs at 400 Hz, so `imu_init_samples: 400`.

Ablations on the same bags:
- **Hard remapping:** the runway diverged to 18.6 km.
- **No attitude aiding:** bridge2 z-span was 1.89 km and the tunnel z-span 205 m.
- **Stamp shift off:** bridge2 window z −112 m (on: −90 m).

Synthetic checks:
- `synth_backend_test.py`: the true loop closes 8.92 m of drift to 0.00 m, and a false
  loop 28 m off is rejected (χ² 330 > 22.5).
- `router_test.py`: 6/6 scenes pass (cloud fallback path).

## Known limits

- **runwayD is sensor-limited.** The gyro clips at 250°/s while the platform spins, and
  yaw on a flat runway is barely observable. Saturation-aware noise helps a little; the real
  fix is an IMU with a larger range or a heading source.
- Long featureless stretches stay IMU-carried along the axis. To bound along-axis drift you
  need another velocity source: wheel odometry, VO from the GEODE stereo cameras, or GNSS at
  the exits. The backend's anisotropic odometry is ready for such factors.
- The level-ground attitude source (2) assumes average slopes of a few degrees. Keep it off
  (`attitude.source: 0`) in hilly off-road terrain without an AHRS.
