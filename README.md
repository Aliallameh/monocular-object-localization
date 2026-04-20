# Monocular Bin Tracking & 3-D Localization

Real-time single-camera pipeline that detects a garbage bin, estimates its
world-frame position via ground-plane ray casting, and streams per-frame
coordinates to stdout — no GPU, no pre-trained model weights.

```
bash run.sh --video input.mp4 --calib calib.json
```

```
[frame 0001] bin @ world (2.44, -0.04, 0.33) m  conf=0.92  state=STATIONARY  dt=58ms
[frame 0002] bin @ world (2.44, -0.04, 0.33) m  conf=0.92  state=STATIONARY  dt=31ms
...
[summary] frames=875  detector_hits=100.0%  tracker_outputs=100.0%  mean_dt=33.9ms
```

---

## Table of Contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Detection](#detection)
- [Localization](#localization)
- [Tracking and Kalman filter](#tracking-and-kalman-filter)
- [Occlusion handling](#occlusion-handling)
- [Results and benchmarks](#results-and-benchmarks)
- [Output format](#output-format)
- [Waypoint evaluation](#waypoint-evaluation)
- [Jetson deployment notes](#jetson-deployment-notes)
- [Known limitations](#known-limitations)

---

## Quick start

Requirements: Python 3.10+, OpenCV, NumPy (auto-installed by `run.sh`).

`calib.json` and `input.mp4` are **not committed** — provide them at runtime:

```bash
git clone https://github.com/Aliallameh/monocular-object-localization.git
cd monocular-object-localization
bash run.sh --video input.mp4 --calib calib.json
```

Validate:

```bash
.venv/bin/python tools/validate_submission.py
.venv/bin/python -m unittest discover -s tests -v
```

---

## How it works

```
Frame -> Detector -> Tracker -> Localizer -> Kalman smoother -> stdout + CSV
          (4 cues)   (IoU +     (ground-       (constant-vel.
                      flow)      plane ray)      6-DOF state)
```

| Module | File | Role |
|---|---|---|
| Detection | `detector.py` | Multi-cue proposals |
| Tracking | `tracker_utils.py` | BBox Kalman + LK flow |
| Localization | `localizer.py` | Camera-to-world projection |
| Visualization | `observer.py` | Overlay video + event log |

---

## Detection

A four-channel classical detector, no learned model weights:

| Channel | What it catches |
|---|---|
| Blue HSV segmentation | Saturated blue body |
| Dark rectangular shape | Gray/dark bin on light floor |
| Edge/Canny shape | Blurred or desaturated frames |
| MOG2 motion foreground | Any moving object as safety net |

Candidates from all four channels are merged with NMS (IoU threshold 0.45).

**Why not YOLOv8?** Benchmarked on the supplied clip:

| | Hybrid (ours) | YOLOv8-n (off-shelf) |
|---|---:|---:|
| Detection rate | 100.0% | 92.2% |
| Mean latency | 34 ms | 16.7 ms |

YOLOv8-n misses ~68 frames. The 2× speedup does not justify a 7.8% miss rate
when the 250 ms CPU budget is already met by a wide margin.

**Ablation results** (two runs per channel: disable-one and solo):

| Channel | Disable-one rate | Drop | Solo rate | Solo notes |
|---|---:|---:|---:|---|
| Blue HSV | 100.0% | 0.0% | 100.0% | Wins NMS on every frame in the full run |
| Dark rect | 100.0% | 0.0% | 100.0% | Independently finds the bin every frame |
| Edge shape | 100.0% | 0.0% | 100.0% | Independently finds the bin every frame |
| Motion foreground | 100.0% | 0.0% | 96.8% | Misses first ~9 frames (MOG2 needs warm-up) |

`solo_first_5_frames` from `results/detector_ablation.json` confirms each channel
fires on real frames, not hypothetical coverage.

**Why the delivered run shows `blue_hsv_shape` every frame:** In the full pipeline,
blue HSV produces the highest-confidence candidate on this scene and wins NMS.
The other channels also fire and find the bin, but their proposals are suppressed
by NMS. Solo ablation verifies they are genuine fallbacks — not dead code.

**Threshold plateau** (no detection loss across this range):

| Parameter | Default | Plateau |
|---|---|---|
| `min_area_px` | 550 px | 300–1200 px |
| `aspect_min` | 0.45 | 0.25–0.80 |
| `aspect_max` | 3.25 | 2.0–4.5 |

**Limitation:** The detector is tuned to this scene. A different bin colour
or outdoor lighting requires retuning — all thresholds are exposed as
`__init__` kwargs in `detector.py`.

---

## Localization

### Coordinate frames

```
OpenCV camera frame          World frame (origin: pole base)
  +X -> right                  +X -> away from pole (forward)
  +Y -> down                   +Y -> left
  +Z -> forward (optical)      +Z -> up
```

### Camera-to-world transform derivation

Camera at height `h = camera_height_m`, pitched down `|tilt|` degrees
(calib.json stores downward as a negative number).

**Step 1 — Axis swap (level camera):**

When the camera is level, its optical axis (+Z_cam) points along +X_world:

```
R_axis_swap = [[0,  0,  1],
               [-1, 0,  0],
               [0, -1,  0]]
```

**Step 2 — Pitch about world Y:**

Let `theta = -tilt_rad` (positive for downward tilt):

```
R_pitch = [[cos(theta),  0,  sin(theta)],
           [0,           1,  0         ],
           [-sin(theta), 0,  cos(theta)]]
```

**Step 3 — Compose and translate:**

```
R_cw = R_pitch @ R_axis_swap
t_cw = [0, 0, h]

P_world = R_cw @ P_cam + t_cw
```

A runtime sign check (`optical_world[0] > 0, optical_world[2] < 0`) guards
against sign errors. See `localizer.py:build_camera_to_world`.

### Ground-contact model (primary estimator)

The bottom-centre pixel of the bounding box is unprojected onto the ground
plane (Z_world = 0), then the centroid is lifted by half the bin height:

```
p         = undistort_normalize([u_bottom, v_bottom])
r_cam     = normalize([p_x, p_y, 1])
r_world   = R_cw @ r_cam
lambda    = -h / r_world_z
P_ground  = t_cw + lambda * r_world
P_centroid = P_ground + [0, 0, H/2]    # H = 0.65 m
```

**Note on z_world:** z_world is construction-imposed as `H/2 = 0.325 m`
(ground contact + half bin height). It is not freely estimated. Full vertical
localization would require stereo or a depth sensor.

### Height-based depth estimate (diagnostic fallback)

```
Z = fy * H / h_px
X = (u_c - cx) / fx * Z
Y = (v_c - cy) / fy * Z
```

Both estimates are logged to `results/diagnostics.csv` per frame.

---

## Tracking and Kalman filter

### BBox tracker

Single-target `BBoxKalmanTracker` with multi-cue association:

- IoU + centre distance + area ratio + confidence + HSV histogram
- States: `STATIONARY`, `CONFIRMED`, `OCCLUDED`
- Max age before drop: 35 frames

### Position Kalman smoother

**State vector:**

```
x = [x, y, z, vx, vy, vz]    (metres, metres/second)
```

**Transition matrix** (dt = frame interval, 1/fps):

```
F = I + dt * coupling block [[0_3, I_3], [0_3, 0_3]]
```

**Process noise Q** (white-noise-acceleration discretization):

```
Q_pos   = dt^4 / 4 * q
Q_cross = dt^3 / 2 * q
Q_vel   = dt^2 * q
```

Measurement model H selects position only (3 of 6 states).
Covariance update uses the Joseph form to maintain positive semi-definiteness.
Measurement variance R is adapted per frame — tighter for high-confidence
blue-HSV detections, looser for edge-fallback detections.

**Runtime config** (`configs/default.yaml`):

```yaml
kalman:
  process_var: 3.0
  measurement_var: 0.01
```

**On hyperparameter validation:** A grid search was originally run using
`results/output.csv` as ground truth — circular evaluation. The filter was
scored against the same detector output it smooths. All configs produced
~0.001 m RMSE because z_world is construction-imposed constant and x/y
barely vary. Those results have been invalidated. The grid search script
(`experiments/kalman_gridsearch.py`) has been rewritten to use jitter on
stationary windows — an independent signal. Re-run it to tune empirically:

```
python experiments/kalman_gridsearch.py --video input.mp4 --calib calib.json
```

**Measured smoothing** (raw vs filtered, no external GT needed):

| Metric | Value |
|---|---|
| Frame-step std reduction | 59.9% |
| Second-difference std reduction | 84.4% |

---

## Occlusion handling

When the detector misses a frame:

1. **BBox Kalman prediction** — extrapolates position from last known velocity
2. **LK optical flow** — propagates the bounding box via tracked feature points
3. **OCCLUDED stdout line** — explicit flag with `occlusion_age` counter

**What the delivered run shows:** 0 occluded frames. The bin is detected in
every frame via the blue HSV channel; the fallback chain is not triggered.
No real person-occlusion event occurs in the delivered sequence.

Occlusion handling is verified through **three synthetic stress scenarios**
(detection suppressed + frames blurred on a different video segment):

| Scenario | Dropout frames | Continuity | Mean IoU vs baseline | Min IoU |
|---|---:|---:|---:|---:|
| mid_stop_dropout (frames 185–210) | 26 | 100.0% | 0.766 | 0.600 |
| moving_crossing_dropout (frames 432–445) | 14 | 100.0% | 0.675 | 0.578 |
| late_low_confidence_dropout (frames 748–762) | 15 | 100.0% | 0.909 | 0.890 |
| **All three combined** | **55** | **100.0%** | — | **0.578** |

All 55 OCCLUDED frames appear with the `occlusion_age` counter in
`results/occlusion_stress_suite.json`. Recovery is immediate via the
LK flow bridge — zero missed tracker outputs across all dropout windows.

Real-world occlusion robustness is implemented and stress-tested on
independent video segments, but 0 real occluded frames appear in the
primary delivered run.

---

## Results and benchmarks

Measured on CPU (no GPU) with the supplied `calib.json`:

| Metric | Value |
|---|---|
| Frames processed | 875 / 875 |
| Detector hit rate | 100.0% |
| Tracker output rate | 100.0% |
| Occluded frames (real) | 0 |
| Flow-assisted frames | 0 |
| First stdout from Python | 622 ms |
| Mean frame time | 33.9 ms |
| p95 frame time | 35.4 ms |
| Kalman frame-step σ reduction | 59.9% |
| Waypoint RMSE XY | 4.41 m (see Waypoint evaluation) |

---

## Output format

```
results/
+-- output.csv                   <- per-frame pose + bbox + tracking state
+-- summary.json                 <- run stats, thresholds, Kalman config
+-- diagnostics.csv              <- ground-contact vs height-based depth delta
+-- observer_overlay.mp4         <- annotated video with motion states
+-- observer_events.json         <- segmented motion events
+-- trajectory.png               <- top-down XY path (raw + Kalman filtered)
+-- trajectory_raw_vs_filtered.png
+-- qa_report.json
+-- occlusion_stress_suite.json
+-- run_manifest.json
```

`output.csv` schema:

```
frame_id, timestamp_ms,
x1, y1, x2, y2,                          <- bbox pixels
status, track_state, occlusion_age,
detector_source,
x_cam, y_cam, z_cam,                      <- camera frame
x_world, y_world, z_world,               <- world frame (centroid)
conf,
x_world_raw, y_world_raw, z_world_raw,   <- pre-Kalman
x_world_filt, y_world_filt, z_world_filt <- post-Kalman
```

---

## Waypoint evaluation

`waypoints.json` provides pixel coordinates of the three floor tape markers.
The pipeline projects each pixel through the camera model onto the ground
plane to get estimated world-frame stop positions, then computes RMSE
between the bin trajectory near each stop and those estimated positions.

### Projected waypoint positions (from camera model)

| Marker | Pixel (u, v) | Projected world (x, y) m |
|---|---|---|
| A (green) | (960, 529) | (5.20, 0.00) |
| B (orange) | (1193, 436) | (7.13, −1.21) |
| C (red) | (817, 385) | (8.87, 0.92) |

These projections are within the scenario's stated 5–9 m range, which
validates the camera model geometry.

### Bin position vs. waypoints

| Stop | Bin centroid XY (m) | Waypoint XY (m) | Error XY (m) |
|---|---|---|---|
| A | (2.42, −0.04) | (5.20, 0.00) | 2.78 |
| B | (3.43, 1.00) | (7.13, −1.21) | 4.31 |
| C | (3.32, −0.15) | (8.87, 0.92) | 5.66 |
| **RMSE XY** | | | **4.41 m** |

### Why the gap is 4.41 m

The 4.41 m RMSE does **not** mean the localization formula is wrong. The
geometry check confirms this: the `height_scale_error_factor` for stop A is
2.37, meaning the bin's bounding box pixel height is 2.37× larger than it
would be if the bin were at the waypoint depth (5.2 m). Dividing:
5.2 m / 2.37 ≈ 2.2 m — matching the pipeline's estimate of 2.42 m.

**Conclusion:** The bin in the video is physically at ~2–4 m from the pole
during the delivered sequence, while the tape markers on the floor are at
5–9 m. The pipeline correctly localises the bin where it actually appears.
The gap is a content mismatch between the delivered clip and the assessment
scenario description, not a localization error.

### Waypoint pixel reliability

The QA pixel probe (`qa_report.json: waypoint_pixel_probes`) samples the
image color at each declared waypoint pixel:

| Marker | Expected colour | Observed at pixel |
|---|---|---|
| A | green | blue |
| B | orange | neutral/gray |
| C | red | green |

None of the pixel samples match the declared marker colours. This means the
provided pixel coordinates may not land precisely on the tape markers in the
actual frames, further reducing confidence in the waypoint RMSE as an
independent accuracy metric. The `qa_report.json` flags this explicitly as
`"waypoint_rmse_status": "large_residual"`.

---

## Jetson deployment notes

The CPU pipeline fits within the Orin NX thermal/power envelope.

**Model quantisation:** If switching to a learned backbone, use TensorRT
FP16 on Orin NX. For INT8 quantisation, collect representative calibration
frames from the deployment environment — do not quantise on a lab dataset.
RKNN targets RK-series SoCs; Orin NX is NVIDIA, so TensorRT is the
correct toolchain.

**Moving platform:** The current transform assumes a fixed pole:

```
P_world = R_cw @ P_cam + t_cw      t_cw = [0, 0, h] (constant)
```

On a moving UAV this becomes:

```
P_world = T_world_body(t) @ T_body_cam @ P_cam
```

where `T_world_body(t)` must come from the vehicle EKF (IMU + GNSS + VIO).
The body-to-camera extrinsic `T_body_cam` is fixed after mount calibration.

**Flight controller output:** Use `LANDING_TARGET` or
`VISION_POSITION_ESTIMATE` at 10–20 Hz. Both accept position covariance,
which this pipeline provides via `sigma_x / sigma_y / sigma_z` per frame.

**Latency budget on Orin NX:**

| Stage | Budget |
|---|---:|
| Capture (CSI / USB) | 5–12 ms |
| Detect (TensorRT FP16) | 8–25 ms |
| Localise | < 1 ms |
| Kalman update | < 1 ms |
| Serialise / transmit | 1–3 ms |
| **End-to-end** | **30–55 ms** |

---

## Known limitations

- **Scene-specific detector.** All four channels are calibrated for this
  clip. A different bin colour or lighting requires retuning. All thresholds
  are exposed as `__init__` kwargs in `detector.py`.
- **z_world is construction-imposed.** The pipeline does not estimate
  vertical position freely — it enforces `z_world = H/2 = 0.325 m` via
  the ground-plane contact assumption. Correct for a flat floor; fails on
  slopes.
- **Ground-contact model needs a clean bottom edge.** Partial bbox
  truncation or wheel occlusion degrades x/y accuracy.
- **RMSE vs waypoints is 4.41 m.** This reflects a content mismatch
  between the video and the scenario (bin at 2–4 m, tape at 5–9 m),
  not a geometry error. See Waypoint evaluation.
- **Kalman hyperparameters not independently validated.** The original
  grid search was circular; re-run `experiments/kalman_gridsearch.py`
  to tune empirically once video is available.
- **No real occlusion in delivered clip.** Occlusion handling is
  implemented and stress-tested synthetically, but 0 real occluded frames
  appear in the delivered run.
- **calib.json is not committed.** Provide it at runtime via `--calib`.
  All extrinsics (height, tilt, K, distortion) are read from that file.
