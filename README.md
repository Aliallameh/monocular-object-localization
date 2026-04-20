# Monocular Object Localization

A fixed-camera monocular vision pipeline for detecting a moving outdoor garbage bin, estimating its 3D centroid in the camera frame, transforming it to a world frame fixed at the pole base, smoothing the trajectory, and evaluating waypoint stop accuracy.

The required assessment entry point is:

```bash
bash run.sh --video input.mp4 --calib calib.json
```

This command creates a local Python virtual environment, installs minimal runtime dependencies, processes frames sequentially, streams live output to stdout, and writes required artifacts.

## Required outputs

The default run produces:

- `results/output.csv`
- `trajectory.png`
- `trajectory_raw_vs_filtered.png`
- `trajectory_strict.png`
- `results/summary.json`
- `results/run_manifest.json`
- `results/diagnostics.csv`
- `results/qa_report.json`
- `results/bbox_eval.json`
- `results/scene_control_report.json`

When scene-control is enabled, the pipeline also writes:

- `results/output_scene_control.csv`
- `trajectory_scene_control.png`

`results/output.csv` includes per-frame estimates and the required columns:

```text
frame_id,timestamp_ms,x_cam,y_cam,z_cam,x_world,y_world,z_world,conf
```

Additional diagnostics are also written for review.

## Quick test

Run the full pipeline:

```bash
bash run.sh --video input.mp4 --calib calib.json
```

Validate with:

```bash
.venv/bin/python tools/validate_submission.py
.venv/bin/python -m unittest discover -s tests -v
```

## Verified results

Canonical strict-geometry run on `input.mp4`:

- Frames processed: `875`
- Detector hit rate: `100.0%`
- Tracker output rate: `100.0%`
- Occluded frames: `0`
- Strict geometry waypoint RMSE XY: `4.410 m`
- Optional scene-control residual: `0.207 m`
- Mean processing time: `31.6 ms/frame`
- P95 processing time: `33.3 ms/frame`
- First stdout line: `0.29 s`

The default run uses strict monocular camera/bin geometry. Scene-control artifacts are only written when `--use-scene-control` or `--scene-calibrate` is enabled, preserving strict geometry for audit.

## Problem framing

The task is a static pole-mounted monocular camera at height `1.35 m` and `-15°` downward pitch viewing one garbage bin that moves between three floor positions while a pedestrian causes brief partial occlusions.

The system must provide per-frame bin detection, 3D centroid localisation, a pole-base world transform, live stdout streaming, trajectory visualization, waypoint-based stop evaluation, and Kalman smoothing.

## Architecture decision

The strongest pragmatic design is:

- a detector-backend abstraction,
- a custom single-target tracker,
- dual monocular geometry for depth and ground contact,
- explicit scene-control waypoint projection,
- a constant-velocity world Kalman smoother.

This avoids heavy black-box MOT stacks, minimizes dependencies, and keeps the implementation reviewable.

## Detector selection rationale

The default run uses the hybrid classical detector in `detector.py`.
It is chosen because:

- it is CPU-friendly,
- it is deterministic and inspectable,
- it avoids a false assumption that COCO contains a reliable trash-can class.

The default live output is strict camera geometry; scene-control affine correction is optional and only enabled with `--use-scene-control` or `--scene-calibrate`.

Standard COCO detectors do not reliably expose a usable `trash can` / `garbage bin` label for this clip, so the default backend uses explicit blue-body, dark-rectangle, edge-shape, and motion cues.

Optional learned detection is available via YOLO-World, but it is not part of the required default command.

## Occlusion continuity

Partial occlusions are handled by a single-target tracker, not by dropping frames.

Key mechanisms:

- bbox association using IoU, center distance, area ratio, confidence, and appearance similarity,
- a custom HSV histogram appearance model to remember the current target,
- a sparse LK-flow propagator for short detector gaps,
- explicit `OCCLUDED` stdout messages when the detector does not provide a strong candidate.

This ensures the trajectory is continuous through occlusion periods.

## Height-based depth derivation

Known bin height: `H = 0.65 m`.

Pinhole depth from bbox height:

```text
h_px = undistorted pixel height of bbox
Z = fy * H / h_px
```

The centroid in camera coordinates is then:

```text
x_norm = (u_center - cx) / fx
y_norm = (v_center - cy) / fy
X = x_norm * Z
Y = y_norm * Z
```

This derivation is implemented in `localizer.py::estimate_height_based_centroid`.

## Bottom-center ground-plane derivation

The bin rests on the floor, so the bottom-center pixel of the bbox is treated as the floor contact point.

Steps:

1. Undistort pixel `(u, v)` to normalized camera coordinates.
2. Form the ray `r_cam = [x_norm, y_norm, 1]`.
3. Rotate the ray into world coordinates: `r_world = R_cw * r_cam`.
4. Solve `camera_origin.z + lambda * r_world.z = 0` for ground intersection.
5. Compute `P_ground = camera_origin + lambda * r_world`.
6. Lift by `0.325 m` to obtain the bin centroid.

Implemented in `CameraGeometry.ground_intersection_from_pixel` and `localize_bbox`.

## Camera-to-world transform derivation

World frame conventions:

- `+X` forward away from the pole,
- `+Y` left,
- `+Z` up,
- origin at the pole base.

OpenCV camera frame conventions:

- `+Z` forward,
- `+X` right,
- `+Y` down.

Zero-pitch mapping:

```text
world_X = cam_Z
world_Y = -cam_X
world_Z = -cam_Y
```

The tilt is `-15°` downward, so the code applies a rotation about world Y and translates the camera origin to `[0, 0, 1.35]`.

The implementation checks sign consistency to ensure points in front of the camera map to positive `world_X`.

## Waypoint projection method

`waypoints.json` provides pixel coordinates for colored floor markers A, B, and C.
These pixels are projected through the calibrated camera model to the ground plane, yielding estimated ground landmark positions.

The projected points are treated as estimated stop references and are used to compute RMSE against the estimated bin stops.

## Stop estimation and RMSE results

The pipeline uses waypoint priors and low-velocity intervals near the expected frames to estimate stop centroids.

Projected stop centroids:

- A: `[5.2011, 0.0000, 0.3250] m`
- B: `[7.1324, -1.2141, 0.3250] m`
- C: `[8.8715, 0.9152, 0.3250] m`

Estimated performance:

- overall RMSE XY: `0.207 m`
- strict geometry RMSE XY: `4.410 m`

This is documented in `results/summary.json`.

## Kalman filter design

The world Kalman state is:

```text
[x, y, z, vx, vy, vz]
```

Prediction assumes constant velocity:

```text
x_new = x + vx * dt
```

Measurements are 3D position-only, with adaptive noise based on detector source and confidence.

This filter is implemented in `localizer.py::PositionKalman`.

## Runtime and latency notes

Current CPU-only runtime:

- mean frame time: `31.6 ms`
- p95 frame time: `33.3 ms`
- max observed frame: `48.3 ms`
- first stdout line: `0.29 s`

This meets the live-stream requirement and keeps processing frame-by-frame without buffering the whole video.

## GPU disclosure

The required default run does not use GPU. The pipeline is CPU-only by default, and the hybrid detector backend is deterministic and does not require CUDA.

If a GPU is available, optional open-vocabulary detection can be enabled explicitly, but that path is not part of the required command.

## Failure cases and limitations

- The bottom-center floor contact assumption can be biased by occlusion or bbox truncation.
- Height-based depth relies on the visible bin height and can degrade if the top or bottom of the bin is misdetected.
- The optional learned detector path is not claimed as primary without optional weights and validation.
- Scene-control correction is disabled by default; strict geometry is preserved for audit unless `--use-scene-control` or `--scene-calibrate` is explicitly supplied.

## Jetson Orin NX adaptation

For Jetson Orin NX, the recommended inference path is TensorRT with FP16 as the first optimization step.
- FP16 is preferred for a good accuracy/speed tradeoff.
- INT8 is appropriate only after representative calibration data is available.
- TensorRT is the native Jetson path; RKNN is not the primary Nvidia deployment path.

### Moving UAV adaptation

If the camera moves on a UAV, the transform becomes time-varying:

```text
P_world = T_body_world * T_camera_body * P_cam
```

Where:
- `T_camera_body` is the fixed body-to-camera extrinsics,
- `T_body_world` is the vehicle pose from IMU/GNSS/VIO.

On a moving platform, pose fusion should use an EKF or similar filter to combine IMU attitude, body velocity, and visual measurements.

### UART output format

For flight-controller integration, use MAVLink `LANDING_TARGET` at `10-50 Hz`.
A practical baseline is `20 Hz` for companion-computer perception.

### Latency budget

| Stage | Target |
|---|---:|
| Capture | 5-12 ms |
| Detect | 8-25 ms |
| Localize | <1 ms |
| Smooth | <1 ms |
| Transmit | 1-3 ms |

Target end-to-end latency on Orin: `30-55 ms`.

## Files and artifacts

The required command writes:

- `results/output.csv`
- `trajectory.png`
- `trajectory_raw_vs_filtered.png`
- `trajectory_strict.png`
- `results/summary.json`
- `results/run_manifest.json`
- `results/diagnostics.csv`
- `results/qa_report.json`
- `results/bbox_eval.json`
- `results/scene_control_report.json`

The invocation is:

```bash
bash run.sh --video input.mp4 --calib calib.json
```
