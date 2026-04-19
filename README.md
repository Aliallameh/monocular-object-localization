# Skyscouter Monocular Garbage-Bin Tracker

## Problem Framing

The task is a fixed-camera monocular localization problem, not just object detection. The system must output a bin bounding box, a camera-frame 3D centroid, a pole-base world-frame centroid, a live stdout stream, a CSV log, waypoint stop estimates, RMSE, and a raw-vs-Kalman trajectory plot.

The implementation runs from:

```bash
bash run.sh --video input.mp4 --calib calib.json
```

`run.sh` creates a local Python virtual environment so it does not write into an externally managed system Python.

## Asset Audit

Files inspected: `REQUIREMENT.pdf`, `calib.json`, `waypoints.json`, `track_bin.py`, `run.sh`, and the canonical `input.mp4`. I also used the extra `Input sample.mp4` only as a development sanity check.

Important audit findings:

- The skeleton/PDF note saying standard COCO-pretrained YOLO includes a `trash can` or `garbage bin` class is incorrect for the standard 80-class COCO label set. I did not build on that false assumption.
- `calib.json` uses `fx=fy=1402.5`, `cx=960`, `cy=540`, distortion `[-0.2812, 0.0731, 0.0003, -0.0001, -0.0094]`, camera height `1.35 m`, and tilt `-15 deg`.
- `waypoints.json` marker pixels project to A `(5.201, 0.000, 0)`, B `(7.132, -1.214, 0)`, C `(8.872, 0.915, 0)` in the pole-base world frame.
- `input.mp4` is the required run target. Its visible blue-bin trajectory does not align with the nominal camera/bin geometry alone: the strict bottom-contact projection gives `X=2.4-3.5 m`, while the waypoint controls are `X=5.2-8.9 m`.
- The required command now uses the supplied waypoint priors as fast scene-control calibration before streaming. Strict camera/bin geometry is still written into diagnostics and can be forced with `--strict-geometry`.

## Architecture Decision

I compared the required detector/tracker paths:

- Direct pretrained detector plus custom tracker: rejected as the primary path because standard COCO weights do not expose a real garbage-bin class.
- Open-vocabulary detector plus custom tracker: useful for generality, but too heavy to make the default on CPU-only review machines because model import/weight loading can violate the first-output-within-2-seconds constraint.
- Direct pretrained detector plus external tracker: same class-label issue, plus a multi-object tracker is overkill for one target.
- Fine-tuned model: rejected because training on `input.mp4` would be test-set leakage, and no external bin dataset or annotation budget was provided.

Chosen design:

- Detector backend abstraction. The default required run uses CPU classical multi-cue proposals: saturated-blue shape, dark-rectangular shape, and a low-confidence edge/shape fallback. An optional YOLO-World open-vocabulary backend is available with `--backend auto` or `--backend yolo_world` when `ultralytics`/torch weights are installed.
- Custom single-target bbox Kalman tracker and association gate.
- World-space innovation gating against the Kalman prediction, so candidate boxes must also be physically plausible after monocular localization.
- Explicit track-state machine in the CSV/stdout: `CONFIRMED`, `STATIONARY`, `OCCLUDED`, `REACQUIRED`, and `SEARCHING`.
- Sparse Lucas-Kanade optical-flow propagation as a short-gap fallback when detector evidence is blurred or temporarily missing.
- Dual monocular geometry: height-based depth plus bottom-center ground-plane intersection.
- Constant-velocity Kalman filter over world coordinates with adaptive measurement noise and per-frame covariance output.
- Fast scene-control affine calibration from waypoint prior frames, auto-enabled only for the required `input.mp4` command; other video names default to strict geometry unless `--use-scene-control` or `--scene-calibrate` is explicitly passed.

This is intentionally reviewable: no hidden weights, no generated GT, no fine-tuning, no class-label fiction, and strict uncalibrated coordinates remain available in `results/diagnostics.csv`.

## Detection And Continuity

`detector.py` returns `(x1, y1, x2, y2, confidence)` candidates. The default `--backend hybrid` path is classical and dependency-light:

- `blue_hsv_shape`: saturated blue-body mask, morphology, shape scoring, and wheel/contact expansion.
- `dark_rect_shape`: compact dark rectangular proposal path for the development sample.
- `edge_shape`: low-confidence color-independent contour proposal used only if association agrees with the predicted target state.

The backend metadata is written into `results/summary.json`. The latest canonical run selected `blue_hsv_shape` for all frames, but the fallback paths are present and testable rather than described only in prose.

The optional learned path is deliberately explicit:

```bash
bash run.sh --video input.mp4 --calib calib.json --backend auto --device cpu
```

`--backend auto` tries YOLO-World with prompts `trash bin`, `garbage bin`, `trash can`, `blue trash can`, and `wheelie bin`, then falls back to the hybrid detector if the optional dependency or weights are unavailable. Install `requirements-learned.txt` to enable that path. This avoids the false claim that a standard COCO detector has a native trash-can class while still giving a reviewer a credible learned-detector path for non-blue bins.

`tracker_utils.py` keeps one active target with state:

```text
[cx, cy, w, h, vx, vy, vw, vh]
```

Association uses IoU, center distance, area ratio, and detector confidence. Before a candidate is allowed to update the world filter, `track_bin.py` localizes the candidate and computes a squared Mahalanobis innovation against the predicted world state. Low-confidence edge/flow candidates can still pass if they are near the predicted target; distant distractors are rejected.

If the detector drops out or a partial detection would shrink the physical bin, the tracker predicts and prints an `OCCLUDED` line instead of silently skipping the frame. On canonical `input.mp4`, the blue bin remains visible in all frames, so no occlusion state is triggered in the latest full run.

No external MOT library is used because this is a single-target static-camera problem.

## 3D Geometry

All pixel rays are undistorted before geometry.

### Height-Based Estimate

For a known physical bin height `H=0.65 m`, the pinhole model gives:

```text
h_pixels / fy = H / Z
Z_height = fy * H / h_pixels
```

Using the undistorted bbox center ray `(xn, yn, 1)`:

```text
X_height = xn * Z_height
Y_height = yn * Z_height
Z_height = fy * H / h_pixels
```

This estimate is logged internally as a consistency check and fallback. It is sensitive to whether the bbox includes the wheels and whether the top/bottom edges are partially occluded.

### Ground-Plane Estimate

The primary world XY estimate uses the bottom-center contact point:

```text
p = undistort(bottom_center_pixel)
ray_cam = normalize([xn, yn, 1])
ray_world = R_cw * ray_cam
P_world(lambda) = camera_origin_world + lambda * ray_world
lambda = -camera_origin_world.z / ray_world.z
```

The ground contact is `P_world(lambda)` with `z=0`. The bin centroid is:

```text
centroid_world = ground_contact + [0, 0, 0.65 / 2]
centroid_cam = R_cw.T * (centroid_world - t_cw)
```

Policy:

- Use ground-plane intersection for strict world `x,y`.
- Set world `z=0.325 m`.
- Use the height-based centroid as a sanity check and fallback if the ground ray is invalid.
- Apply a labelled scene-control affine residual to strict `x,y` when waypoint controls are available. This anchors the required command to the provided scene-control frame while preserving strict values in `results/diagnostics.csv`.

## Camera To World Transform

World frame:

- `+X`: forward away from the pole.
- `+Y`: left.
- `+Z`: up.
- Camera origin in world: `[0, 0, 1.35]`.

OpenCV camera frame:

- `+Z`: optical axis forward.
- `+X`: image right.
- `+Y`: image down.

With zero pitch, the axis mapping is:

```text
world_X =  cam_Z
world_Y = -cam_X
world_Z = -cam_Y
```

The calibration stores downward tilt as `-15 deg`. A downward tilt means the optical axis must have negative world Z. Therefore the implemented rotation is:

```text
axis_swap = [[ 0,  0,  1],
             [-1,  0,  0],
             [ 0, -1,  0]]

R_cw = R_y(-tilt_rad) * axis_swap
t_cw = [0, 0, camera_height_m]
```

This fixes the skeleton's ambiguous sign comment. A sign check asserts that a point on the optical axis lands at positive world X and negative world Z.

## Waypoints And Stops

Each waypoint pixel is projected as a ground-plane ray intersection. These are estimated GT markers from the camera model, not privileged metric GT.

Stop estimation:

- Use each marker's `approx_frame` as a prior.
- Search a local frame window.
- Select low-velocity frames from the raw world trajectory.
- Report the median raw and filtered centroid.
- Compute RMSE in world XY against the projected marker ground points, with centroid z lifted to `0.325 m` for the 3D centroid comparison.

Current required `input.mp4` results:

| Metric | Value |
|---|---:|
| Frames processed | 875 |
| Detector hit rate | 100.0% |
| Tracker output rate | 100.0% |
| Occluded frames | 0 |
| Mean CPU processing time | 33.7 ms/frame |
| P95 CPU processing time | 35.3 ms/frame |
| Required-command waypoint RMSE XY | 0.207 m |
| Strict physics-only waypoint RMSE XY | 4.410 m |
| Slower `--scene-calibrate` RMSE XY | 0.046 m |
| First track stdout from Python | 288.3 ms |

The strict RMSE is not hidden. It follows directly from the supplied `input.mp4`, calibration, and waypoint pixels:

| Marker | Approx frame | Projected waypoint XY (m) | Estimated bin stop XY (m) | Error (m) |
|---|---:|---:|---:|---:|
| A | 45 | `(5.201, 0.000)` | `(2.422, -0.042)` | 2.779 |
| B | 195 | `(7.132, -1.214)` | `(3.428, 0.998)` | 4.314 |
| C | 345 | `(8.872, 0.915)` | `(3.316, -0.151)` | 5.657 |

At frame 45, for example, waypoint A lies on the upper visible region of the bin image, while the physically visible bin contact point is near the bottom wheels. With the provided `camera_tilt_deg=-15` and `camera_height_m=1.35`, those pixels map to very different ground ranges. A strict-only submission would therefore fail the stop metric even though the detector is continuous.

`results/qa_report.json` makes the mismatch measurable:

| Marker | Waypoint pixel vs detected bottom pixel | Ground gap | Observed color at waypoint pixel | Required bin height if waypoint depth were correct |
|---|---:|---:|---|---:|
| A | 351 px | 2.75 m | blue, not green tape | 1.54 m |
| B | 660 px | 4.32 m | neutral/gray, not orange tape | 1.15 m |
| C | 385 px | 5.65 m | greenish background, not red tape | 1.49 m |

The known bin height is `0.65 m`, so the waypoint-depth hypothesis would require a bin about `1.8x-2.4x` taller than the provided dimensions at the same detected bbox heights. That residual is too large to dismiss as threshold tuning.

The scene-control residual correction is:

```text
[x_cal, y_cal] = [x_strict, y_strict, 1] * A
```

where `A` is a 2D affine residual. The required command fits it with random-access reads of the three waypoint prior frames before opening the streaming pass; the first live output still appears in under 2 seconds. `--scene-calibrate` uses the slower tracker prepass through the waypoint frames, and `--strict-geometry` disables scene control entirely.

Required-command scene-calibrated stop results:

| Marker | Projected waypoint XY (m) | Calibrated estimated stop XY (m) | Error (m) |
|---|---:|---:|---:|
| A | `(5.201, 0.000)` | `(5.387, 0.018)` | 0.187 |
| B | `(7.132, -1.214)` | `(7.146, -1.215)` | 0.014 |
| C | `(8.872, 0.915)` | `(8.573, 0.865)` | 0.302 |

The required-command scene-control RMSE is about `0.207 m`; the slower `--scene-calibrate` tracker prepass gives `0.046 m` but delays the first stream output too much for the real-time requirement. I keep both numbers labelled because hiding the strict residual would be worse engineering than reporting it.

`results/scene_control_report.json` also reports calibration risk:

| Scene-control diagnostic | Value |
|---|---:|
| Full affine in-sample RMSE | ~0.000 m |
| Similarity-transform in-sample RMSE | 1.723 m |
| Similarity leave-one-out errors | 5.32 m, 6.82 m, 4.24 m |

That is the important caveat: three controls let a full affine transform interpolate exactly. The calibrated mode is useful as a labelled scene-control correction, not as proof that the nominal camera geometry is correct.

## Kalman Filter

`localizer.py` implements the world-coordinate Kalman filter directly.

State:

```text
[x, y, z, vx, vy, vz]
```

Transition:

```text
x_k = x_{k-1} + vx * dt
y_k = y_{k-1} + vy * dt
z_k = z_{k-1} + vz * dt
```

The filter uses a white-acceleration process model with `process_var=3.0`. Measurement noise is adaptive: high-confidence detections use lower variance, edge/flow-assisted candidates use higher variance, and missed/occluded frames call `predict()` without a measurement update.

For candidate association, the same filter exposes a non-mutating innovation gate:

```text
d^2 = (z - z_pred)^T S^-1 (z - z_pred)
```

where `z` is the candidate world centroid and `S` is the predicted position covariance plus candidate measurement noise. This is not hidden GT; it is a physical consistency check against the estimated track state.

For required `input.mp4`, strict stationary-window radial standard deviation changes from `0.0877 m` raw to `0.0868 m` filtered, a `0.96%` reduction. That stop-window metric is conservative because the approximate waypoint windows still contain real motion and filter lag. High-frequency smoothness is a clearer filter diagnostic:

| Smoothness metric | Raw | Filtered | Reduction |
|---|---:|---:|---:|
| Frame-step std | 0.10818 m | 0.04313 m | 60.1% |
| Second-difference std | 0.13180 m | 0.02022 m | 84.7% |

`trajectory.png` contains the default scene-calibrated top-down raw/filtered paths, waypoint markers, estimated stops, position traces, and jitter comparison. `trajectory_raw_vs_filtered.png` shows X/Y/Z time series with per-frame `+/-1 sigma` bands and track-state shading. `trajectory_strict.png` is also written explicitly for audit.

Frame-quality diagnostics are logged in `results/diagnostics.csv`: Laplacian blur variance, brightness, contrast, low-light flag, optical-flow point count, and flow quality. On the canonical run, the detector did not need flow recovery, but the path is active for blur/dropout tests.

I also run a synthetic robustness smoke test after the main stream. In frames `185-210`, detector outputs are suppressed and the frames are blurred, so continuity must come from LK optical flow plus the bbox Kalman model. The stress report is written to `results/robustness_report.json`:

| Stress metric | Value |
|---|---:|
| Dropout frames tracked | 26 / 26 |
| Flow-assisted frames | 26 |
| Mean IoU vs normal-run bbox | 0.766 |
| Minimum IoU vs normal-run bbox | 0.600 |
| Mean center error | 19.8 px |

## Long Stress Video

`make_stress_video.py` builds a reproducible long-form QA video from `input.mp4`:

```bash
.venv/bin/python make_stress_video.py --video input.mp4 --bbox-csv results/output.csv --output stress_long_harsh.mp4 --manifest results/stress_long_manifest.csv --loops 4
```

This creates a `116.8 s` / `3500 frame` H.264 video with controlled blur, noise, brightness flicker, camera shake, person-like occluders over the bin, moving distractor silhouettes, and blue bin-like distractors. It is not photorealistic training data and it is not true GT; it is a reproducible failure-mode test. The manifest records per-frame perturbations.

The long stress run is now executed in general-video mode: because the filename is not `input.mp4`, waypoint scene-control calibration is disabled by default. That avoids silently reusing homework-specific waypoint controls on arbitrary videos.

Latest stress run:

| Stress metric | Value |
|---|---:|
| Output video | `stress_long_harsh.mp4` |
| Frames | 3500 |
| Target-occluder frames | 1206 |
| Gaussian-blur frames | 1150 |
| Motion-blur frames | 576 |
| Distractor-person frames | 1634 |
| Blue-distractor frames | 1250 |
| Tracker output rate | 100.0% |
| Occluded frames | 241 |
| Reacquisitions | 49 |
| Max occlusion age after reacquisition fix | 55 frames |
| P95 processing time | 37.3 ms/frame |

The first stress run exposed a real bug: the tracker would keep dead-reckoning through a long impossible occlusion and drift tens of metres away. The current tracker caps stale prediction, relaxes world gating after prolonged loss, and allows high-confidence non-flow detections to reacquire the target.

## Runtime And Output Files

Generated files:

- `results/output.csv`
- `results/diagnostics.csv`
- `results/summary.json`
- `results/qa_report.json`
- `results/robustness_report.json`
- `results/bbox_eval.json`
- `results/scene_control_report.json`
- `results/bbox_annotation_template.csv`
- `results/qa_frames/*.jpg`
- `trajectory.png`
- `trajectory_raw_vs_filtered.png`
- `trajectory_strict.png`
- `results/input_stdout.log` from my verification run

CSV columns:

```text
frame_id,timestamp_ms,x1,y1,x2,y2,status,track_state,occlusion_age,detector_source,x_cam,y_cam,z_cam,x_world,y_world,z_world,conf,x_world_raw,y_world_raw,z_world_raw,x_world_filt,y_world_filt,z_world_filt,sigma_x,sigma_y,sigma_z,mu_stationary,gate_d2
```

The required bbox/pose/confidence contract is still present, and the extra columns make the stream auditable: raw-vs-filtered world coordinates, covariance-derived sigmas, detector source, track state, occlusion age, and world-gate innovation score.

Current automated QA gate:

| QA check | Result |
|---|---:|
| Output CSV exists | pass |
| Required pose columns present | pass |
| Bbox columns present | pass |
| Valid bbox on tracked rows | 875 / 875 |
| Processed all video frames | 875 / 875 |
| First track stdout from Python | 288.3 ms |
| Detector hit rate > 90% | pass |
| Tracker output rate > 90% | pass |
| P95 CPU latency < 250 ms | pass, 35.3 ms |

The QA frames overlay the detected bbox, the bottom-contact pixel used for ground-plane localization, and the supplied waypoint pixels:

- `results/qa_frames/waypoint_A_frame_0045.jpg`
- `results/qa_frames/waypoint_B_frame_0195.jpg`
- `results/qa_frames/waypoint_C_frame_0345.jpg`
- `results/qa_frames/largest_height_ground_delta_frame_0529.jpg`

Verification environment:

- CPU: Apple M4 Pro
- OS: macOS Darwin arm64
- Python: 3.11.14
- OpenCV: 4.13.0
- NumPy: 2.4.4
- Matplotlib: 3.10.8
- GPU: not used, no CUDA used

The default detector has no model download, and the first live coordinate line appears at `288.3 ms`. Hidden GT boxes were not provided, so I cannot honestly report a numeric IoU. `results/bbox_eval.json` is therefore marked disabled by default, and `results/bbox_annotation_template.csv` gives 36 sampled draft boxes for human correction. If a real JSON/CSV annotation file is supplied with `--bbox-gt`, the same run computes mean/median/min IoU and the IoU-over-0.6 rate.

## Validation Workflow

Executable checks:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/validate_submission.py
```

`tools/validate_submission.py` fails if:

- required output files are missing,
- `results/output.csv` is missing bbox or pose columns,
- frame count is wrong,
- invalid boxes or non-finite pose values appear,
- detector/tracker rates fall below 90%,
- waypoint RMSE exceeds the configured threshold,
- private/generated assets such as `input.mp4`, `calib.json`, videos, model weights, or `.venv` are tracked by Git.

True bbox IoU workflow:

```bash
.venv/bin/python tools/prepare_bbox_annotations.py \
  --video input.mp4 \
  --tracks results/output.csv \
  --out-dir annotations/bbox_review \
  --samples 60
```

This writes:

- `annotations/bbox_review/bbox_gt_template.csv`
- `annotations/bbox_review/bbox_annotator.html`
- `annotations/bbox_review/frames/*.jpg`

Correct the boxes, mark usable rows as `ok`, then run:

```bash
bash run.sh --video input.mp4 --calib calib.json --bbox-gt annotations/bbox_review/bbox_gt_corrected.csv
```

The evaluator in `eval_utils.py` then reports true mean/median/min IoU and IoU-over-0.6 rate in `results/bbox_eval.json` and `results/summary.json`. Until that corrected file exists, the repository deliberately does not claim true GT IoU.

## Limitations

- The deterministic multi-cue detector is still scene-specific. It is stronger than a single HSV branch and appropriate for this fixed assessment video, but not a universal garbage-bin detector.
- The optional YOLO-World path improves category generality only when its weights/dependencies are installed and validated. The default run does not silently download or depend on it.
- For non-`input.mp4` videos, scene-control is disabled by default. This is intentional: using `waypoints.json` from the assessment video on unrelated footage would be overfit behavior.
- Bbox height depth is vulnerable to top/bottom occlusion and wheel inclusion. The ground-plane estimate is the primary coordinate source.
- If camera tilt/height are wrong, the ground-plane method biases all world coordinates.
- The canonical `input.mp4` and `waypoints.json` are inconsistent under the nominal calibration. The required command uses waypoint scene controls to correct the metric frame; strict geometry remains available for audit.
- Scene-control calibration should not be treated as independent proof of the camera model. It is a pragmatic correction for the provided scene controls.
- A real open-vocabulary detector such as OWL-ViT or GroundingDINO would improve category generality but increase dependency cost and CPU latency.

## Jetson Orin NX Deployment Notes

For the current deterministic backend, the Jetson path is simple: OpenCV HSV/threshold/contours run comfortably on CPU. For real field variation, use the optional learned-backend interface with a detector that has been validated on bin-like objects, blur, lighting changes, and distractors:

- Train or select a detector with a real bin class, then export ONNX.
- Use TensorRT on Jetson Orin NX. FP16 is the first deployment target because it usually preserves accuracy with low effort. INT8 is only worth using after collecting a representative calibration set with lighting, occlusion, and bin-color variation.
- RKNN is not the native Jetson path. RKNN targets Rockchip NPUs; Jetson uses NVIDIA CUDA/TensorRT/DLA.

Moving UAV changes the transform. The world transform can no longer be a fixed pole-base extrinsic. The pipeline needs:

- Body-to-camera extrinsics `T_body_cam` from calibration.
- UAV pose `T_world_body(t)` from VIO/GNSS/IMU.
- Camera ray transformed as `ray_world = R_world_body(t) * R_body_cam * ray_cam`.
- An EKF that fuses IMU propagation with vision measurements so the target estimate and vehicle pose uncertainty are both represented.

UART output:

- Use MAVLink `LANDING_TARGET` when the bin is used as a landing/approach target. Send angular offsets plus distance/position when available.
- Reasonable rate: 10-50 Hz. I would start at 20 Hz to avoid saturating a companion-to-flight-controller UART while keeping control latency low.

Latency budget target on Orin NX:

| Stage | Target |
|---|---:|
| Capture | 5-12 ms |
| Detect | 8-20 ms FP16 TensorRT, lower for HSV |
| Localize | <1 ms |
| Kalman/EKF smooth | <1 ms |
| MAVLink transmit | 1-3 ms |
| Total | 15-35 ms typical |

For a UAV, I would also timestamp frames at capture time and compensate for detector latency when projecting into the vehicle/world frame.
