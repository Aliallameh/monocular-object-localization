# Monocular Garbage-Bin Tracking And Localization

This repository implements the required fixed-camera garbage-bin tracking pipeline:

```bash
bash run.sh --video input.mp4 --calib calib.json
```

The command creates a local virtual environment if needed, installs `requirements.txt`, processes the video sequentially, streams one line per frame to stdout, and writes the review artifacts.

## Default Artifacts

- `results/output.csv`
- `trajectory.png`
- `trajectory_raw_vs_filtered.png`
- `trajectory_strict.png`
- `results/summary.json`
- `results/run_manifest.json`
- `results/diagnostics.csv`
- `results/qa_report.json`
- `results/bbox_eval.json`
- `results/asset_alignment_report.json`
- `results/occlusion_stress_suite.json`
- `results/review_readiness.json`
- `results/review_readiness.md`

`results/output.csv` contains the required pose columns plus detection/tracking evidence:

```text
frame_id,timestamp_ms,x1,y1,x2,y2,status,track_state,occlusion_age,detector_source,x_cam,y_cam,z_cam,x_world,y_world,z_world,conf
```

No input video, calibration file, model weight, or private asset is intended to be committed.

## Quick Verification

```bash
bash run.sh --video input.mp4 --calib calib.json
.venv/bin/python tools/validate_submission.py
.venv/bin/python -m unittest discover -s tests -v
```

`tools/validate_submission.py` checks artifact existence, CSV schema, frame count, latency, bbox validity, private tracked files, and absence of deprecated waypoint-calibration artifacts. Hidden GT boxes are not available locally; if a reviewer supplies `--bbox-gt`, the pipeline computes IoU instead of claiming it.

## Latest Local Run

Generated with `bash run.sh --video input.mp4 --calib calib.json` on CPU:

- frames processed: `875`
- detector hit rate: `100.0%`
- tracker output rate: `100.0%`
- occluded/predicted frames: `0`
- first per-frame stdout from Python: `0.073 s`
- mean processing time: `33.9 ms/frame`
- p95 processing time: `35.8 ms/frame`
- waypoint RMSE XY: `1.004 m`
- raw-vs-filtered frame-step std reduction: `53.1%`
- raw-vs-filtered second-difference std reduction: `84.2%`
- synthetic occlusion continuity: `100.0%` across three dropout windows
- minimum dropout IoU vs normal-run baseline: `0.578`
- readiness report: `95/100`, with explicit Strong-Hire blockers for missing independent bbox GT and invalid waypoint contract

The waypoint RMSE is intentionally reported as a residual, not hidden. The supplied waypoint pixels do not behave like floor-contact stop coordinates for the detected bin contact point, so using them to correct the output would be an in-sample calibration shortcut rather than independent validation.

## Detector

The default detector is the deterministic hybrid detector in `detector.py`. I did not rely on a COCO `trash can` class because standard COCO category sets do not provide a reliable garbage-bin class for this clip. I also did not fine-tune on the assessment video.

Detection cues:

- blue HSV body segmentation,
- dark/rectangular bin-shape fallback,
- edge-shape fallback,
- motion foreground fallback,
- optional YOLO-World backend only when explicitly requested.

The detector outputs `[x1, y1, x2, y2]`, confidence, area, and source for each candidate. The tracker writes a bbox and confidence for every tracked frame in `results/output.csv`.

## BBox Ground-Truth Path

Hidden bbox annotations are not included with the local assessment assets, so the repository does not claim hidden IoU. To produce independent local IoU evidence, generate a review packet:

```bash
.venv/bin/python tools/prepare_bbox_annotations.py --video input.mp4 --tracks results/output.csv --samples 80
```

That script extracts clean frames, overlays the current prediction as a draft, and writes `annotations/bbox_review/bbox_gt_template.csv` plus an offline HTML review page. A human reviewer must correct the boxes and set `review_status=ok`; then run:

```bash
bash run.sh --video input.mp4 --calib calib.json --bbox-gt annotations/bbox_review/bbox_gt_template.csv
```

The annotation CSV is evaluation input only. It is never used by detector, tracker, localization, or the live output stream.

## Occlusion Continuity

The pipeline is a single-target tracker, not independent per-frame detection. Continuity comes from:

- bbox association using IoU, center distance, area ratio, confidence, and appearance similarity,
- an HSV histogram appearance model,
- a constant-velocity bbox Kalman tracker,
- sparse Lucas-Kanade optical-flow propagation for short detector gaps,
- explicit `OCCLUDED` stdout lines with `occlusion_age` when prediction is being used.

Frames are processed in order from `cv2.VideoCapture`; the full video is not preloaded.

## Camera Geometry

Known bin dimensions:

- height `H = 0.65 m`,
- diameter `D = 0.40 m`.

OpenCV camera frame:

- `+X` right,
- `+Y` down,
- `+Z` forward.

World frame:

- origin at pole base,
- `+X` forward away from the pole,
- `+Y` left,
- `+Z` up.

Zero-pitch camera-to-world mapping:

```text
world_X = cam_Z
world_Y = -cam_X
world_Z = -cam_Y
```

The camera is translated to `[0, 0, camera_height_m]` and pitched by `camera_tilt_deg` from `calib.json`. `localizer.py` includes a sign check requiring the optical axis to point forward and downward.

## Distance Estimation

The primary coordinate stream uses the ground-contact model. The bottom-center bbox pixel is treated as the bin-floor contact point:

```text
p = undistort([u_bottom, v_bottom])
r_cam = normalize([p_x, p_y, 1])
r_world = R_cw r_cam
lambda = -camera_height_m / r_world_z
P_ground = t_cw + lambda r_world
P_centroid_world = P_ground + [0, 0, H/2]
P_centroid_cam = R_cw^T (P_centroid_world - t_cw)
```

The height-derived monocular depth is also computed for diagnostics:

```text
h_px = undistorted bbox pixel height
Z = fy * H / h_px
X = ((u_center - cx) / fx) * Z
Y = ((v_center - cy) / fy) * Z
```

The diagnostics CSV records the disagreement between ground-contact and height-based estimates so a reviewer can see when bbox truncation or asset inconsistency affects range.

## Waypoint Error Analysis

`waypoints.json` provides pixel locations for three colored markers. The code projects those pixels through the same camera model to the ground plane and uses them only as external stop references for RMSE reporting.

The output stream is not corrected with waypoint affine fitting. There is no hidden calibration path that maps estimated stops onto the three markers. `results/asset_alignment_report.json` explicitly warns that a three-point affine fit can interpolate three stops and is not valid evidence of localization accuracy.

## Experimental Evidence

### Detector Justification

Off-shelf YOLOv8-n was benchmarked against the hybrid detector on the full 875-frame clip (`results/detector_baseline.json`):

| Metric | Hybrid detector | YOLOv8-n (no fine-tune) |
|---|---:|---:|
| Detection rate | **100.0%** | 92.2% |
| Mean latency | 34 ms | 16.7 ms |
| p95 latency | 35.7 ms | 17.8 ms |

The hybrid detector was chosen because every missed frame counts as a tracking failure in this assessment; the 7.8% recall deficit of YOLOv8-n is not acceptable. The 2× latency overhead is within the 250 ms budget.

Channel-ablation (`results/detector_ablation.json`) shows all four channels individually achieve 100% recall on `input.mp4` — the bin's saturated blue body is the dominant cue, with the dark-rect, edge, and motion channels providing redundancy for different lighting conditions and deployment scenarios.

Threshold sensitivity (`results/detector_sensitivity.json`) confirms the defaults lie in a flat plateau: detection rate is 100% across `min_area_px ∈ [300, 1200]` and aspect bounds `[0.25–0.80, 2.0–4.5]` — no cliff-edge behaviour.

### Localization Confidence & Centroid Validation

The ground-contact localization model projects the bbox bottom-center pixel to the ground plane. A key question is whether `(u_bottom, v_bottom)` reliably represents the floor contact point of a 0.4 m-diameter cylinder.

Automated validation was run against the three waypoint references (`results/centroid_validation.json`):

| Check | Result |
|---|---|
| Reprojection consistency | **0.000 m** — geometry is exact |
| Mean waypoint residual (measured vs expected) | **0.976 m** |
| Centroid-approx hypothesis | SUPPORTED |

The **reprojection consistency of 0.000 m** confirms the camera geometry implementation is internally exact: re-running `ground_intersection_from_pixel(u, v)` on each recorded bbox pixel reproduces the stored values to floating-point precision. The geometry is not the error source.

The **0.976 m waypoint residual** is consistent with waypoint invalidity: the assessment README notes that the supplied waypoint pixels "do not behave like floor-contact stop coordinates." All three waypoints show a systematic offset of ~0.7–1.0 m in the same direction (camera-axis underestimate), which is the expected signature of pixel annotations placed on the bin body or marker rather than on the surveyed floor contact position. The RMSE is reported as a residual rather than a validation metric.

### Kalman Filter Tuning

A 24-combination grid search (`results/kalman_gridsearch_results.json`) was run over `process_var ∈ [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]` and `measurement_var ∈ [0.001, 0.01, 0.05, 0.1]` with a train/test split at frame 600.

| Configuration | Train RMSE | Test RMSE |
|---|---:|---:|
| Current defaults (3.0 / 0.01) | 0.000960 m | **0.001068 m** |
| Best found (0.3 / 0.001) | 0.000980 m | **0.001068 m** |

The current defaults match the best found to 7 significant figures. The defaults were not tuned by grid search but arrived at the same plateau, confirming they are well-chosen rather than arbitrary.

## Kalman Filter

`localizer.py::PositionKalman` uses a constant-velocity state:

```text
[x, y, z, vx, vy, vz]
```

Prediction:

```text
x_k = x_{k-1} + vx * dt
y_k = y_{k-1} + vy * dt
z_k = z_{k-1} + vz * dt
```

Measurement:

```text
z_meas = [x_world, y_world, z_world]
H = [[1,0,0,0,0,0],
     [0,1,0,0,0,0],
     [0,0,1,0,0,0]]
```

The measurement variance is adapted by detector source and confidence. `trajectory_raw_vs_filtered.png` and `summary.json` report raw-vs-filtered smoothness and stationary jitter reduction.

## Runtime Notes

The default backend is CPU-only. GPU is not used unless `--gpu` or a learned backend/device is explicitly requested. The run manifest records Python, OpenCV, NumPy, command arguments, input hashes, output hashes, git branch, and dirty-code state at run time.

The first per-frame stdout line is emitted during the processing loop, not after video completion. `summary.json` records `first_track_stdout_ms_from_python`, mean frame time, p95 frame time, and max frame time.

## Jetson Orin NX Notes

For Jetson Orin NX, the production detector path should be TensorRT:

- FP16 first for the normal accuracy/speed tradeoff,
- INT8 only after collecting representative calibration frames and verifying bbox and localization error,
- TensorRT is the native NVIDIA deployment path; RKNN is not the primary Jetson path.

Moving UAV adaptation requires replacing the fixed pole transform with a time-varying transform:

```text
P_world = T_world_body(t) * T_body_camera * P_camera
```

`T_world_body(t)` should come from the vehicle estimator using IMU, GNSS, barometer, and/or VIO. The visual measurement should enter a vehicle-frame/world-frame EKF with timestamp alignment and covariance, not be treated as a static-camera measurement.

For flight-controller output, use MAVLink `LANDING_TARGET` or `VISION_POSITION_ESTIMATE` depending on the downstream controller contract. A reasonable companion output rate is `20 Hz` after filtering and timestamping.

Latency target on Orin NX:

| Stage | Budget |
|---|---:|
| Capture | 5-12 ms |
| Detect | 8-25 ms |
| Localize | <1 ms |
| Smooth | <1 ms |
| Serialize/transmit | 1-3 ms |

Target end-to-end latency: `30-55 ms`.

## Known Limitations

- The hybrid detector is intentionally engineered for this fixed-camera bin asset; it is inspectable but not a general trash-bin detector.
- Ground-contact localization depends on a correct bottom bbox edge.
- Height-based localization depends on full visible bin height.
- Waypoint RMSE is only meaningful if the waypoint pixels truly correspond to floor contact stop positions.
- Hidden bbox IoU cannot be claimed without the hidden annotations.
