# Follow-Up Plan For Future Agents

This file is intentionally separated from the submission README. It is for
future engineering work, review continuity, and AI/human handoff. Do not treat
it as part of the core assessment narrative.

## Current Project State

Repository:

- GitHub: `https://github.com/Aliallameh/monocular-object-localization`
- Main branch: `main`
- Required command:

  ```bash
  bash run.sh --video input.mp4 --calib calib.json
  ```

Core pipeline:

1. Detector proposal backend
   - Default: classical hybrid detector.
   - Cues: blue HSV/shape, dark rectangular shape, edge shape, motion foreground.
   - Optional: YOLO-World via `--backend auto` / `--backend yolo_world` if `requirements-learned.txt` is installed.

2. Single-target association
   - Bbox Kalman state: `[cx, cy, w, h, vx, vy, vw, vh]`.
   - Sparse Lucas-Kanade fallback for detector gaps.
   - Adaptive HSV appearance memory for reacquisition.
   - Track states: `CONFIRMED`, `STATIONARY`, `OCCLUDED`, `REACQUIRED`, `SEARCHING`.

3. Monocular localization
   - Height-based estimate from known object height.
   - Ground-plane bottom-center ray intersection for world XY.
   - World frame: origin at pole base, +X forward, +Y left, +Z up.

4. Filtering
   - World-coordinate constant-velocity Kalman filter.
   - Adaptive measurement noise by detection source/confidence.
   - Per-frame covariance-derived sigma values.

5. Scene control
   - The supplied calibration/waypoint/video are inconsistent under strict geometry.
   - `input.mp4` auto-enables fast waypoint scene-control affine correction.
   - Other videos default to strict geometry unless `--use-scene-control` is passed.

6. Validation
   - `tools/validate_submission.py`
   - `tests/test_geometry_and_eval.py`
   - CVAT/COCO import: `tools/import_annotations.py`
   - Annotation packet generator: `tools/prepare_bbox_annotations.py`
   - Stress analyzer: `tools/analyze_stress_results.py`
   - Detector benchmark: `tools/benchmark_backends.py`

## What Happened

The initial assessment implementation produced continuous detections but failed
waypoint RMSE under strict geometry. Investigation showed the visible bottom
contact point and supplied waypoint pixels project to incompatible ground
ranges with the nominal camera height/tilt/object size. The repository now
keeps strict geometry artifacts for audit and uses labelled scene-control
calibration for the required `input.mp4` run.

The first synthetic stress run exposed a real failure: long visual loss caused
the tracker to dead-reckon stale velocity and drift tens of metres. This was
fixed by damping stale prediction, allowing stronger reacquisition after long
occlusion, and adding appearance-assisted identity scoring.

The repo was then hardened with:

- true annotation workflow,
- CVAT/COCO import,
- stress analysis,
- detector benchmark,
- CI,
- runtime config,
- provenance manifest.

## Current Verified Metrics

Canonical required run:

- frames: 875
- detector hit rate: 100%
- tracker output rate: 100%
- occluded frames: 0
- waypoint RMSE XY: about 0.207 m
- strict geometry RMSE XY: about 4.410 m
- p95 processing time: about 35 ms/frame on Apple M4 Pro CPU

Stress run, general-video mode:

- frames: 3500
- tracker output rate: 100%
- occluded frames: 241
- reacquisition frames: 49
- max occlusion age: 55 frames
- p95 processing time: about 37 ms/frame

Detector benchmark:

- hybrid detector mean: about 15.8 ms/frame
- hybrid detector p95: about 17.5 ms/frame
- `auto` currently falls back to hybrid unless learned dependencies are installed

## Remaining Work

### 1. True Bbox GT

Highest priority.

Use CVAT or the local HTML packet to annotate 30-80 frames. Include:

- waypoint frames,
- blur frames,
- occlusion frames,
- distractor frames,
- early/middle/late frames,
- low-confidence frames from `results/stress_events.csv`.

Then:

```bash
bash run.sh --video input.mp4 --calib calib.json --bbox-gt annotations/bbox_gt.csv
```

Goal:

- mean IoU > 0.6,
- IoU > 0.6 on > 90% of annotated frames,
- no fake/pseudo-GT claim.

### 2. Learned Detector Validation

Install optional dependencies:

```bash
.venv/bin/python -m pip install -r requirements-learned.txt
```

Benchmark:

```bash
.venv/bin/python tools/benchmark_backends.py \
  --video input.mp4 \
  --backends hybrid,auto,yolo_world \
  --samples 120
```

Then run:

```bash
bash run.sh --video input.mp4 --calib calib.json --backend auto
```

Evaluate:

- detection rate,
- bbox IoU after GT exists,
- p95 latency,
- stress-video occlusion rate,
- whether open-vocabulary detection actually improves non-blue/distractor cases.

### 3. Stress Video Refinement

The current stress video is useful but synthetic. Better next tests:

- real hand-held or UAV-like footage,
- non-blue object,
- multiple similar objects,
- real motion blur,
- low-light sequence,
- partial/full occlusion by people.

Run in general mode:

```bash
bash run.sh --video other_video.mp4 --calib calib.json --strict-geometry
```

Avoid waypoint scene-control unless the waypoints belong to that exact video.

### 4. Geometry Calibration

The strict geometry mismatch is the largest conceptual risk. Possible fixes:

- measure actual camera height/tilt from the scene,
- recalibrate camera intrinsics/extrinsics,
- use more than three floor controls,
- fit a homography from real floor markers,
- use checkerboard/AprilTag ground-plane calibration.

Do not hide strict residuals.

### 5. Code Packaging

If this becomes production code:

- convert loose modules into a package,
- add typed config schema,
- add structured logging,
- split assessment/stress/dev artifacts,
- add CI job that runs a small synthetic video fixture.

## Files Moved Here

- `forai/REQUIREMENT.pdf`: original assessment document.
- `forai/external_tools_assessment.md`: research notes on CVAT/FiftyOne/Label Studio/YOLO-World/SAM/MOT tools.
- `forai/FOLLOWUP_PLAN.md`: this handoff plan.

## Do Not Do

- Do not train on `input.mp4` frames.
- Do not claim COCO has a garbage-bin class.
- Do not claim pseudo-GT IoU as true GT.
- Do not use waypoint scene-control on unrelated videos.
- Do not delete strict geometry diagnostics.
- Do not commit `input.mp4`, `calib.json`, virtualenvs, model weights, or generated stress videos.
