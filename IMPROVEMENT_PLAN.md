# Monocular Bin Tracking — Comprehensive Improvement Plan

**Objective:** Implement the improvements identified in the CTO technical review, with evidence-based validation. Do NOT hardcode ground truth; all improvements must be independently verifiable.

**Branch Strategy:** Work on `temp1`; merge to `main` when all workstreams pass verification.

---

## Workstream A: Finalize & Verify Observer Module ✓ FIRST

**Goal:** Ensure observer visualization (overlay.mp4 + events.json) is production-ready.

### Tasks
- [ ] Verify observer.py generates both results/observer_overlay.mp4 AND results/observer_events.json
- [ ] Confirm motion states are correctly labeled (STATIONARY, MOVING, CHANGING_DIRECTION, OCCLUDED, etc.)
- [ ] Verify overlay shows: bbox, confidence, detector source, world coordinates, uncertainty
- [ ] Run full pipeline: `bash run.sh --video input.mp4 --calib calib.json`
- [ ] Verify summary.json includes observer metadata
- [ ] Commit observer work to main

### Evidence: Results/observer_overlay.mp4 (31 MB), results/observer_events.json, summary.json

---

## Workstream B: Detector Improvement with Evidence ✓ SECOND

**Goal:** Understand why hybrid detector works; compare to alternatives; justify thresholds.

### Tasks
- [ ] **Baseline (YOLOv8):** Run YOLOv8 off-the-shelf (no fine-tune) on input.mp4; record mAP, FPS, per-class recall
  - File: `experiments/detector_baseline_yolo.py`
  - Output: `results/detector_baseline.json` with mAP, latency, per-frame timings
  
- [ ] **Ablation Study:** Disable each proposal channel (blue HSV, dark rect, edge, motion); measure recall drop
  - File: `experiments/detector_ablation.py`
  - Output: `results/detector_ablation.csv` with detection_rate by channel combination
  
- [ ] **Sensitivity Analysis:** Vary HSV range, aspect ratio bounds, min area; plot detection count vs parameter
  - File: `experiments/detector_sensitivity.py`
  - Output: `results/detector_sensitivity_plots/` with heatmaps showing plateau regions
  
- [ ] **Justification:** Document in README section "Detector Design Rationale"
  - Why hybrid over YOLO? (e.g., "YOLO mAP=X, hybrid recall=Y")
  - Why these aspect ratios / min area? (e.g., "plateau between 0.40–0.55; chose 0.45")

### Evidence: results/detector_baseline.json, detector_ablation.csv, detector_sensitivity_plots, updated README

---

## Workstream C: Localization Error Decomposition ✓ THIRD

**Goal:** Quantify sources of 1.004m RMSE; establish confidence region on coordinates.

### Tasks
- [ ] **Synthetic Ground Truth:** Generate test frames with known 3D positions (checkerboard on floor)
  - File: `experiments/synthetic_calibration_frames.py`
  - Output: synthetic_calib_frames/, ground_truth_positions.json
  
- [ ] **Bbox Uncertainty:** Measure width/height noise by comparing predicted bbox to synthetic GT
  - File: `experiments/bbox_uncertainty.py`
  - Output: `results/bbox_noise_distribution.json` (σ_width, σ_height)
  
- [ ] **Calibration Sensitivity:** Perturb K, tilt, height by ±2, ±5%; measure RMSE change
  - File: `experiments/calibration_sensitivity.py`
  - Output: `results/calibration_sensitivity.json` (∂RMSE/∂K, ∂RMSE/∂tilt, etc.)
  
- [ ] **Centroid Approximation Error:** Analyze bin floor-contact vs bottom-center pixel
  - Manually measure 3–5 key frames: overlay bin contour, mark floor contact, record pixel offset
  - File: `experiments/centroid_approximation.py`
  - Output: `results/centroid_offset_analysis.json`
  
- [ ] **Confidence Interval:** Combine error sources into final RMSE confidence band
  - File: `experiments/localization_confidence.py`
  - Output: `results/localization_error_budget.json` with 68% / 95% bounds
  
- [ ] **Documentation:** Add to README section "Localization Accuracy & Confidence"
  - Table: "Error sources: bbox uncertainty (σ), calibration uncertainty (σ), centroid approx (σ)"
  - Plot: RMSE vs distance; confidence band

### Evidence: results/localization_error_budget.json, error-source decomposition, README update

---

## Workstream D: Kalman Hyperparameter Tuning ✓ FOURTH

**Goal:** Replace magic constants (process_var=3.0, measurement_var=0.01) with tuned values.

### Tasks
- [ ] **Grid Search:** Sweep process_var ∈ [0.3, 0.5, 1.0, 2.0, 3.0, 5.0], measurement_var ∈ [0.001, 0.01, 0.05, 0.1]
  - File: `experiments/kalman_gridsearch.py`
  - Train on frames 0–600, test on 600–875
  - Metrics: RMSE, smoothness (frame-step std reduction), jitter
  - Output: `results/kalman_gridsearch_results.json`, heatmap.png
  
- [ ] **Best Hyperparameters:** If significantly better than default, update config_utils.py
  - Threshold: improvement > 5% RMSE reduction OR > 10% jitter reduction
  
- [ ] **Sensitivity Plot:** Show that best values are in a plateau, not a cliff
  - File: output heatmap to `results/kalman_sensitivity_heatmap.png`
  
- [ ] **Documentation:** Add to README section "Kalman Filter Tuning"
  - Explain why tuned values differ from defaults
  - Show cross-validation RMSE curves

### Evidence: results/kalman_gridsearch_results.json, heatmap.png, README update

---

## Workstream E: Real Occlusion Testing ✓ FIFTH

**Goal:** Validate occlusion handling against real (not synthetic) person occlusion in input.mp4.

### Tasks
- [ ] **Manual Frame Annotation:** Identify frames in input.mp4 where person occludes bin
  - Mark: frame_start, frame_end, occlusion_type (full, partial, brief)
  - Output: `experiments/real_occlusion_frames.json`
  
- [ ] **Continuity Metrics:** For each occlusion, measure tracker IoU and center error vs baseline
  - File: `experiments/real_occlusion_analysis.py`
  - Metrics: IoU (min, mean), center error (max, mean), flow-assisted count
  - Output: `results/real_occlusion_analysis.json`
  
- [ ] **Comparison:** Compare real occlusion to synthetic dropout results
  - Table: "Real occlusion: X% continuity, Y pixel error vs synthetic Z%, Z' error"
  - Analysis: why differences? (synthetic is harder/easier?)
  
- [ ] **Documentation:** Add to README section "Occlusion Robustness"
  - Real vs synthetic comparison
  - Limitations of synthetic testing

### Evidence: results/real_occlusion_analysis.json, comparison table, README update

---

## Workstream F: Integration & Verification ✓ FINAL

**Goal:** Merge all improvements; verify pipeline integrity; finalize artifacts.

### Tasks
- [ ] Run full pipeline on main: `bash run.sh --video input.mp4 --calib calib.json`
- [ ] Verify all results/ artifacts exist and are valid (schema check)
- [ ] Run unit tests: `python -m unittest discover -s tests -v`
- [ ] Type-check (if mypy available): `mypy detector.py localizer.py track_bin.py observer.py`
- [ ] Commit improvements: atomic commits per workstream
- [ ] Update README with all new sections from A–E
- [ ] Generate final summary: `ls -lh results/` + artifact count + timing

### Evidence: results/ directory, test output, type-check output, git log

---

## File Structure for Experiments

```
experiments/
  detector_baseline_yolo.py
  detector_ablation.py
  detector_sensitivity.py
  synthetic_calibration_frames.py
  bbox_uncertainty.py
  calibration_sensitivity.py
  centroid_approximation.py
  localization_confidence.py
  kalman_gridsearch.py
  real_occlusion_frames.json
  real_occlusion_analysis.py
```

---

## Success Criteria

✓ All workstreams complete with independent verification  
✓ No hardcoded ground truth; only algorithmic / sensitivity evidence  
✓ README updated with all improvements and justifications  
✓ results/ contains observer video, error budgets, gridsearch heatmap  
✓ Full pipeline runs end-to-end; artifacts match schema  
✓ Unit tests pass; code style consistent  
✓ Commits are atomic and well-documented  

---

## Timeline Estimate

| Workstream | Est. Time | Dependencies |
|---|---|---|
| A: Observer finalization | 10 min | None |
| B: Detector evidence | 60 min | A (optional) |
| C: Localization decomp | 90 min | None |
| D: Kalman tuning | 45 min | None |
| E: Real occlusion | 30 min | None |
| F: Integration | 20 min | A–E |
| **TOTAL** | **255 min** (~4.5 hr) | Sequential |

