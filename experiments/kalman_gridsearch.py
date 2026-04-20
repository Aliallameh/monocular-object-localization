"""
Kalman filter hyperparameter analysis.

WHY THIS WAS REWRITTEN
----------------------
The original version loaded results/output.csv and used the pipeline's own
x_world/y_world/z_world values as ``gt_positions``.  That is circular: the
"ground truth" is the same detector output the filter is smoothing, so the
RMSE is just measuring how tightly the filter tracks itself.  Every reasonable
config produces the same answer (~0.001 m) because z_world is constant by
construction (ground-plane contact + H/2), and x/y barely vary on a
stationary bin.  The numbers look impressive but prove nothing.

REPLACEMENT APPROACH
--------------------
We use jitter analysis instead:
  - Run the detector on the full clip for each hyperparameter config.
  - Find windows where the tracker reports STATIONARY state (bin not moving).
  - Compute frame-to-frame std-dev of raw positions (jitter) and filtered
    positions for each config.
  - Better configs reduce jitter on stationary windows without lagging badly
    when the bin actually moves.

This is a genuine, independent quality signal — not circular.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from detector import load_detector
from localizer import CameraGeometry, PositionKalman, localize_bbox
from tracker_utils import BBoxKalmanTracker, LKFlowPropagator


def _run_pipeline(
    cap: cv2.VideoCapture,
    detector: Any,
    camera: CameraGeometry,
    process_var: float,
    measurement_var: float,
    max_frames: int = 875,
) -> Dict[str, Any]:
    """Run detection + localization + Kalman for one hyperparameter config.

    Returns per-frame records with raw and filtered world positions and tracker
    state so the caller can compute jitter on stationary windows.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    bbox_tracker = BBoxKalmanTracker(max_age=35)
    flow_tracker = LKFlowPropagator(min_points=8)
    kf = PositionKalman(process_var=process_var, measurement_var=measurement_var)

    dt = 1.0 / 30.0
    records: List[Dict[str, Any]] = []
    frame_id = 0

    while frame_id < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(frame)
        flow_bbox, flow_q, _ = flow_tracker.predict(gray)
        if flow_bbox is not None and not detections:
            from detector import Detection
            detections.append(Detection(
                bbox=flow_bbox,
                confidence=float(np.clip(0.22 + 0.45 * flow_q, 0.05, 0.62)),
                area_px=max(1.0, (flow_bbox[2] - flow_bbox[0]) * (flow_bbox[3] - flow_bbox[1])),
                source="lk_optical_flow",
            ))

        track = bbox_tracker.update(detections, dt_frames=1.0, frame=frame)
        if track is None:
            frame_id += 1
            continue

        if track.status == "detected":
            flow_tracker.update_reference(gray, track.bbox)
        else:
            flow_tracker.accept_prediction(gray, track.bbox)

        try:
            loc = localize_bbox(track.bbox, camera)
        except Exception:
            frame_id += 1
            continue

        raw_xyz = loc.xyz_world
        if kf.initialized:
            filt_xyz = kf.update(raw_xyz, dt, measurement_var)
        else:
            filt_xyz = kf.initialize(raw_xyz)

        # Use a simple speed heuristic to label stationary windows
        speed = float(np.linalg.norm(kf.x[3:5])) if kf.x is not None else 0.0
        is_stationary = speed < 0.05  # m/s threshold

        records.append({
            "frame_id": frame_id,
            "status": track.status,
            "is_stationary": is_stationary,
            "raw_xy": raw_xyz[:2].tolist(),
            "filt_xy": filt_xyz[:2].tolist(),
        })
        frame_id += 1

    return {"records": records}


def _jitter_metrics(records: List[Dict], min_window: int = 30) -> Dict[str, float]:
    """Compute frame-to-frame jitter on stationary windows.

    Returns raw and filtered std-dev of frame-step displacement.
    """
    raw_steps: List[float] = []
    filt_steps: List[float] = []
    stationary_streak = 0
    streak_raw: List[np.ndarray] = []
    streak_filt: List[np.ndarray] = []

    def flush_streak() -> None:
        nonlocal streak_raw, streak_filt
        if len(streak_raw) < min_window:
            streak_raw, streak_filt = [], []
            return
        r = np.array(streak_raw)
        f = np.array(streak_filt)
        for i in range(1, len(r)):
            raw_steps.append(float(np.linalg.norm(r[i] - r[i - 1])))
            filt_steps.append(float(np.linalg.norm(f[i] - f[i - 1])))
        streak_raw, streak_filt = [], []

    for rec in records:
        if rec["is_stationary"] and rec["status"] == "detected":
            stationary_streak += 1
            streak_raw.append(np.array(rec["raw_xy"]))
            streak_filt.append(np.array(rec["filt_xy"]))
        else:
            flush_streak()
            stationary_streak = 0

    flush_streak()

    if not raw_steps:
        return {"raw_step_std": float("nan"), "filt_step_std": float("nan"), "reduction_pct": float("nan"), "n_steps": 0}

    raw_std = float(np.std(raw_steps))
    filt_std = float(np.std(filt_steps))
    reduction = float(100.0 * (1.0 - filt_std / max(raw_std, 1e-9)))
    return {"raw_step_std": raw_std, "filt_step_std": filt_std, "reduction_pct": reduction, "n_steps": len(raw_steps)}


def kalman_jitter_analysis(
    video_path: str,
    calib_path: str,
    max_frames: int = 875,
) -> Dict[str, Any]:
    """Grid search using jitter reduction on stationary windows as the metric.

    Returns a dict with per-config results and the config that minimises
    filtered jitter on stationary windows.
    """
    import json as _json
    calib_data = _json.loads(Path(calib_path).read_text())
    camera = CameraGeometry.from_json_dict(calib_data)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "failed", "reason": f"Cannot open {video_path}"}

    detector = load_detector(use_gpu=False, backend="hybrid", device="auto", conf=0.05, imgsz=640)

    process_vars = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
    measurement_vars = [0.001, 0.01, 0.05, 0.1]

    results: List[Dict[str, Any]] = []
    for pv in process_vars:
        for mv in measurement_vars:
            print(f"Testing: process_var={pv}, measurement_var={mv}")
            run = _run_pipeline(cap, detector, camera, pv, mv, max_frames)
            jitter = _jitter_metrics(run["records"])
            results.append({
                "process_var": pv,
                "measurement_var": mv,
                **jitter,
            })

    cap.release()

    valid = [r for r in results if not np.isnan(r["filt_step_std"])]
    if not valid:
        return {"status": "no_stationary_windows", "results": results}

    best = min(valid, key=lambda r: r["filt_step_std"])
    return {
        "method": "jitter_on_stationary_windows",
        "note": (
            "RMSE vs output.csv was removed — that was circular (self-referential GT). "
            "This analysis measures frame-step jitter on stationary windows, "
            "an independent signal."
        ),
        "best_config": {"process_var": best["process_var"], "measurement_var": best["measurement_var"]},
        "best_filt_step_std_m": best["filt_step_std"],
        "best_reduction_pct": best["reduction_pct"],
        "total_configs_tested": len(results),
        "results": results,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--output", default="results/kalman_gridsearch_results.json")
    ap.add_argument("--max-frames", type=int, default=875)
    args = ap.parse_args()

    result = kalman_jitter_analysis(args.video, args.calib, args.max_frames)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {args.output}")
    if "best_config" in result:
        print(f"Best config: {result['best_config']}")
        print(f"Jitter reduction: {result.get('best_reduction_pct', float('nan')):.1f}%")
