"""Kalman smoothing-strength sweep.

We have no independent positional GT for the bin in this clip, so we don't
compute a position RMSE (that would be circular against our own detector
output). Instead: run the pipeline once, cache the raw localizer positions,
then replay the Kalman over the same cached signal for every (pv, mv) pair.
Each config is scored on frame-step and 2nd-difference std reduction. Same
input for every config, so results are comparable and deterministic.
"""

from __future__ import annotations

import argparse
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


def _collect_raw_track(video_path: str, calib_path: str, max_frames: int) -> Dict[str, Any]:
    """Run the pipeline once and cache per-frame raw world positions."""
    calib = json.loads(Path(calib_path).read_text())
    camera = CameraGeometry.from_json_dict(calib)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    detector = load_detector(use_gpu=False, backend="hybrid", device="auto", conf=0.05, imgsz=640)
    bbox_tracker = BBoxKalmanTracker(max_age=35)
    flow_tracker = LKFlowPropagator(min_points=8)

    fps = cap.get(cv2.CAP_PROP_FPS) or camera.fps
    dt = 1.0 / max(1e-6, float(fps))

    raw_xy: List[List[float]] = []
    status_seq: List[str] = []
    frame_id = 0
    while frame_id < max_frames:
        ok, frame = cap.read()
        if not ok:
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
        raw_xy.append(loc.xyz_world[:2].tolist())
        status_seq.append(track.status)
        frame_id += 1

    cap.release()
    return {"raw_xy": raw_xy, "status": status_seq, "dt": dt, "frames": frame_id}


def _replay_kalman(
    raw_xy: List[List[float]],
    dt: float,
    process_var: float,
    measurement_var: float,
) -> Dict[str, float]:
    kf = PositionKalman(process_var=process_var, measurement_var=measurement_var)
    filt_xy: List[List[float]] = []
    for xy in raw_xy:
        xyz = np.array([xy[0], xy[1], 0.325], dtype=np.float64)
        if kf.initialized:
            out = kf.update(xyz, dt, measurement_var)
        else:
            out = kf.initialize(xyz)
        filt_xy.append([float(out[0]), float(out[1])])

    raw = np.asarray(raw_xy, dtype=np.float64)
    flt = np.asarray(filt_xy, dtype=np.float64)
    if len(raw) < 3:
        return {"n": len(raw)}

    raw_step = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    flt_step = np.linalg.norm(np.diff(flt, axis=0), axis=1)
    raw_d2 = np.linalg.norm(np.diff(raw, n=2, axis=0), axis=1)
    flt_d2 = np.linalg.norm(np.diff(flt, n=2, axis=0), axis=1)
    raw_low_motion_std, flt_low_motion_std, low_motion_count = _low_motion_radial_jitter(raw, flt)

    raw_step_std = float(np.std(raw_step))
    flt_step_std = float(np.std(flt_step))
    raw_d2_std = float(np.std(raw_d2))
    flt_d2_std = float(np.std(flt_d2))

    return {
        "n_frames": int(len(raw)),
        "raw_step_std_m": raw_step_std,
        "filt_step_std_m": flt_step_std,
        "frame_step_reduction_pct": 100.0 * (1.0 - flt_step_std / max(raw_step_std, 1e-9)),
        "raw_second_diff_std_m": raw_d2_std,
        "filt_second_diff_std_m": flt_d2_std,
        "second_diff_reduction_pct": 100.0 * (1.0 - flt_d2_std / max(raw_d2_std, 1e-9)),
        "low_motion_window_frames": int(low_motion_count),
        "raw_low_motion_radial_std_m": raw_low_motion_std,
        "filt_low_motion_radial_std_m": flt_low_motion_std,
        "low_motion_jitter_reduction_pct": (
            100.0 * (1.0 - flt_low_motion_std / max(raw_low_motion_std, 1e-9))
            if np.isfinite(raw_low_motion_std) and np.isfinite(flt_low_motion_std)
            else float("nan")
        ),
    }


def _low_motion_radial_jitter(raw: np.ndarray, flt: np.ndarray) -> tuple[float, float, int]:
    """Proxy stationary jitter on the slowest contiguous part of the track.

    No surveyed stop intervals exist for this clip. This uses the lowest-velocity
    quartile of the raw signal as a low-motion proxy and reports variance about
    each selected sample set's own centroid. It is smoothing evidence only.
    """

    if len(raw) < 12:
        return float("nan"), float("nan"), 0
    speed = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    threshold = float(np.quantile(speed, 0.25))
    mask = np.concatenate([[False], speed <= threshold])
    if int(np.count_nonzero(mask)) < 6:
        return float("nan"), float("nan"), int(np.count_nonzero(mask))
    raw_sel = raw[mask]
    flt_sel = flt[mask]
    raw_radial = np.linalg.norm(raw_sel - np.mean(raw_sel, axis=0), axis=1)
    flt_radial = np.linalg.norm(flt_sel - np.mean(flt_sel, axis=0), axis=1)
    return float(np.std(raw_radial)), float(np.std(flt_radial)), int(len(raw_sel))


def kalman_replay_gridsearch(
    video_path: str,
    calib_path: str,
    max_frames: int = 875,
) -> Dict[str, Any]:
    cache = _collect_raw_track(video_path, calib_path, max_frames)
    raw_xy = cache["raw_xy"]
    dt = cache["dt"]

    process_vars = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
    measurement_vars = [0.001, 0.01, 0.05, 0.1]

    results: List[Dict[str, Any]] = []
    for pv in process_vars:
        for mv in measurement_vars:
            print(f"Replay: process_var={pv}, measurement_var={mv}")
            m = _replay_kalman(raw_xy, dt, pv, mv)
            results.append({"process_var": pv, "measurement_var": mv, **m})

    # Report both "best by frame-step" and "best by 2nd-diff" so the tradeoff
    # is visible instead of hidden behind one single ranking.
    valid = [r for r in results if "filt_step_std_m" in r and np.isfinite(r["filt_step_std_m"])]
    best_step = max(valid, key=lambda r: r["frame_step_reduction_pct"]) if valid else None
    best_d2 = max(valid, key=lambda r: r["second_diff_reduction_pct"]) if valid else None

    return {
        "method": "offline_kalman_replay_over_cached_raw_track",
        "note": (
            "No independent positional GT for the bin in this clip. We report "
            "smoothing strength only (frame-step and 2nd-difference std "
            "reduction), not position RMSE."
        ),
        "low_motion_jitter_note": (
            "Low-motion jitter is computed from the lowest-velocity quartile of "
            "the raw replayed track. It is not surveyed stop-position accuracy."
        ),
        "raw_track_frames": len(raw_xy),
        "dt_s": dt,
        "pipeline_default": {"process_var": 3.0, "measurement_var": 0.010},
        "best_by_frame_step_reduction": (
            {
                "process_var": best_step["process_var"],
                "measurement_var": best_step["measurement_var"],
                "frame_step_reduction_pct": best_step["frame_step_reduction_pct"],
                "second_diff_reduction_pct": best_step["second_diff_reduction_pct"],
            }
            if best_step is not None
            else None
        ),
        "best_by_second_diff_reduction": (
            {
                "process_var": best_d2["process_var"],
                "measurement_var": best_d2["measurement_var"],
                "frame_step_reduction_pct": best_d2["frame_step_reduction_pct"],
                "second_diff_reduction_pct": best_d2["second_diff_reduction_pct"],
            }
            if best_d2 is not None
            else None
        ),
        "caveat": (
            "Tuning on one video without a validation set is a single-scene "
            "fit. The pipeline keeps process_var=3.0, measurement_var=0.010 "
            "rather than the grid optimum."
        ),
        "results": results,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--output", default="results/kalman_gridsearch_results.json")
    ap.add_argument("--max-frames", type=int, default=875)
    args = ap.parse_args()

    result = kalman_replay_gridsearch(args.video, args.calib, args.max_frames)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {args.output}")
    best = result.get("best_by_frame_step_reduction")
    if best:
        print(f"Best (frame-step): pv={best['process_var']}, mv={best['measurement_var']} "
              f"-> {best['frame_step_reduction_pct']:.1f}% reduction")
