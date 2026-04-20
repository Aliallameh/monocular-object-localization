"""Synthetic robustness checks for detector dropout / blur continuity."""

from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

from detector import Detection, BlueBinDetector
from frame_quality import assess_frame_quality
from tracker_utils import BBoxKalmanTracker, LKFlowPropagator, bbox_iou


def run_dropout_stress_test(
    video_path: str,
    detector: BlueBinDetector,
    baseline_rows: List[Dict[str, Any]],
    frame_start: int = 150,
    frame_end: int = 230,
    dropout_start: int = 185,
    dropout_end: int = 210,
    scenario_name: str = "mid_stop_dropout",
) -> Dict[str, Any]:
    """Suppress detections in a short interval and check continuity.

    The baseline boxes come from the normal run. During the dropout interval we
    force the detector candidate list to empty and allow only optical-flow and
    Kalman propagation. This tests the continuity path without fabricating GT.
    """

    baseline = {int(row["frame_id"]): tuple(float(v) for v in row["bbox"]) for row in baseline_rows}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"enabled": False, "reason": "video_open_failed"}

    tracker = BBoxKalmanTracker(max_age=35)
    flow = LKFlowPropagator(min_points=8)
    frame_id = frame_start
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    tracked = 0
    flow_frames = 0
    dropout_tracked = 0
    dropout_ious: List[float] = []
    dropout_center_errors: List[float] = []
    status_counts: Dict[str, int] = {}

    while frame_id <= frame_end:
        ok, frame = cap.read()
        if not ok:
            break
        # Mild blur only in the dropout interval to exercise the harder path.
        test_frame = cv2.GaussianBlur(frame, (9, 9), 0) if dropout_start <= frame_id <= dropout_end else frame
        gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
        quality = assess_frame_quality(test_frame)

        detections = [] if dropout_start <= frame_id <= dropout_end else detector.detect(test_frame)
        flow_bbox, flow_quality, _ = flow.predict(gray)
        if flow_bbox is not None and (not detections or quality.is_blurry):
            flow_frames += 1
            detections.append(
                Detection(
                    bbox=flow_bbox,
                    confidence=float(np.clip(0.22 + 0.45 * flow_quality, 0.05, 0.62)),
                    area_px=max(1.0, (flow_bbox[2] - flow_bbox[0]) * (flow_bbox[3] - flow_bbox[1])),
                    source="lk_optical_flow",
                )
            )

        track = tracker.update(detections, dt_frames=1.0, frame=test_frame)
        if track is not None:
            tracked += 1
            if track.matched_detection is not None and track.matched_detection.source == "lk_optical_flow":
                flow.accept_prediction(gray, track.bbox)
            elif track.status == "detected":
                flow.update_reference(gray, track.bbox)
            status_counts[track.status] = status_counts.get(track.status, 0) + 1
            if dropout_start <= frame_id <= dropout_end:
                dropout_tracked += 1
                ref = baseline.get(frame_id)
                if ref is not None:
                    dropout_ious.append(bbox_iou(track.bbox, ref))
                    dropout_center_errors.append(_center_error(track.bbox, ref))

        frame_id += 1

    cap.release()
    processed = max(0, frame_id - frame_start)
    dropout_total = max(0, min(frame_end, dropout_end) - dropout_start + 1)
    return {
        "enabled": True,
        "name": scenario_name,
        "scenario": "detections_suppressed_and_frames_blurred",
        "frame_start": frame_start,
        "frame_end": frame_end,
        "dropout_start": dropout_start,
        "dropout_end": dropout_end,
        "frames_processed": processed,
        "tracker_output_rate": tracked / max(1, processed),
        "dropout_frames": dropout_total,
        "dropout_tracked_frames": dropout_tracked,
        "dropout_continuity_rate": dropout_tracked / max(1, dropout_total),
        "flow_assisted_frames": flow_frames,
        "status_counts": status_counts,
        "dropout_mean_iou_vs_baseline": _safe_mean(dropout_ious),
        "dropout_min_iou_vs_baseline": _safe_min(dropout_ious),
        "dropout_mean_center_error_px": _safe_mean(dropout_center_errors),
        "dropout_max_center_error_px": _safe_max(dropout_center_errors),
    }


def run_dropout_stress_suite(
    video_path: str,
    detector: BlueBinDetector,
    baseline_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run several detector-dropout windows against the same baseline output."""

    scenarios = [
        {
            "scenario_name": "mid_stop_dropout",
            "frame_start": 150,
            "frame_end": 230,
            "dropout_start": 185,
            "dropout_end": 210,
        },
        {
            "scenario_name": "moving_crossing_dropout",
            "frame_start": 412,
            "frame_end": 470,
            "dropout_start": 432,
            "dropout_end": 445,
        },
        {
            "scenario_name": "late_low_confidence_dropout",
            "frame_start": 728,
            "frame_end": 787,
            "dropout_start": 748,
            "dropout_end": 762,
        },
    ]
    reports = [run_dropout_stress_test(video_path, detector, baseline_rows, **scenario) for scenario in scenarios]
    enabled_reports = [r for r in reports if r.get("enabled")]
    continuity = [float(r["dropout_continuity_rate"]) for r in enabled_reports]
    min_iou = [
        float(r["dropout_min_iou_vs_baseline"])
        for r in enabled_reports
        if r.get("dropout_min_iou_vs_baseline") is not None
    ]
    max_center = [
        float(r["dropout_max_center_error_px"])
        for r in enabled_reports
        if r.get("dropout_max_center_error_px") is not None
    ]
    return {
        "enabled": bool(enabled_reports),
        "scenario_count": len(enabled_reports),
        "reports": reports,
        "min_dropout_continuity_rate": _safe_min(continuity),
        "min_dropout_iou_vs_baseline": _safe_min(min_iou),
        "max_dropout_center_error_px": _safe_max(max_center),
        "interpretation": (
            "Detections are artificially suppressed and blurred; continuity is compared "
            "against the normal pipeline output, not hidden ground truth."
        ),
    }


def _center_error(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ac = np.array([(a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5], dtype=np.float64)
    bc = np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5], dtype=np.float64)
    return float(np.linalg.norm(ac - bc))


def _safe_mean(values: List[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _safe_min(values: List[float]) -> float | None:
    return float(np.min(values)) if values else None


def _safe_max(values: List[float]) -> float | None:
    return float(np.max(values)) if values else None
