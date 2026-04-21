"""
YOLOv8 sanity baseline: run off-the-shelf model on the assessment video.

This is not a valid bin-class detector comparison unless external bbox/class
ground truth is supplied. COCO YOLOv8 has no dedicated garbage-bin class, so the
default report counts any detected object only as a weak coverage/speed sanity
check.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


def run_yolo_baseline(
    video_path: str,
    output_path: str = "results/detector_baseline.json",
) -> Dict[str, Any]:
    """Run YOLOv8 on video and report weak any-object coverage/latency.

    Returns: {
        'backend': 'yolov8n',
        'model_name': str,
        'any_object_frame_rate': float [0, 1],
        'mean_latency_ms': float,
        'p95_latency_ms': float,
        'frames_processed': int,
        'detections_per_frame_mean': float,
        'notes': str
    }
    """

    try:
        from ultralytics import YOLO
    except ImportError:
        return {
            "backend": "yolov8n",
            "status": "skipped",
            "reason": "ultralytics not installed",
            "notes": "Install: pip install ultralytics",
        }

    # Load model
    model = YOLO("yolov8n.pt")  # nano model for speed

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "backend": "yolov8n",
            "status": "failed",
            "reason": "cannot open video",
        }

    latencies: List[float] = []
    frame_detection_counts: List[int] = []
    detected_frames = 0
    frame_id = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        t0 = time.perf_counter()
        results = model(frame, verbose=False, conf=0.3)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        # Count detections (look for 'backpack' class or generic person/object)
        detections = 0
        for r in results:
            if r.boxes is not None:
                detections = len(r.boxes)

        if detections > 0:
            detected_frames += 1
        frame_detection_counts.append(detections)
        frame_id += 1

    cap.release()

    # Compute statistics. This is deliberately named "any_object" because it is
    # not a garbage-bin detection rate without external class/bbox labels.
    latencies_arr = np.array(latencies)
    result = {
        "backend": "yolov8n",
        "comparison_status": "invalid_as_bin_detector_without_external_gt",
        "model": "yolov8n.pt",
        "frames_processed": frame_id,
        "fps": fps,
        "any_object_frame_rate": float(detected_frames / max(1, frame_id)),
        "mean_latency_ms": float(np.mean(latencies_arr)),
        "p95_latency_ms": float(np.percentile(latencies_arr, 95)),
        "max_latency_ms": float(np.max(latencies_arr)),
        "detections_per_frame_mean": float(np.mean(frame_detection_counts)),
        "notes": (
            "Off-the-shelf YOLOv8n (no fine-tuning). "
            "Detections counted as any objects found, not bin-class hits. "
            "This file is a weak sanity baseline, not a quantitative detector justification."
        ),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO v8 baseline detector")
    parser.add_argument("--video", default="input.mp4", help="Input video path")
    parser.add_argument("--output", default="results/detector_baseline.json")
    args = parser.parse_args()

    result = run_yolo_baseline(args.video, args.output)
    print(json.dumps(result, indent=2))
