"""
Detector channel ablation: measure recall impact of each proposal source.

Purpose: Justify that all four channels (blue HSV, dark rect, edge, motion)
contribute meaningfully to detection rate. This prevents unsupported claims
like "motion foreground is unnecessary" when we haven't measured.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from detector import HybridBinDetector


def _detection_rate(video_path: str, detector: HybridBinDetector) -> tuple[float, int]:
    """Run detector over all frames; return (detection_rate, total_frames)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    hits: List[int] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hits.append(1 if detector.detect(frame) else 0)
    cap.release()
    return float(np.mean(hits)) if hits else 0.0, len(hits)


def run_ablation(
    video_path: str,
    output_path: str = "results/detector_ablation.json",
) -> Dict[str, Any]:
    """Ablate each detector channel and measure detection-rate drop."""

    ablations = [
        ("without_blue_hsv",         dict(enable_blue=False)),
        ("without_dark_rect",        dict(enable_dark=False)),
        ("without_edge_shape",       dict(enable_edge=False)),
        ("without_motion_foreground",dict(enable_motion=False)),
    ]

    print("Running baseline (all channels enabled)…")
    baseline_rate, total_frames = _detection_rate(video_path, HybridBinDetector())
    print(f"  baseline detection rate: {baseline_rate:.4f} ({total_frames} frames)")

    channel_results: Dict[str, Any] = {}
    for label, kwargs in ablations:
        print(f"Running ablation: {label}…")
        rate, _ = _detection_rate(video_path, HybridBinDetector(**kwargs))
        drop = baseline_rate - rate
        channel_results[label] = {
            "detection_rate": round(rate, 4),
            "recall_drop_abs": round(drop, 4),
            "recall_drop_pct": round(drop / max(baseline_rate, 1e-9) * 100, 2),
        }
        print(f"  {label}: {rate:.4f} (drop {drop:.4f})")

    output = {
        "video": str(video_path),
        "total_frames": total_frames,
        "baseline_detection_rate": round(baseline_rate, 4),
        "channel_ablations": channel_results,
        "interpretation": (
            "Each row shows what happens when that channel is disabled. "
            "Large recall_drop_pct means the channel is critical."
        ),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved → {output_path}")
    return output


if __name__ == "__main__":
    result = run_ablation("input.mp4", "results/detector_ablation.json")
    print(json.dumps(result, indent=2))
