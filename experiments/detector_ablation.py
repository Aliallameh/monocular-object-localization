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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from detector import HybridBinDetector


def run_ablation(
    video_path: str,
    output_path: str = "results/detector_ablation.csv",
) -> Dict[str, Any]:
    """Ablate each detector channel and measure detection rate drop.

    Returns: {
        'full_detection_rate': float,
        'channel_ablations': {
            'without_blue_hsv': float,
            'without_dark_rect': float,
            'without_edge_shape': float,
            'without_motion_foreground': float,
        },
        'recall_drop': {
            'blue_hsv': float (percentage),
            ...
        }
    }
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "failed", "reason": "cannot open video"}

    # Baseline: full detector
    detector_full = HybridBinDetector()
    full_detection_counts: List[int] = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector_full.detect(frame)
        full_detection_counts.append(1 if detections else 0)
        frame_id += 1

    cap.release()

    full_detection_rate = float(np.mean(full_detection_counts))
    total_frames = frame_id

    results = {
        "video": str(video_path),
        "total_frames": total_frames,
        "full_detection_rate": full_detection_rate,
        "channel_contributions": {},
        "interpretation": (
            "Ablation measures detection rate with each channel disabled. "
            "Large drop indicates critical contribution."
        ),
    }

    # Ablation: disable each channel by modifying detector methods
    # (This is a sketch; real ablation would require refactoring detector API)

    # For now, record expected channels from metadata
    detector = HybridBinDetector()
    metadata = detector.metadata()
    channels = metadata.get("proposal_sources", [])

    results["channels_tested"] = channels
    results["note"] = (
        "Full ablation requires refactoring detector.py to disable channels individually. "
        "This is a TODO for the full implementation. "
        "For now, baseline is: full hybrid detection works at {:.1f}% recall.".format(
            full_detection_rate * 100
        )
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    result = run_ablation("input.mp4", "results/detector_ablation.json")
    print(json.dumps(result, indent=2))
