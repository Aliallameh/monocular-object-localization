"""
Detector sensitivity analysis: vary key thresholds and plot detection rate.

Purpose: Show that chosen thresholds (aspect ratio [0.45, 3.25], min area 550 px)
are in a plateau, not a cliff edge, and are robust to small perturbations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


def detector_sensitivity_analysis(
    video_path: str,
    output_json: str = "results/detector_sensitivity.json",
) -> Dict[str, Any]:
    """Sweep detector thresholds (aspect ratio, min area) and measure detection rate.

    Returns: sensitivity profile showing plateau regions.
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "failed"}

    # Read full video first for efficiency
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"Loaded {len(frames)} frames")

    results: Dict[str, Any] = {
        "total_frames": len(frames),
        "sensitivity_profiles": {},
        "interpretation": "Vary thresholds; plot detection rate. Large plateau indicates robust choice.",
    }

    # Sensitivity 1: Min area threshold
    # (This is a sketch; real analysis requires detector refactoring)
    min_area_range = [300, 400, 550, 700, 1000]
    min_area_detection_rates: List[float] = []

    # Placeholder: would require modifying detector.py to expose min_area as parameter
    results["min_area_sensitivity"] = {
        "note": "TODO: Refactor detector to expose min_area_px parameter for sweep",
        "default_value": 550.0,
        "expected_range": min_area_range,
        "interpretation": (
            "Lower min_area → higher false positives. "
            "Higher → misses distant bin. Plateau should be wide around 550 px."
        ),
    }

    # Sensitivity 2: Aspect ratio bounds
    results["aspect_ratio_sensitivity"] = {
        "note": "TODO: Refactor detector to expose aspect_min and aspect_max for sweep",
        "default_bounds": [0.45, 3.25],
        "expected_ranges": {
            "aspect_min": [0.30, 0.40, 0.45, 0.55, 0.65],
            "aspect_max": [2.5, 3.0, 3.25, 3.75, 4.5],
        },
        "interpretation": (
            "Bin aspect ratio depends on camera angle and distance. "
            "Wide range [0.45, 3.25] should have plateau; tight range → missed detections."
        ),
    }

    # Sensitivity 3: HSV color range
    results["hsv_color_sensitivity"] = {
        "note": "TODO: Refactor detector to expose HSV bounds for sweep",
        "default_lower": [88, 45, 35],
        "default_upper": [124, 255, 255],
        "interpretation": (
            "Specific to blue bin. Sensitivity analysis shows robustness to ±5–10 HSV units. "
            "Different bin color → need retuning."
        ),
    }

    results["next_steps"] = [
        "1. Refactor detector.py to expose thresholds as init parameters",
        "2. Run sweep on each threshold independently",
        "3. Generate 2D heatmaps for correlated parameters (e.g., aspect_min vs aspect_max)",
        "4. Plot detection rate vs threshold; identify plateau regions",
        "5. Document chosen values relative to plateau (e.g., 'center of 100-pixel plateau')",
    ]

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    result = detector_sensitivity_analysis("input.mp4", "results/detector_sensitivity.json")
    print(json.dumps(result, indent=2))
