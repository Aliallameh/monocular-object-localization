"""
Detector sensitivity analysis: vary key thresholds and measure detection-rate plateau.

Purpose: Show that chosen thresholds (aspect_min=0.45, aspect_max=3.25,
min_area_px=550) sit in a stable plateau — not on a cliff edge — so a
±10–15% perturbation does not collapse recall.
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


def _detection_rate(frames: List[np.ndarray], detector: HybridBinDetector) -> float:
    hits = [1 if detector.detect(f) else 0 for f in frames]
    return float(np.mean(hits)) if hits else 0.0


def run_sensitivity(
    video_path: str,
    output_path: str = "results/detector_sensitivity.json",
) -> Dict[str, Any]:
    """Sweep min_area_px, aspect_min, aspect_max independently; record detection rate."""

    print("Loading frames…")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"  loaded {len(frames)} frames")

    baseline_rate = _detection_rate(frames, HybridBinDetector())
    print(f"  baseline detection rate: {baseline_rate:.4f}")

    # --- Sweep 1: min_area_px ---
    area_sweep_values = [300, 400, 550, 700, 900, 1200]
    area_results = []
    for v in area_sweep_values:
        rate = _detection_rate(frames, HybridBinDetector(min_area_px=float(v)))
        area_results.append({"min_area_px": v, "detection_rate": round(rate, 4)})
        print(f"  min_area_px={v} → {rate:.4f}")

    # --- Sweep 2: aspect_min ---
    asp_min_values = [0.25, 0.35, 0.45, 0.55, 0.65, 0.80]
    asp_min_results = []
    for v in asp_min_values:
        rate = _detection_rate(frames, HybridBinDetector(aspect_min=v))
        asp_min_results.append({"aspect_min": v, "detection_rate": round(rate, 4)})
        print(f"  aspect_min={v} → {rate:.4f}")

    # --- Sweep 3: aspect_max ---
    asp_max_values = [2.0, 2.5, 3.0, 3.25, 3.75, 4.5]
    asp_max_results = []
    for v in asp_max_values:
        rate = _detection_rate(frames, HybridBinDetector(aspect_max=v))
        asp_max_results.append({"aspect_max": v, "detection_rate": round(rate, 4)})
        print(f"  aspect_max={v} → {rate:.4f}")

    # Identify plateau regions (contiguous range where rate >= baseline - 0.02)
    def plateau_range(sweep: List[Dict], key: str, tol: float = 0.02) -> Dict[str, Any]:
        in_plateau = [(r[key], r["detection_rate"]) for r in sweep
                      if r["detection_rate"] >= baseline_rate - tol]
        if not in_plateau:
            return {"plateau_start": None, "plateau_end": None, "plateau_size": 0}
        vals = [p[0] for p in in_plateau]
        return {
            "plateau_start": min(vals),
            "plateau_end": max(vals),
            "plateau_size": len(in_plateau),
            "tolerance_used": tol,
        }

    output = {
        "video": str(video_path),
        "total_frames": len(frames),
        "baseline_detection_rate": round(baseline_rate, 4),
        "defaults": {"min_area_px": 550, "aspect_min": 0.45, "aspect_max": 3.25},
        "min_area_sensitivity": {
            "sweep": area_results,
            "plateau": plateau_range(area_results, "min_area_px"),
        },
        "aspect_min_sensitivity": {
            "sweep": asp_min_results,
            "plateau": plateau_range(asp_min_results, "aspect_min"),
        },
        "aspect_max_sensitivity": {
            "sweep": asp_max_results,
            "plateau": plateau_range(asp_max_results, "aspect_max"),
        },
        "interpretation": (
            "Default thresholds are well-chosen if they sit inside the plateau "
            "(detection rate within 2% of baseline across a wide parameter range). "
            "Cliff-edge behaviour would show detection_rate collapsing for small perturbations."
        ),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved → {output_path}")
    return output


if __name__ == "__main__":
    result = run_sensitivity("input.mp4", "results/detector_sensitivity.json")
    print(json.dumps(result, indent=2))
