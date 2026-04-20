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


def _detection_rate(
    video_path: str,
    detector: HybridBinDetector,
    record_sample_frames: bool = False,
    max_samples: int = 5,
) -> tuple[float, int, List[int]]:
    """Run detector over all frames; return (detection_rate, total_frames, sample_frames).

    sample_frames: first max_samples frame indices that produced a detection.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    hits: List[int] = []
    samples: List[int] = []
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detected = bool(detector.detect(frame))
        hits.append(1 if detected else 0)
        if detected and record_sample_frames and len(samples) < max_samples:
            samples.append(frame_id)
        frame_id += 1
    cap.release()
    return float(np.mean(hits)) if hits else 0.0, len(hits), samples


def run_ablation(
    video_path: str,
    output_path: str = "results/detector_ablation.json",
) -> Dict[str, Any]:
    """Ablate each detector channel and measure detection-rate drop.

    Two runs per channel:
    - disable_one: all channels except this one — measures redundancy of the other three
    - solo: only this channel enabled — verifies each channel can independently detect
    """

    channels = [
        ("blue_hsv",         dict(enable_blue=False),  dict(enable_dark=False, enable_edge=False, enable_motion=False)),
        ("dark_rect",        dict(enable_dark=False),  dict(enable_blue=False, enable_edge=False, enable_motion=False)),
        ("edge_shape",       dict(enable_edge=False),  dict(enable_blue=False, enable_dark=False, enable_motion=False)),
        ("motion_foreground",dict(enable_motion=False),dict(enable_blue=False, enable_dark=False, enable_edge=False)),
    ]

    print("Running baseline (all channels enabled)…")
    baseline_rate, total_frames, _ = _detection_rate(video_path, HybridBinDetector())
    print(f"  baseline detection rate: {baseline_rate:.4f} ({total_frames} frames)")

    channel_results: Dict[str, Any] = {}
    for name, disable_kwargs, solo_kwargs in channels:
        print(f"Running disable-one: without_{name}…")
        rate_disabled, _, _ = _detection_rate(video_path, HybridBinDetector(**disable_kwargs))
        drop = baseline_rate - rate_disabled

        print(f"Running solo: only_{name}…")
        rate_solo, _, solo_samples = _detection_rate(
            video_path, HybridBinDetector(**solo_kwargs),
            record_sample_frames=True,
        )

        channel_results[f"without_{name}"] = {
            "detection_rate": round(rate_disabled, 4),
            "recall_drop_abs": round(drop, 4),
            "recall_drop_pct": round(drop / max(baseline_rate, 1e-9) * 100, 2),
            "solo_detection_rate": round(rate_solo, 4),
            "solo_first_5_frames": solo_samples,
        }
        print(f"  disable-one: {rate_disabled:.4f} (drop {drop:.4f})  solo: {rate_solo:.4f}")

    output = {
        "video": str(video_path),
        "total_frames": total_frames,
        "baseline_detection_rate": round(baseline_rate, 4),
        "channel_ablations": channel_results,
        "interpretation": (
            "disable-one: all channels except this one enabled — measures fall-back coverage. "
            "solo: only this channel enabled — verifies independent detectability. "
            "solo_first_5_frames: first frame indices detected when running that channel alone."
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
