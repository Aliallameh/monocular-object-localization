"""Benchmark detector backends on sampled video frames.

This tool is intentionally separate from `track_bin.py`. It measures detector
proposal latency and source mix without invoking tracking or waypoint
calibration. It is useful for deciding whether an optional learned backend is
worth using on a given machine/video.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detector import load_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark detector backends on sampled video frames")
    parser.add_argument("--video", required=True)
    parser.add_argument("--backends", default="hybrid", help="Comma-separated: hybrid,auto,yolo_world")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--output-json", default="results/detector_benchmark.json")
    parser.add_argument("--output-csv", default="results/detector_benchmark.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = sample_video_frames(args.video, args.samples)
    reports: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    for backend in [b.strip() for b in args.backends.split(",") if b.strip()]:
        report, rows = benchmark_backend(backend, args, frames)
        reports.append(report)
        csv_rows.extend(rows)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps({"video": args.video, "reports": reports}, indent=2), encoding="utf-8")
    write_csv(Path(args.output_csv), csv_rows)
    for report in reports:
        status = "ok" if report["available"] else "unavailable"
        print(
            f"[benchmark] {report['backend']} {status}: "
            f"mean={report.get('mean_ms_per_frame')}ms p95={report.get('p95_ms_per_frame')}ms "
            f"detection_frames={report.get('frames_with_detections')}/{report.get('frames')}",
            flush=True,
        )
    print(f"[benchmark] wrote {args.output_json} and {args.output_csv}", flush=True)


def sample_video_frames(video: str, samples: int) -> List[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Video has no readable frame count: {video}")
    frame_ids = np.linspace(0, total - 1, num=min(samples, total), dtype=int)
    frames: List[tuple[int, np.ndarray]] = []
    for frame_id in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, frame = cap.read()
        if ok:
            frames.append((int(frame_id), frame))
    cap.release()
    return frames


def benchmark_backend(
    backend: str,
    args: argparse.Namespace,
    frames: List[tuple[int, np.ndarray]],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    try:
        detector = load_detector(
            use_gpu=args.device.startswith("cuda"),
            backend=backend,
            device=args.device,
            conf=args.conf,
            imgsz=args.imgsz,
        )
    except Exception as exc:
        return (
            {
                "backend": backend,
                "available": False,
                "error": repr(exc),
                "frames": len(frames),
            },
            [],
        )

    times: List[float] = []
    counts: List[int] = []
    source_counts: Counter[str] = Counter()
    rows: List[Dict[str, Any]] = []
    for frame_id, frame in frames:
        t0 = time.perf_counter()
        dets = detector.detect(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed_ms)
        counts.append(len(dets))
        for det in dets:
            source_counts[det.source] += 1
        best = max(dets, key=lambda d: d.confidence, default=None)
        rows.append(
            {
                "backend": backend,
                "frame_id": frame_id,
                "latency_ms": f"{elapsed_ms:.4f}",
                "detections": len(dets),
                "best_conf": "" if best is None else f"{best.confidence:.4f}",
                "best_source": "" if best is None else best.source,
                "best_x1": "" if best is None else f"{best.bbox[0]:.2f}",
                "best_y1": "" if best is None else f"{best.bbox[1]:.2f}",
                "best_x2": "" if best is None else f"{best.bbox[2]:.2f}",
                "best_y2": "" if best is None else f"{best.bbox[3]:.2f}",
            }
        )

    arr = np.asarray(times, dtype=np.float64)
    report = {
        "backend": backend,
        "available": True,
        "metadata": detector.metadata() if hasattr(detector, "metadata") else {"backend": detector.__class__.__name__},
        "frames": len(frames),
        "frames_with_detections": int(np.sum(np.asarray(counts) > 0)),
        "mean_ms_per_frame": float(np.mean(arr)) if len(arr) else None,
        "median_ms_per_frame": float(np.median(arr)) if len(arr) else None,
        "p95_ms_per_frame": float(np.percentile(arr, 95)) if len(arr) else None,
        "max_ms_per_frame": float(np.max(arr)) if len(arr) else None,
        "mean_detections_per_frame": float(np.mean(counts)) if counts else None,
        "source_counts": dict(source_counts),
    }
    return report, rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "backend",
        "frame_id",
        "latency_ms",
        "detections",
        "best_conf",
        "best_source",
        "best_x1",
        "best_y1",
        "best_x2",
        "best_y2",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
