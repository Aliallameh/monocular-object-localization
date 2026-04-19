"""Analyze tracker behavior on generated stress videos.

This joins `results/stress_long_manifest.csv` with a tracker output CSV and
reports condition-specific failure modes: blur, synthetic occluders, distractor
people, blue distractors, occlusion ages, coordinate jumps, and low-confidence
segments. It does not pretend to be true GT; it is a reproducible robustness
diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


CONDITIONS = [
    "target_occluders",
    "distractor_people",
    "blue_distractors",
    "gaussian_blur",
    "motion_blur",
    "protected_waypoint_window",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze stress-video tracker output")
    parser.add_argument("--manifest", default="results/stress_long_manifest.csv")
    parser.add_argument("--tracks", default="results/stress_output.csv")
    parser.add_argument("--output", default="results/stress_analysis.json")
    parser.add_argument("--events-csv", default="results/stress_events.csv")
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_csv_by_frame(Path(args.manifest), "out_frame")
    tracks = read_csv_by_frame(Path(args.tracks), "frame_id")
    common_frames = sorted(set(manifest) & set(tracks))
    if not common_frames:
        raise RuntimeError("No overlapping frames between manifest and track CSV")

    rows = [merge_rows(frame_id, manifest[frame_id], tracks[frame_id]) for frame_id in common_frames]
    report = build_report(rows, args.top_k)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_events_csv(Path(args.events_csv), report["worst_events"])
    print(f"[stress-analysis] frames={report['frames']} occluded={report['overall']['occluded_frames']}", flush=True)
    print(f"[stress-analysis] max_occlusion_age={report['overall']['max_occlusion_age']}", flush=True)
    print(f"[stress-analysis] max_frame_step_m={report['overall']['max_frame_step_m']:.3f}", flush=True)
    print(f"[stress-analysis] wrote {args.output} and {args.events_csv}", flush=True)


def read_csv_by_frame(path: Path, frame_key: str) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row[frame_key])] = row
    return out


def merge_rows(frame_id: int, manifest: Dict[str, str], track: Dict[str, str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"frame_id": frame_id}
    row.update(manifest)
    row.update(track)
    for key in CONDITIONS:
        row[key] = float(row.get(key, 0.0)) > 0.0
    row["conf_value"] = parse_float(row.get("conf"))
    row["x_world_value"] = parse_float(row.get("x_world"))
    row["y_world_value"] = parse_float(row.get("y_world"))
    row["occlusion_age_value"] = parse_int(row.get("occlusion_age"))
    row["is_occluded"] = str(row.get("track_state", "")).upper() == "OCCLUDED" or str(row.get("status", "")) == "occluded"
    row["is_reacquired"] = str(row.get("track_state", "")).upper() == "REACQUIRED"
    return row


def build_report(rows: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    jumps = compute_jumps(rows)
    occlusion_runs = compute_runs(rows, lambda r: bool(r["is_occluded"]))
    reacq_runs = compute_runs(rows, lambda r: bool(r["is_reacquired"]))
    low_conf_runs = compute_runs(rows, lambda r: parse_float(r.get("conf")) < 0.35)
    worst_events = worst_event_rows(rows, jumps, top_k)
    return {
        "frames": len(rows),
        "overall": {
            "occluded_frames": int(sum(bool(r["is_occluded"]) for r in rows)),
            "reacquired_frames": int(sum(bool(r["is_reacquired"]) for r in rows)),
            "low_confidence_frames_lt_0_35": int(sum(parse_float(r.get("conf")) < 0.35 for r in rows)),
            "max_occlusion_age": int(max((int(r["occlusion_age_value"]) for r in rows), default=0)),
            "max_frame_step_m": float(max((j["step_m"] for j in jumps), default=0.0)),
            "p95_frame_step_m": percentile([j["step_m"] for j in jumps], 95),
            "mean_confidence": mean([parse_float(r.get("conf")) for r in rows]),
            "state_counts": dict(Counter(str(r.get("track_state", "")) for r in rows)),
            "source_counts": dict(Counter(str(r.get("detector_source", "")) for r in rows)),
        },
        "by_condition": {condition: summarize_condition(rows, condition) for condition in CONDITIONS},
        "occlusion_runs": summarize_runs(occlusion_runs),
        "reacquisition_runs": summarize_runs(reacq_runs),
        "low_confidence_runs": summarize_runs(low_conf_runs),
        "worst_events": worst_events,
        "interpretation": (
            "Stress analysis is conditioned on synthetic perturbation metadata. "
            "It measures robustness symptoms, not true object-location accuracy."
        ),
    }


def summarize_condition(rows: List[Dict[str, Any]], condition: str) -> Dict[str, Any]:
    subset = [r for r in rows if bool(r[condition])]
    if not subset:
        return {"frames": 0}
    return {
        "frames": len(subset),
        "occluded_rate": float(np.mean([bool(r["is_occluded"]) for r in subset])),
        "reacquired_rate": float(np.mean([bool(r["is_reacquired"]) for r in subset])),
        "mean_confidence": mean([parse_float(r.get("conf")) for r in subset]),
        "p10_confidence": percentile([parse_float(r.get("conf")) for r in subset], 10),
        "max_occlusion_age": int(max((int(r["occlusion_age_value"]) for r in subset), default=0)),
        "state_counts": dict(Counter(str(r.get("track_state", "")) for r in subset)),
        "source_counts": dict(Counter(str(r.get("detector_source", "")) for r in subset)),
    }


def compute_jumps(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    jumps: List[Dict[str, Any]] = []
    prev = None
    for row in rows:
        x = row["x_world_value"]
        y = row["y_world_value"]
        if np.isfinite(x) and np.isfinite(y) and prev is not None:
            step = float(np.hypot(x - prev["x_world_value"], y - prev["y_world_value"]))
            jumps.append(
                {
                    "frame_id": int(row["frame_id"]),
                    "prev_frame_id": int(prev["frame_id"]),
                    "step_m": step,
                    "track_state": row.get("track_state", ""),
                    "detector_source": row.get("detector_source", ""),
                }
            )
        if np.isfinite(x) and np.isfinite(y):
            prev = row
    return jumps


def compute_runs(rows: List[Dict[str, Any]], pred) -> List[Dict[str, int]]:
    runs: List[Dict[str, int]] = []
    start = None
    last = None
    for row in rows:
        frame_id = int(row["frame_id"])
        if pred(row):
            if start is None:
                start = frame_id
            last = frame_id
        elif start is not None and last is not None:
            runs.append({"start": start, "end": last, "length": last - start + 1})
            start = None
            last = None
    if start is not None and last is not None:
        runs.append({"start": start, "end": last, "length": last - start + 1})
    return runs


def summarize_runs(runs: List[Dict[str, int]]) -> Dict[str, Any]:
    lengths = [r["length"] for r in runs]
    return {
        "count": len(runs),
        "max_length": int(max(lengths, default=0)),
        "mean_length": mean(lengths),
        "runs": runs[:20],
    }


def worst_event_rows(rows: List[Dict[str, Any]], jumps: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    by_frame = {int(r["frame_id"]): r for r in rows}
    jump_by_frame = {int(j["frame_id"]): float(j["step_m"]) for j in jumps}
    scored: List[tuple[float, Dict[str, Any]]] = []
    for row in rows:
        frame_id = int(row["frame_id"])
        conf = parse_float(row.get("conf"))
        age = int(row["occlusion_age_value"])
        step = jump_by_frame.get(frame_id, 0.0)
        condition_count = sum(1 for c in CONDITIONS if bool(row[c]))
        score = 2.5 * step + 0.08 * age + max(0.0, 0.6 - conf) + 0.25 * condition_count
        if row["is_occluded"]:
            score += 1.0
        if row["is_reacquired"]:
            score += 0.7
        if score > 0.8:
            scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    out: List[Dict[str, Any]] = []
    for score, row in scored[:top_k]:
        frame_id = int(row["frame_id"])
        out.append(
            {
                "frame_id": frame_id,
                "score": float(score),
                "step_m": float(jump_by_frame.get(frame_id, 0.0)),
                "track_state": row.get("track_state", ""),
                "status": row.get("status", ""),
                "detector_source": row.get("detector_source", ""),
                "conf": parse_float(row.get("conf")),
                "occlusion_age": int(row["occlusion_age_value"]),
                "conditions": [c for c in CONDITIONS if bool(row[c])],
                "x_world": parse_float(row.get("x_world")),
                "y_world": parse_float(row.get("y_world")),
            }
        )
    return out


def write_events_csv(path: Path, events: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["frame_id", "score", "step_m", "track_state", "status", "detector_source", "conf", "occlusion_age", "conditions", "x_world", "y_world"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for event in events:
            event = dict(event)
            event["conditions"] = "|".join(event["conditions"])
            writer.writerow(event)


def mean(values: Iterable[float]) -> float:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    return float(np.mean(arr)) if len(arr) else float("nan")


def percentile(values: Iterable[float], q: float) -> float:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    return float(np.percentile(arr, q)) if len(arr) else float("nan")


def parse_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def parse_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


if __name__ == "__main__":
    main()
