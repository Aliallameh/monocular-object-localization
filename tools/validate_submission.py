"""Fail-fast validation for the submitted repository artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List


REQUIRED_CSV_COLUMNS = {
    "frame_id",
    "timestamp_ms",
    "x1",
    "y1",
    "x2",
    "y2",
    "x_cam",
    "y_cam",
    "z_cam",
    "x_world",
    "y_world",
    "z_world",
    "conf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated submission artifacts")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--expect-frames", type=int, default=875)
    parser.add_argument("--max-rmse", type=float, default=None, help="Legacy cap for waypoint proxy residual")
    parser.add_argument("--enforce-rmse", action="store_true", help="Legacy gate; applies to proxy residual only")
    parser.add_argument("--max-p95-ms", type=float, default=250.0)
    parser.add_argument("--require-bbox-gt", action="store_true", help="Require supplied bbox GT evaluation in summary")
    parser.add_argument("--require-contact-gt", action="store_true", help="Require manual floor-contact point GT")
    parser.add_argument("--require-occlusion-gt", action="store_true", help="Require GT rows marked occluded")
    parser.add_argument("--min-gt-frames", type=int, default=30)
    parser.add_argument("--min-gt-occluded-frames", type=int, default=5)
    parser.add_argument("--min-gt-iou-rate", type=float, default=0.90)
    parser.add_argument("--max-contact-ground-error-m", type=float, default=None)
    parser.add_argument("--min-occlusion-continuity-rate", type=float, default=0.90)
    parser.add_argument("--allow-private-tracked", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures: List[str] = []
    results_dir = Path(args.results_dir)

    summary = read_json(results_dir / "summary.json", failures)
    manifest = read_json(results_dir / "run_manifest.json", failures)
    validate_output_csv(results_dir / "output.csv", args.expect_frames, failures)
    validate_summary(summary, args, failures)
    validate_manifest(manifest, failures)
    validate_files(failures)
    if not args.allow_private_tracked:
        validate_git_tracked_files(failures)

    if failures:
        print("[validate] FAILED")
        for failure in failures:
            print(f" - {failure}")
        raise SystemExit(1)
    print("[validate] PASS")


def read_json(path: Path, failures: List[str]) -> Dict[str, Any]:
    if not path.exists():
        failures.append(f"missing JSON file: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"invalid JSON {path}: {exc}")
        return {}


def validate_output_csv(path: Path, expect_frames: int, failures: List[str]) -> None:
    if not path.exists():
        failures.append(f"missing output CSV: {path}")
        return
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing = REQUIRED_CSV_COLUMNS - cols
        if missing:
            failures.append(f"output CSV missing required columns: {sorted(missing)}")
        rows = list(reader)
    if len(rows) != expect_frames:
        failures.append(f"output CSV row count {len(rows)} != expected {expect_frames}")
    bad_boxes = 0
    bad_pose = 0
    for row in rows:
        try:
            x1, y1, x2, y2 = [float(row[k]) for k in ("x1", "y1", "x2", "y2")]
            if not (x2 > x1 and y2 > y1):
                bad_boxes += 1
        except Exception:
            bad_boxes += 1
        try:
            vals = [float(row[k]) for k in ("x_cam", "y_cam", "z_cam", "x_world", "y_world", "z_world", "conf")]
            if any(not (abs(v) < 1e6) for v in vals):
                bad_pose += 1
        except Exception:
            bad_pose += 1
    if bad_boxes:
        failures.append(f"{bad_boxes} rows have invalid bbox values")
    if bad_pose:
        failures.append(f"{bad_pose} rows have invalid pose/conf values")


def validate_summary(summary: Dict[str, Any], args: argparse.Namespace, failures: List[str]) -> None:
    if not summary:
        return
    if int(summary.get("frames_processed", -1)) != args.expect_frames:
        failures.append(f"summary frames_processed={summary.get('frames_processed')} expected {args.expect_frames}")
    if float(summary.get("tracker_output_rate", 0.0)) < 0.90:
        failures.append(f"tracker_output_rate too low: {summary.get('tracker_output_rate')}")
    if float(summary.get("detector_hit_rate", 0.0)) < 0.90:
        failures.append(f"detector_hit_rate too low: {summary.get('detector_hit_rate')}")
    p95 = float(summary.get("p95_processing_ms_per_frame", 1e9))
    if p95 > args.max_p95_ms:
        failures.append(f"p95 latency {p95:.1f}ms > {args.max_p95_ms:.1f}ms")
    metrics = summary.get("metrics", {})
    proxy = metrics.get("waypoint_proxy_residual_xy_m")
    if proxy is None:
        failures.append("waypoint proxy residual missing from summary metrics")
    elif args.enforce_rmse:
        max_rmse = 0.35 if args.max_rmse is None else float(args.max_rmse)
        if float(proxy) > max_rmse:
            failures.append(f"waypoint proxy residual {proxy} > {max_rmse}")
    if "scene_calibration" in summary or "waypoint_calibrated" in summary:
        failures.append("summary contains deprecated waypoint-calibration output fields")
    for stale_key in ("rmse_xy_m", "rmse_3d_centroid_m", "test_rmse", "position_rmse", "gt_positions"):
        if stale_key in metrics:
            failures.append(f"summary metrics use stale/invalid key: {stale_key}")
    bbox_eval = summary.get("bbox_evaluation", {})
    if bbox_eval.get("enabled") and bbox_eval.get("iou_over_0_6_rate") is not None:
        if float(bbox_eval["iou_over_0_6_rate"]) < 0.90:
            failures.append(f"bbox IoU>0.6 rate too low: {bbox_eval['iou_over_0_6_rate']}")
    if args.require_bbox_gt:
        if not bbox_eval.get("enabled"):
            failures.append("bbox GT required but summary bbox_evaluation is not enabled")
        elif int(bbox_eval.get("gt_frames", 0)) < args.min_gt_frames:
            failures.append(
                f"bbox GT has {bbox_eval.get('gt_frames')} usable frames < required {args.min_gt_frames}"
            )
        else:
            independence = bbox_eval.get("annotation_independence_check", {})
            if independence.get("likely_draft_boxes_not_independent"):
                failures.append("bbox GT appears identical to tracker output; not independent reviewed GT")
            rate = bbox_eval.get("iou_over_0_6_rate")
            if rate is None or float(rate) < args.min_gt_iou_rate:
                failures.append(f"bbox GT IoU>0.6 rate {rate} < {args.min_gt_iou_rate}")
    if args.require_occlusion_gt:
        if not bbox_eval.get("enabled"):
            failures.append("occlusion GT required but bbox_evaluation is not enabled")
        elif int(bbox_eval.get("gt_occluded_frames", 0)) < args.min_gt_occluded_frames:
            failures.append(
                f"occlusion GT has {bbox_eval.get('gt_occluded_frames')} frames < required {args.min_gt_occluded_frames}"
            )
        else:
            occluded = bbox_eval.get("occluded_frames", {})
            continuity = occluded.get("continuity_rate")
            if continuity is None or float(continuity) < args.min_occlusion_continuity_rate:
                failures.append(
                    f"occlusion GT continuity {continuity} < {args.min_occlusion_continuity_rate}"
                )
    contact_eval = summary.get("contact_point_evaluation", {})
    if args.require_contact_gt:
        if not contact_eval.get("enabled"):
            failures.append("contact GT required but summary contact_point_evaluation is not enabled")
        elif int(contact_eval.get("matched_frames", 0)) <= 0:
            failures.append("contact GT has no matched tracker frames")
        elif args.max_contact_ground_error_m is not None:
            err = contact_eval.get("mean_ground_xy_error_m")
            if err is None or float(err) > float(args.max_contact_ground_error_m):
                failures.append(
                    f"mean contact ground error {err} > {args.max_contact_ground_error_m}"
                )


def validate_files(failures: List[str]) -> None:
    required = [
        "README.md",
        "run.sh",
        "track_bin.py",
        "detector.py",
        "localizer.py",
        "requirements.txt",
        "results/output.csv",
        "results/run_manifest.json",
        "results/occlusion_stress_suite.json",
        "results/contact_point_eval.json",
        "results/observer_events.json",
        "trajectory.png",
        "trajectory_raw_vs_filtered.png",
        "results/observer_overlay.mp4",
    ]
    for path in required:
        if not Path(path).exists():
            failures.append(f"missing required file: {path}")
    forbidden = [
        "results/output_waypoint_calibrated.csv",
        "results/output_scene_control.csv",
        "results/scene_control_report.json",
        "trajectory_waypoint_calibrated.png",
        "trajectory_scene_control.png",
        "trajectory_strict.png",
        "trajectory_stress_strict.png",
        "results/localization_error_budget.json",
        "results/centroid_validation.json",
    ]
    for path in forbidden:
        if Path(path).exists():
            failures.append(f"deprecated calibration artifact exists: {path}")


def validate_manifest(manifest: Dict[str, Any], failures: List[str]) -> None:
    if not manifest:
        return
    for key in ["git", "python", "libraries", "inputs", "outputs", "run_metrics"]:
        if key not in manifest:
            failures.append(f"run manifest missing key: {key}")
    output_csv = manifest.get("outputs", {}).get("output_csv", {})
    if not output_csv.get("exists"):
        failures.append("run manifest does not record existing output CSV")
    if output_csv.get("sha256") and len(str(output_csv["sha256"])) != 64:
        failures.append("run manifest output CSV sha256 is malformed")
    if "trajectory_strict_png" in manifest.get("outputs", {}):
        failures.append("run manifest still records removed trajectory_strict.png artifact")
    run_metrics = manifest.get("run_metrics", {})
    for stale_key in ("rmse_xy_m", "strict_rmse_xy_m", "test_rmse", "position_rmse"):
        if stale_key in run_metrics:
            failures.append(f"run manifest uses stale/invalid metric key: {stale_key}")


def validate_git_tracked_files(failures: List[str]) -> None:
    try:
        proc = subprocess.run(["git", "ls-files"], check=True, text=True, capture_output=True)
    except Exception:
        return
    forbidden_names = {"input.mp4", "calib.json", "Input sample.mp4", "stress_long_harsh.mp4"}
    forbidden_prefixes = (".venv/", "ProjectA/", "projectA/", "__pycache__/")
    forbidden_exact = {
        "scene_calibration.py",
        "results/output_waypoint_calibrated.csv",
        "results/output_scene_control.csv",
        "results/scene_control_report.json",
        "trajectory_waypoint_calibrated.png",
        "trajectory_scene_control.png",
        "trajectory_strict.png",
        "trajectory_stress_strict.png",
        "results/localization_error_budget.json",
        "results/centroid_validation.json",
    }
    for name in proc.stdout.splitlines():
        base = os.path.basename(name)
        if name in forbidden_exact and Path(name).exists():
            failures.append(f"deprecated calibration artifact/code is tracked: {name}")
        if base in forbidden_names or name.startswith(forbidden_prefixes):
            failures.append(f"private/generated asset is tracked: {name}")
        if base.endswith((".pyc", ".pt")):
            failures.append(f"binary/cache/model file is tracked: {name}")


if __name__ == "__main__":
    main()
