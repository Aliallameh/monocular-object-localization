"""Build an evidence-based assessment readiness report.

This script reads generated artifacts and scores only what the repository can
prove locally. It intentionally refuses to award hidden-GT bbox credit or
waypoint-accuracy credit when the corresponding evidence is missing or invalid.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    summary = read_json(Path("results/summary.json"))
    qa = read_json(Path("results/qa_report.json"))
    readiness = build_readiness(summary, qa)
    out_json = Path("results/review_readiness.json")
    out_md = Path("results/review_readiness.md")
    out_json.write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(readiness), encoding="utf-8")
    print(f"[readiness] score={readiness['overall_score']}/100 decision={readiness['decision']}", flush=True)
    print(f"[readiness] wrote {out_json} and {out_md}", flush=True)


def build_readiness(summary: Dict[str, Any], qa: Dict[str, Any]) -> Dict[str, Any]:
    sections: Dict[str, Dict[str, Any]] = {}
    sections["requirement_compliance"] = score_requirement_compliance(summary, qa)
    sections["technical_correctness"] = score_technical_correctness(summary, qa)
    sections["code_quality"] = score_code_quality()
    sections["occlusion_runtime_realism"] = score_occlusion_runtime(summary)
    sections["math_derivation_clarity"] = score_math_docs()
    sections["deployment_realism"] = score_deployment_docs()
    sections["honesty"] = score_honesty(summary, qa)

    weighted = {
        "requirement_compliance": 30,
        "technical_correctness": 25,
        "code_quality": 10,
        "occlusion_runtime_realism": 10,
        "math_derivation_clarity": 10,
        "deployment_realism": 5,
        "honesty": 10,
    }
    total = 0.0
    for name, weight in weighted.items():
        total += weight * float(sections[name]["score"]) / 10.0
    overall = int(round(total))
    blockers = collect_blockers(sections)
    decision = decision_from_score(overall, blockers)
    return {
        "overall_score": overall,
        "decision": decision,
        "strong_hire_blockers": blockers,
        "sections": sections,
        "interpretation": (
            "This is an audit report, not a marketing score. Hidden IoU and invalid waypoint RMSE "
            "do not receive full credit without independent evidence."
        ),
    }


def score_requirement_compliance(summary: Dict[str, Any], qa: Dict[str, Any]) -> Dict[str, Any]:
    checks = qa.get("contract_checks", {})
    points = {
        "output_csv_exists": bool(checks.get("output_csv_exists")),
        "required_pose_columns_present": bool(checks.get("required_pose_columns_present")),
        "bbox_columns_present": bool(checks.get("bbox_columns_present")),
        "processed_all_video_frames": bool(checks.get("processed_all_video_frames")),
        "first_stdout_under_2s": bool(checks.get("first_track_stdout_under_2s")),
        "detector_hit_rate_over_90pct": bool(checks.get("detector_hit_rate_over_90pct")),
        "tracker_output_rate_over_90pct": bool(checks.get("tracker_output_rate_over_90pct")),
        "latency_under_250ms_p95": bool(checks.get("p95_cpu_latency_under_250ms")),
        "trajectory_png_exists": Path("trajectory.png").exists(),
        "raw_vs_filtered_png_exists": Path("trajectory_raw_vs_filtered.png").exists(),
    }
    score = 10.0 * sum(points.values()) / len(points)
    return {"score": round(score, 1), "evidence": points}


def score_technical_correctness(summary: Dict[str, Any], qa: Dict[str, Any]) -> Dict[str, Any]:
    bbox_eval = summary.get("bbox_evaluation", {})
    rmse = safe_float(summary.get("metrics", {}).get("rmse_xy_m"))
    waypoint_contract = qa.get("asset_alignment", {}).get("waypoint_contract_assessment", {})
    evidence = {
        "finite_pose_stream": output_pose_stream_is_finite(Path("results/output.csv")),
        "bbox_gt_iou_available": bool(bbox_eval.get("enabled")),
        "waypoint_rmse_xy_m": rmse,
        "waypoint_contract_status": waypoint_contract.get("status"),
        "z_world_centroid_near_expected": z_world_is_near_expected(Path("results/output.csv")),
    }
    score = 5.0
    if evidence["finite_pose_stream"]:
        score += 1.5
    if evidence["z_world_centroid_near_expected"]:
        score += 1.0
    if evidence["bbox_gt_iou_available"]:
        score += 1.0
    if rmse is not None and rmse <= 0.30 and evidence["waypoint_contract_status"] == "plausible":
        score += 1.5
    elif rmse is not None and rmse <= 1.25:
        score += 0.5
    return {"score": min(10.0, round(score, 1)), "evidence": evidence}


def score_code_quality() -> Dict[str, Any]:
    evidence = {
        "run_sh_executable": os.access("run.sh", os.X_OK),
        "validator_exists": Path("tools/validate_submission.py").exists(),
        "tests_exist": Path("tests/test_geometry_and_eval.py").exists(),
        "deprecated_calibration_code_absent": not Path("scene_calibration.py").exists(),
    }
    return {"score": round(10.0 * sum(evidence.values()) / len(evidence), 1), "evidence": evidence}


def score_occlusion_runtime(summary: Dict[str, Any]) -> Dict[str, Any]:
    suite = summary.get("occlusion_stress_suite", {})
    evidence = {
        "first_stdout_ms": summary.get("first_track_stdout_ms_from_python"),
        "p95_processing_ms": summary.get("p95_processing_ms_per_frame"),
        "stress_suite_enabled": bool(suite.get("enabled")),
        "min_dropout_continuity_rate": suite.get("min_dropout_continuity_rate"),
        "min_dropout_iou_vs_baseline": suite.get("min_dropout_iou_vs_baseline"),
    }
    score = 4.0
    if safe_float(evidence["first_stdout_ms"]) is not None and float(evidence["first_stdout_ms"]) <= 2000.0:
        score += 1.5
    if safe_float(evidence["p95_processing_ms"]) is not None and float(evidence["p95_processing_ms"]) <= 250.0:
        score += 1.5
    if evidence["stress_suite_enabled"] and safe_float(evidence["min_dropout_continuity_rate"]) == 1.0:
        score += 2.0
    if safe_float(evidence["min_dropout_iou_vs_baseline"]) is not None and float(evidence["min_dropout_iou_vs_baseline"]) >= 0.55:
        score += 1.0
    return {"score": min(10.0, round(score, 1)), "evidence": evidence}


def score_math_docs() -> Dict[str, Any]:
    text = Path("README.md").read_text(encoding="utf-8") if Path("README.md").exists() else ""
    evidence = {
        "height_depth_derivation": "Z = fy * H / h_px" in text,
        "ground_intersection_derivation": "lambda = -camera_height_m / r_world_z" in text,
        "camera_world_axes": "world_X = cam_Z" in text,
        "kalman_state_documented": "[x, y, z, vx, vy, vz]" in text,
    }
    return {"score": round(10.0 * sum(evidence.values()) / len(evidence), 1), "evidence": evidence}


def score_deployment_docs() -> Dict[str, Any]:
    text = Path("README.md").read_text(encoding="utf-8") if Path("README.md").exists() else ""
    evidence = {
        "tensor_rt_fp16_int8": "TensorRT" in text and "FP16" in text and "INT8" in text,
        "moving_uav_transform": "T_world_body(t)" in text,
        "mavlink_named": "MAVLink" in text,
        "latency_budget": "Latency target" in text,
    }
    return {"score": round(10.0 * sum(evidence.values()) / len(evidence), 1), "evidence": evidence}


def score_honesty(summary: Dict[str, Any], qa: Dict[str, Any]) -> Dict[str, Any]:
    bbox_eval = summary.get("bbox_evaluation", {})
    asset = qa.get("asset_alignment", {})
    evidence = {
        "no_waypoint_calibration_fields": "scene_calibration" not in summary and "waypoint_calibrated" not in summary,
        "hidden_iou_not_claimed_without_gt": not bbox_eval.get("enabled") and "no IoU is claimed" in str(bbox_eval.get("interpretation", "")),
        "waypoint_contract_status_reported": bool(asset.get("waypoint_contract_assessment", {}).get("status")),
        "deprecated_artifacts_absent": all(
            not Path(path).exists()
            for path in [
                "results/output_waypoint_calibrated.csv",
                "results/scene_control_report.json",
                "trajectory_waypoint_calibrated.png",
            ]
        ),
    }
    return {"score": round(10.0 * sum(evidence.values()) / len(evidence), 1), "evidence": evidence}


def collect_blockers(sections: Dict[str, Dict[str, Any]]) -> List[str]:
    blockers: List[str] = []
    tech = sections["technical_correctness"]["evidence"]
    if not tech.get("bbox_gt_iou_available"):
        blockers.append("No independent bbox GT/IoU evidence is available locally.")
    if tech.get("waypoint_contract_status") != "plausible":
        blockers.append("Waypoint pixels are not geometrically valid independent stop references.")
    elif safe_float(tech.get("waypoint_rmse_xy_m")) is not None and float(tech["waypoint_rmse_xy_m"]) > 0.30:
        blockers.append("Waypoint RMSE exceeds the 0.30 m target.")
    return blockers


def decision_from_score(score: int, blockers: List[str]) -> str:
    if score >= 90 and not blockers:
        return "Strong Hire"
    if score >= 78:
        return "Hire"
    if score >= 65:
        return "Borderline"
    return "No Hire"


def render_markdown(readiness: Dict[str, Any]) -> str:
    lines = [
        "# Review Readiness",
        "",
        f"Overall score: `{readiness['overall_score']}/100`",
        f"Decision: `{readiness['decision']}`",
        "",
        "## Strong-Hire Blockers",
    ]
    blockers = readiness.get("strong_hire_blockers", [])
    if blockers:
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("- None")
    lines.extend(["", "## Section Scores"])
    for name, section in readiness["sections"].items():
        lines.append(f"- {name}: `{section['score']}/10`")
    lines.append("")
    lines.append(readiness["interpretation"])
    lines.append("")
    return "\n".join(lines)


def output_pose_stream_is_finite(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in ["x_cam", "y_cam", "z_cam", "x_world", "y_world", "z_world", "conf"]:
                value = safe_float(row.get(key))
                if value is None or abs(value) >= 1e6:
                    return False
    return True


def z_world_is_near_expected(path: Path) -> bool:
    if not path.exists():
        return False
    values: List[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            value = safe_float(row.get("z_world"))
            if value is not None:
                values.append(value)
    if not values:
        return False
    mean_z = sum(values) / len(values)
    return abs(mean_z - 0.325) <= 0.05


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


if __name__ == "__main__":
    main()
