"""QA artifacts for the bin-tracking assessment.

These helpers intentionally separate submission outputs from review evidence.
The main CSV stays close to the requested contract, while diagnostics and
annotated frames make geometry failures or asset mismatches inspectable.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np

from localizer import BIN_HEIGHT_M, CameraGeometry


REQUIRED_OUTPUT_COLUMNS = [
    "frame_id",
    "timestamp_ms",
    "x_cam",
    "y_cam",
    "z_cam",
    "x_world",
    "y_world",
    "z_world",
    "conf",
]

BBOX_COLUMNS = ["x1", "y1", "x2", "y2"]


def write_diagnostics_csv(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write high-detail per-frame measurements for review/debugging."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame_id",
        "timestamp_ms",
        "status",
        "x1",
        "y1",
        "x2",
        "y2",
        "bbox_width_px",
        "bbox_height_px",
        "conf",
        "detector_source",
        "raw_x_cam",
        "raw_y_cam",
        "raw_z_cam",
        "filtered_x_cam",
        "filtered_y_cam",
        "filtered_z_cam",
        "raw_x_world",
        "raw_y_world",
        "raw_z_world",
        "filtered_x_world",
        "filtered_y_world",
        "filtered_z_world",
        "height_x_cam",
        "height_y_cam",
        "height_z_cam",
        "ground_x_world",
        "ground_y_world",
        "ground_z_world",
        "strict_raw_x_world",
        "strict_raw_y_world",
        "strict_raw_z_world",
        "strict_filtered_x_world",
        "strict_filtered_y_world",
        "strict_filtered_z_world",
        "height_ground_depth_delta_m",
        "used_height_fallback",
        "blur_laplacian_var",
        "brightness_mean",
        "contrast_std",
        "is_blurry",
        "is_low_light",
        "flow_points",
        "flow_quality",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            bbox = np.asarray(row["bbox"], dtype=np.float64)
            raw_cam = np.asarray(row["raw_cam"], dtype=np.float64)
            filt_cam = np.asarray(row["filtered_cam"], dtype=np.float64)
            raw_world = np.asarray(row["raw_world"], dtype=np.float64)
            filt_world = np.asarray(row["filtered_world"], dtype=np.float64)
            strict_raw_world = np.asarray(row.get("strict_raw_world", row["raw_world"]), dtype=np.float64)
            strict_filt_world = np.asarray(row.get("strict_filtered_world", row["filtered_world"]), dtype=np.float64)
            height_cam = np.asarray(row["height_cam"], dtype=np.float64)
            ground_world = np.asarray(row["ground_world"], dtype=np.float64)
            writer.writerow(
                {
                    "frame_id": int(row["frame_id"]),
                    "timestamp_ms": int(row["timestamp_ms"]),
                    "status": row["status"],
                    "x1": f"{bbox[0]:.2f}",
                    "y1": f"{bbox[1]:.2f}",
                    "x2": f"{bbox[2]:.2f}",
                    "y2": f"{bbox[3]:.2f}",
                    "bbox_width_px": f"{bbox[2] - bbox[0]:.2f}",
                    "bbox_height_px": f"{bbox[3] - bbox[1]:.2f}",
                    "conf": f"{float(row['conf']):.3f}",
                    "detector_source": row.get("detector_source", ""),
                    "raw_x_cam": f"{raw_cam[0]:.4f}",
                    "raw_y_cam": f"{raw_cam[1]:.4f}",
                    "raw_z_cam": f"{raw_cam[2]:.4f}",
                    "filtered_x_cam": f"{filt_cam[0]:.4f}",
                    "filtered_y_cam": f"{filt_cam[1]:.4f}",
                    "filtered_z_cam": f"{filt_cam[2]:.4f}",
                    "raw_x_world": f"{raw_world[0]:.4f}",
                    "raw_y_world": f"{raw_world[1]:.4f}",
                    "raw_z_world": f"{raw_world[2]:.4f}",
                    "filtered_x_world": f"{filt_world[0]:.4f}",
                    "filtered_y_world": f"{filt_world[1]:.4f}",
                    "filtered_z_world": f"{filt_world[2]:.4f}",
                    "height_x_cam": f"{height_cam[0]:.4f}",
                    "height_y_cam": f"{height_cam[1]:.4f}",
                    "height_z_cam": f"{height_cam[2]:.4f}",
                    "ground_x_world": f"{ground_world[0]:.4f}",
                    "ground_y_world": f"{ground_world[1]:.4f}",
                    "ground_z_world": f"{ground_world[2]:.4f}",
                    "strict_raw_x_world": f"{strict_raw_world[0]:.4f}",
                    "strict_raw_y_world": f"{strict_raw_world[1]:.4f}",
                    "strict_raw_z_world": f"{strict_raw_world[2]:.4f}",
                    "strict_filtered_x_world": f"{strict_filt_world[0]:.4f}",
                    "strict_filtered_y_world": f"{strict_filt_world[1]:.4f}",
                    "strict_filtered_z_world": f"{strict_filt_world[2]:.4f}",
                    "height_ground_depth_delta_m": f"{float(row['height_depth_delta_m']):.4f}",
                    "used_height_fallback": int(bool(row["fallback"])),
                    "blur_laplacian_var": f"{float(row.get('blur_laplacian_var', float('nan'))):.4f}",
                    "brightness_mean": f"{float(row.get('brightness_mean', float('nan'))):.4f}",
                    "contrast_std": f"{float(row.get('contrast_std', float('nan'))):.4f}",
                    "is_blurry": int(bool(row.get("is_blurry", False))),
                    "is_low_light": int(bool(row.get("is_low_light", False))),
                    "flow_points": int(row.get("flow_points", 0)),
                    "flow_quality": f"{float(row.get('flow_quality', 0.0)):.4f}",
                }
            )


def build_qa_report(
    video_path: str | Path,
    output_csv: str | Path,
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    waypoint_data: Dict[str, Any],
    projected_waypoints: Dict[str, Dict[str, Any]],
    camera: CameraGeometry,
    first_track_stdout_ms: float | None,
) -> Dict[str, Any]:
    """Build a machine-readable QA report for the latest run."""

    output_csv = Path(output_csv)
    csv_header = _read_csv_header(output_csv)
    frame_count_video = _video_frame_count(video_path)
    bbox_valid = _bbox_valid_count(rows)
    deltas = np.asarray([float(r["height_depth_delta_m"]) for r in rows], dtype=np.float64)
    confs = np.asarray([float(r["conf"]) for r in rows], dtype=np.float64)

    contract_checks = {
        "output_csv_exists": output_csv.exists(),
        "required_pose_columns_present": all(col in csv_header for col in REQUIRED_OUTPUT_COLUMNS),
        "bbox_columns_present": all(col in csv_header for col in BBOX_COLUMNS),
        "status_column_present": "status" in csv_header,
        "tracked_rows_with_valid_bbox": int(bbox_valid),
        "tracked_rows": int(len(rows)),
        "all_tracked_rows_have_valid_bbox": bool(bbox_valid == len(rows) and len(rows) > 0),
        "video_frame_count": int(frame_count_video),
        "processed_frame_count": int(summary.get("frames_processed", 0)),
        "processed_all_video_frames": bool(frame_count_video <= 0 or frame_count_video == summary.get("frames_processed")),
        "first_track_stdout_ms_from_python": None if first_track_stdout_ms is None else float(first_track_stdout_ms),
        "first_track_stdout_under_2s": bool(
            first_track_stdout_ms is not None and first_track_stdout_ms <= 2000.0
        ),
        "detector_hit_rate_over_90pct": bool(float(summary.get("detector_hit_rate", 0.0)) >= 0.90),
        "tracker_output_rate_over_90pct": bool(float(summary.get("tracker_output_rate", 0.0)) >= 0.90),
        "mean_cpu_latency_under_250ms": bool(float(summary.get("mean_processing_ms_per_frame", 1e9)) <= 250.0),
        "p95_cpu_latency_under_250ms": bool(float(summary.get("p95_processing_ms_per_frame", 1e9)) <= 250.0),
    }

    detector_checks = {
        "mean_confidence": _safe_float(np.mean(confs)) if len(confs) else None,
        "min_confidence": _safe_float(np.min(confs)) if len(confs) else None,
        "max_confidence": _safe_float(np.max(confs)) if len(confs) else None,
        "status_counts": _status_counts(rows),
        "detector_sources": _source_counts(rows),
        "flow_assisted_frames": int(summary.get("flow_assisted_frames", 0)),
    }

    blur_values = np.asarray([float(r.get("blur_laplacian_var", np.nan)) for r in rows], dtype=np.float64)
    quality_checks = {
        "blurry_frames": int(summary.get("blurry_frames", 0)),
        "blur_laplacian_var_median": _safe_float(np.nanmedian(blur_values)) if len(blur_values) else None,
        "blur_laplacian_var_min": _safe_float(np.nanmin(blur_values)) if len(blur_values) else None,
    }

    geometry_checks = {
        "height_vs_ground_depth_delta_median_m": _safe_float(np.median(deltas)) if len(deltas) else None,
        "height_vs_ground_depth_delta_p95_m": _safe_float(np.percentile(deltas, 95)) if len(deltas) else None,
        "height_vs_ground_depth_delta_max_m": _safe_float(np.max(deltas)) if len(deltas) else None,
        "centroid_z_expected_m": BIN_HEIGHT_M * 0.5,
        "centroid_z_min_max_m": _z_min_max(rows),
    }

    waypoint_checks = _waypoint_consistency_checks(rows, projected_waypoints, camera)
    pixel_probes = _probe_waypoint_pixels(video_path, waypoint_data)

    rmse_xy = float(summary.get("metrics", {}).get("rmse_xy_m", float("nan")))
    strict_rmse_xy = float(summary.get("strict_metrics", {}).get("rmse_xy_m", float("nan")))
    scene_enabled = bool(summary.get("scene_calibration", {}).get("enabled", False))
    calibrated_rmse = summary.get("waypoint_calibrated", {}).get("metrics", {}).get("rmse_xy_m")
    asset_alignment = {
        "canonical_input_used": Path(video_path).name == "input.mp4",
        "waypoint_rmse_xy_m": rmse_xy,
        "strict_waypoint_rmse_xy_m": strict_rmse_xy,
        "scene_calibration_enabled": scene_enabled,
        "waypoint_calibrated_rmse_xy_m": calibrated_rmse,
        "waypoint_rmse_status": "fail" if np.isfinite(rmse_xy) and rmse_xy > 1.0 else "pass",
        "interpretation": _asset_alignment_interpretation(scene_enabled),
    }

    return {
        "schema_version": 1,
        "video": str(video_path),
        "output_csv": str(output_csv),
        "contract_checks": contract_checks,
        "detector_checks": detector_checks,
        "quality_checks": quality_checks,
        "geometry_checks": geometry_checks,
        "waypoint_pixel_probes": pixel_probes,
        "waypoint_consistency_checks": waypoint_checks,
        "asset_alignment": asset_alignment,
        "stop_metrics": summary.get("metrics", {}),
        "bbox_evaluation": summary.get("bbox_evaluation", {}),
        "bbox_annotation_template": summary.get("bbox_annotation_template"),
        "scene_control_diagnostics": summary.get("scene_control_diagnostics", {}),
        "waypoint_calibrated": summary.get("waypoint_calibrated", {}),
        "robustness_stress_test": summary.get("robustness_stress_test", {}),
        "stops": summary.get("stops", []),
    }


def save_qa_frames(
    out_dir: str | Path,
    video_path: str | Path,
    rows: List[Dict[str, Any]],
    projected_waypoints: Dict[str, Dict[str, Any]],
    stops: List[Dict[str, Any]],
) -> List[str]:
    """Save annotated frames around waypoint priors and worst consistency gaps."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row_by_frame = {int(r["frame_id"]): r for r in rows}
    frames = {int(wp["approx_frame"]): f"waypoint_{name}" for name, wp in projected_waypoints.items()}
    if rows:
        worst = max(rows, key=lambda r: float(r["height_depth_delta_m"]))
        frames[int(worst["frame_id"])] = "largest_height_ground_delta"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    stop_by_name = {str(s["name"]): s for s in stops}
    saved: List[str] = []
    for frame_id, label in sorted(frames.items()):
        row = _nearest_row(row_by_frame, frame_id)
        if row is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_id"]))
        ok, frame = cap.read()
        if not ok:
            continue
        annotated = _annotate_frame(frame, row, projected_waypoints, stop_by_name, label)
        path = out_dir / f"{label}_frame_{int(row['frame_id']):04d}.jpg"
        cv2.imwrite(str(path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        saved.append(str(path))

    cap.release()
    return saved


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, indent=2)


def _annotate_frame(
    frame: np.ndarray,
    row: Dict[str, Any],
    waypoints: Dict[str, Dict[str, Any]],
    stops: Dict[str, Dict[str, Any]],
    label: str,
) -> np.ndarray:
    out = frame.copy()
    bbox = np.asarray(row["bbox"], dtype=np.float64)
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(out, (x1, y1), (x2, y2), (50, 230, 70), 3)
    bottom = (int(round(0.5 * (bbox[0] + bbox[2]))), int(round(bbox[3])))
    cv2.drawMarker(out, bottom, (255, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=28, thickness=3)

    for name, wp in waypoints.items():
        px = (int(round(float(wp["pixel_u"]))), int(round(float(wp["pixel_v"]))))
        color = _bgr_for_marker(str(wp["color"]))
        cv2.drawMarker(out, px, color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=38, thickness=4)
        _put_text(out, f"{name} wp px", (px[0] + 12, px[1] - 12), color)

    raw = np.asarray(row["raw_world"], dtype=np.float64)
    filt = np.asarray(row["filtered_world"], dtype=np.float64)
    lines = [
        f"{label} frame {int(row['frame_id'])}",
        f"bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}) conf={float(row['conf']):.2f}",
        f"contact/centroid raw=({raw[0]:.2f},{raw[1]:.2f},{raw[2]:.2f}) m",
        f"filtered=({filt[0]:.2f},{filt[1]:.2f},{filt[2]:.2f}) m",
        f"height-ground delta={float(row['height_depth_delta_m']):.2f} m",
    ]
    if label.startswith("waypoint_"):
        marker = label.split("_", 1)[1]
        stop = stops.get(marker)
        wp = waypoints.get(marker)
        if stop and wp:
            lines.append(f"{marker} projected wp=({wp['world_ground'][0]:.2f},{wp['world_ground'][1]:.2f}) m")
            lines.append(f"{marker} stop error={float(stop['error_xy_m']):.2f} m")

    y = 34
    for line in lines:
        _put_text(out, line, (24, y), (255, 255, 255), scale=0.72)
        y += 32
    _put_text(out, "magenta cross: detected bottom contact; colored crosses: waypoint pixels", (24, out.shape[0] - 28), (255, 255, 255), scale=0.68)
    return out


def _put_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.7,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    x = max(0, min(x, image.shape[1] - w - 4))
    y = max(h + 4, min(y, image.shape[0] - baseline - 4))
    cv2.rectangle(image, (x - 4, y - h - 5), (x + w + 4, y + baseline + 4), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _waypoint_consistency_checks(
    rows: List[Dict[str, Any]],
    waypoints: Dict[str, Dict[str, Any]],
    camera: CameraGeometry,
) -> List[Dict[str, Any]]:
    row_by_frame = {int(r["frame_id"]): r for r in rows}
    checks: List[Dict[str, Any]] = []
    fy = float(camera.K[1, 1])

    for name, wp in waypoints.items():
        approx = int(wp["approx_frame"])
        row = _nearest_row(row_by_frame, approx)
        if row is None:
            continue

        bbox = np.asarray(row["bbox"], dtype=np.float64)
        bottom_u = 0.5 * (bbox[0] + bbox[2])
        bottom_v = bbox[3]
        waypoint_pixel = np.array([float(wp["pixel_u"]), float(wp["pixel_v"])], dtype=np.float64)
        bottom_pixel = np.array([bottom_u, bottom_v], dtype=np.float64)
        pixel_gap = float(np.linalg.norm(waypoint_pixel - bottom_pixel))

        actual_contact = np.asarray(row["ground_world"], dtype=np.float64)
        waypoint_ground = np.asarray(wp["world_ground"], dtype=np.float64)
        ground_gap = float(np.linalg.norm(actual_contact[:2] - waypoint_ground[:2]))

        uc = 0.5 * (bbox[0] + bbox[2])
        top = camera.undistort_ideal_pixel(uc, bbox[1])
        bottom = camera.undistort_ideal_pixel(uc, bbox[3])
        h_px_undist = max(1.0, float(abs(bottom[1] - top[1])))
        waypoint_cam = camera.world_to_cam(np.asarray(wp["world_centroid"], dtype=np.float64))
        required_height = float(h_px_undist * waypoint_cam[2] / fy) if waypoint_cam[2] > 0 else float("nan")

        checks.append(
            {
                "name": name,
                "approx_frame": approx,
                "actual_frame_used": int(row["frame_id"]),
                "waypoint_pixel_uv": [float(waypoint_pixel[0]), float(waypoint_pixel[1])],
                "detected_bottom_center_uv": [float(bottom_pixel[0]), float(bottom_pixel[1])],
                "pixel_gap_to_detected_bottom_center_px": pixel_gap,
                "detected_ground_xy_m": [float(actual_contact[0]), float(actual_contact[1])],
                "projected_waypoint_ground_xy_m": [float(waypoint_ground[0]), float(waypoint_ground[1])],
                "ground_gap_m": ground_gap,
                "detected_bbox_height_px_undistorted": h_px_undist,
                "provided_bin_height_m": BIN_HEIGHT_M,
                "required_bin_height_if_at_waypoint_depth_m": required_height,
                "height_scale_error_factor": required_height / BIN_HEIGHT_M if np.isfinite(required_height) else None,
                "height_ground_depth_delta_m": float(row["height_depth_delta_m"]),
            }
        )
    return checks


def _probe_waypoint_pixels(video_path: str | Path, waypoint_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    probes: List[Dict[str, Any]] = []
    for marker in waypoint_data.get("markers", []):
        frame_id = int(marker["approx_frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        u, v = int(round(float(marker["pixel_u"]))), int(round(float(marker["pixel_v"])))
        patch = _safe_patch(frame, u, v, radius=7)
        bgr = np.mean(patch.reshape(-1, 3), axis=0)
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hsv = np.mean(hsv_patch.reshape(-1, 3), axis=0)
        probes.append(
            {
                "name": str(marker["name"]),
                "expected_marker_color": str(marker.get("color", "")),
                "frame": frame_id,
                "pixel_uv": [u, v],
                "mean_bgr": [float(v) for v in bgr],
                "mean_hsv": [float(v) for v in hsv],
                "observed_color_class": _classify_hsv(float(hsv[0]), float(hsv[1]), float(hsv[2])),
            }
        )
    cap.release()
    return probes


def _classify_hsv(h: float, s: float, v: float) -> str:
    if v < 45:
        return "dark/low-light"
    if s < 35:
        return "neutral/gray"
    if 88 <= h <= 124:
        return "blue"
    if 35 <= h <= 85:
        return "green"
    if 6 <= h <= 28:
        return "orange/yellow"
    if h <= 5 or h >= 170:
        return "red"
    return "other"


def _safe_patch(frame: np.ndarray, u: int, v: int, radius: int) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, x2 = max(0, u - radius), min(w, u + radius + 1)
    y1, y2 = max(0, v - radius), min(h, v + radius + 1)
    return frame[y1:y2, x1:x2]


def _nearest_row(row_by_frame: Dict[int, Dict[str, Any]], target: int) -> Dict[str, Any] | None:
    if not row_by_frame:
        return None
    frame_id = min(row_by_frame, key=lambda fid: abs(fid - target))
    return row_by_frame[frame_id]


def _read_csv_header(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def _video_frame_count(video_path: str | Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return -1
    count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return count


def _bbox_valid_count(rows: List[Dict[str, Any]]) -> int:
    valid = 0
    for row in rows:
        bbox = np.asarray(row.get("bbox", []), dtype=np.float64)
        if bbox.shape == (4,) and np.isfinite(bbox).all() and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            valid += 1
    return valid


def _status_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = str(row.get("status", ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _source_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = str(row.get("detector_source", "predicted_or_unknown"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _z_min_max(rows: List[Dict[str, Any]]) -> List[float] | None:
    if not rows:
        return None
    z = np.asarray([float(np.asarray(r["filtered_world"], dtype=np.float64)[2]) for r in rows], dtype=np.float64)
    return [float(np.min(z)), float(np.max(z))]


def _bgr_for_marker(name: str) -> tuple[int, int, int]:
    return {
        "green": (40, 220, 60),
        "orange": (0, 165, 255),
        "red": (40, 40, 230),
    }.get(name.lower(), (255, 255, 255))


def _asset_alignment_interpretation(scene_enabled: bool) -> str:
    if scene_enabled:
        return (
            "Scene-control affine calibration is enabled for this run; strict values remain in diagnostics "
            "for audit. Do not compare calibrated waypoint RMSE as an independent geometry validation."
        )
    return (
        "Default run uses strict camera/bin geometry. Waypoint residual is reported as an external consistency "
        "check only and is not used to correct the answer stream."
    )


def _safe_float(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _safe_float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
