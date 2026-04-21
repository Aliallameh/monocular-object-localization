"""Evaluation helpers for bbox annotations and trajectory smoothness."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from localizer import BIN_HEIGHT_M, CameraGeometry
from tracker_utils import bbox_iou


def evaluate_bbox_annotations(gt_path: str | Path | None, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate tracked boxes against an optional annotation file.

    The assessment's hidden boxes are not available locally. This function makes
    the validation path explicit: if a reviewer supplies a JSON or CSV annotation
    file, the run produces real IoU numbers; otherwise it emits a missing-GT
    report plus an annotation template.
    """

    if gt_path is None:
        return {
            "enabled": False,
            "reason": "no_gt_file_provided",
            "interpretation": "Hidden GT boxes are unavailable locally; no IoU is claimed.",
        }

    path = Path(gt_path)
    if not path.exists():
        return {
            "enabled": False,
            "reason": f"gt_file_not_found: {path}",
            "interpretation": "No IoU is claimed because the requested GT file is missing.",
        }

    gt, parse_report = _read_gt_records(path)
    by_frame = {int(row["frame_id"]): row for row in rows}
    per_frame: List[Dict[str, Any]] = []
    for rec in gt:
        frame_id = int(rec["frame_id"])
        row = by_frame.get(frame_id)
        iou = None
        if row is not None:
            iou = bbox_iou(tuple(float(v) for v in row["bbox"]), tuple(float(v) for v in rec["bbox"]))
        per_frame.append(
            {
                "frame_id": frame_id,
                "gt_bbox": [float(v) for v in rec["bbox"]],
                "occluded": rec.get("occluded"),
                "visibility": rec.get("visibility"),
                "review_status": rec.get("review_status", "ok"),
                "bbox_type": rec.get("bbox_type", ""),
                "matched": row is not None,
                "iou": iou,
                "pred_status": None if row is None else row.get("status", ""),
                "pred_track_state": None if row is None else row.get("track_state", ""),
                "pred_detector_source": None if row is None else row.get("detector_source", ""),
            }
        )

    all_stats = _subset_gt_stats(per_frame)
    visible_stats = _subset_gt_stats([r for r in per_frame if r.get("occluded") is False])
    occluded_stats = _subset_gt_stats([r for r in per_frame if r.get("occluded") is True])
    occlusion_labels_present = any(r.get("occluded") is not None for r in per_frame)
    gt_occluded_frames = int(sum(1 for r in per_frame if r.get("occluded") is True))
    gt_visible_frames = int(sum(1 for r in per_frame if r.get("occluded") is False))
    detector_sources = Counter(str(r.get("pred_detector_source", "")) for r in per_frame if r.get("matched"))
    occluded_detector_sources = Counter(
        str(r.get("pred_detector_source", "")) for r in per_frame if r.get("matched") and r.get("occluded") is True
    )
    pred_status_on_occluded = Counter(
        str(r.get("pred_status", "")) for r in per_frame if r.get("matched") and r.get("occluded") is True
    )
    iou_rate = all_stats["iou_over_0_6_rate"]
    mean_iou = all_stats["mean_iou"]
    passes_iou = bool(iou_rate is not None and mean_iou is not None and iou_rate >= 0.90 and mean_iou >= 0.60)
    occl_continuity = occluded_stats["continuity_rate"]
    passes_occl = bool(gt_occluded_frames > 0 and occl_continuity is not None and occl_continuity >= 0.90)
    return {
        "enabled": True,
        "gt_file": str(path),
        "gt_frames": len(gt),
        "gt_visible_frames": gt_visible_frames,
        "gt_occluded_frames": gt_occluded_frames,
        "occlusion_labels_present": occlusion_labels_present,
        "parse_report": parse_report,
        "matched_frames": all_stats["matched_frames"],
        "missing_track_frames": all_stats["missing_track_frames"],
        "continuity_rate": all_stats["continuity_rate"],
        "mean_iou": mean_iou,
        "median_iou": all_stats["median_iou"],
        "min_iou": all_stats["min_iou"],
        "iou_over_0_6_rate": iou_rate,
        "all_frames": all_stats,
        "visible_frames": visible_stats,
        "occluded_frames": occluded_stats,
        "detector_source_counts_on_gt": dict(detector_sources),
        "detector_source_counts_on_gt_occluded": dict(occluded_detector_sources),
        "pred_status_counts_on_gt_occluded": dict(pred_status_on_occluded),
        "worst_iou_frames": _worst_iou_frames(per_frame, n=10),
        "passes_iou_contract_on_supplied_gt": passes_iou,
        "passes_hidden_contract_proxy": passes_iou,
        "passes_occlusion_continuity_on_supplied_gt": passes_occl,
        "interpretation": _gt_interpretation(len(gt), gt_occluded_frames, passes_iou, passes_occl),
    }


def evaluate_contact_points(
    gt_path: str | Path | None,
    rows: List[Dict[str, Any]],
    camera: CameraGeometry,
) -> Dict[str, Any]:
    """Evaluate manually reviewed floor-contact pixels.

    This is not bbox GT and it is not used to correct the trajectory. It checks
    whether the detector's bbox bottom-center pixel lands on the human-reviewed
    bin/floor contact point, then compares the corresponding ground-plane
    projections in metres.
    """

    if gt_path is None:
        return {
            "enabled": False,
            "reason": "no_contact_gt_file_provided",
            "interpretation": "No manually reviewed floor-contact points were supplied.",
        }
    path = Path(gt_path)
    if not path.exists():
        return {
            "enabled": False,
            "reason": f"contact_gt_file_not_found: {path}",
            "interpretation": "No contact-point evaluation is claimed because the file is missing.",
        }

    records, parse_report = _read_contact_gt_records(path)
    by_frame = {int(row["frame_id"]): row for row in rows}
    frames: List[Dict[str, Any]] = []
    for rec in records:
        frame_id = int(rec["frame_id"])
        row = by_frame.get(frame_id)
        if row is None:
            frames.append(
                {
                    "frame_id": frame_id,
                    "matched": False,
                    "manual_contact_px": [float(rec["u"]), float(rec["v"])],
                    "occluded": rec.get("occluded"),
                }
            )
            continue

        bbox = np.asarray(row["bbox"], dtype=np.float64)
        pred_px = np.array([0.5 * (bbox[0] + bbox[2]), bbox[3]], dtype=np.float64)
        manual_px = np.array([float(rec["u"]), float(rec["v"])], dtype=np.float64)
        px_delta = pred_px - manual_px
        manual_ground = camera.ground_intersection_from_pixel(float(manual_px[0]), float(manual_px[1]))
        detected_ground = np.asarray(row["ground_world"], dtype=np.float64)
        manual_centroid = manual_ground + np.array([0.0, 0.0, BIN_HEIGHT_M * 0.5], dtype=np.float64)
        filtered_centroid = np.asarray(row["filtered_world"], dtype=np.float64)
        raw_centroid = np.asarray(row["raw_world"], dtype=np.float64)
        frames.append(
            {
                "frame_id": frame_id,
                "matched": True,
                "review_status": rec.get("review_status", "ok"),
                "occluded": rec.get("occluded"),
                "visibility": rec.get("visibility"),
                "manual_contact_px": [float(manual_px[0]), float(manual_px[1])],
                "detected_bottom_center_px": [float(pred_px[0]), float(pred_px[1])],
                "pixel_error_dx_dy": [float(px_delta[0]), float(px_delta[1])],
                "pixel_error_norm": float(np.linalg.norm(px_delta)),
                "manual_ground_world": manual_ground.tolist(),
                "detected_ground_world": detected_ground.tolist(),
                "ground_xy_error_m": float(np.linalg.norm(detected_ground[:2] - manual_ground[:2])),
                "raw_centroid_xy_error_m": float(np.linalg.norm(raw_centroid[:2] - manual_centroid[:2])),
                "filtered_centroid_xy_error_m": float(np.linalg.norm(filtered_centroid[:2] - manual_centroid[:2])),
                "detector_source": row.get("detector_source", ""),
                "track_state": row.get("track_state", ""),
                "notes": rec.get("notes", ""),
            }
        )

    matched = [f for f in frames if f.get("matched")]
    occluded = [f for f in matched if f.get("occluded") is True]
    px = np.asarray([float(f["pixel_error_norm"]) for f in matched], dtype=np.float64)
    ground = np.asarray([float(f["ground_xy_error_m"]) for f in matched], dtype=np.float64)
    filt = np.asarray([float(f["filtered_centroid_xy_error_m"]) for f in matched], dtype=np.float64)
    return {
        "enabled": True,
        "gt_file": str(path),
        "schema": "manual_floor_contact_points_v1",
        "parse_report": parse_report,
        "gt_frames": len(records),
        "matched_frames": len(matched),
        "missing_track_frames": len(records) - len(matched),
        "occluded_contact_frames": len(occluded),
        "mean_pixel_error": _safe_np_mean(px),
        "max_pixel_error": _safe_np_max(px),
        "mean_ground_xy_error_m": _safe_np_mean(ground),
        "max_ground_xy_error_m": _safe_np_max(ground),
        "mean_filtered_centroid_xy_error_m": _safe_np_mean(filt),
        "max_filtered_centroid_xy_error_m": _safe_np_max(filt),
        "frames": frames,
        "interpretation": (
            "Manual floor-contact GT is used only for validation. It is not used to "
            "calibrate, correct, or generate output coordinates."
        ),
    }


def write_annotation_template(path: str | Path, rows: List[Dict[str, Any]], max_samples: int = 36) -> str:
    """Write sampled predicted boxes as a human-review annotation starting point."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        samples: List[Dict[str, Any]] = []
    else:
        idx = np.linspace(0, len(rows) - 1, num=min(max_samples, len(rows)), dtype=int)
        samples = [rows[int(i)] for i in idx]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "x1", "y1", "x2", "y2", "review_status", "occluded", "visibility", "bbox_type"])
        for row in samples:
            x1, y1, x2, y2 = [float(v) for v in row["bbox"]]
            writer.writerow(
                [int(row["frame_id"]), f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}", "draft", "", "", "amodal_full_bin"]
            )
    return str(path)


def trajectory_smoothness_metrics(raw_world: np.ndarray, filtered_world: np.ndarray) -> Dict[str, float]:
    """Quantify high-frequency trajectory noise before/after filtering."""

    raw_xy = np.asarray(raw_world, dtype=np.float64)[:, :2]
    filt_xy = np.asarray(filtered_world, dtype=np.float64)[:, :2]
    raw_step = np.linalg.norm(np.diff(raw_xy, axis=0), axis=1)
    filt_step = np.linalg.norm(np.diff(filt_xy, axis=0), axis=1)
    raw_accel = np.linalg.norm(np.diff(raw_xy, n=2, axis=0), axis=1)
    filt_accel = np.linalg.norm(np.diff(filt_xy, n=2, axis=0), axis=1)
    return {
        "raw_frame_step_std_m": _std(raw_step),
        "filtered_frame_step_std_m": _std(filt_step),
        "frame_step_std_reduction_pct": _pct_reduction(_std(raw_step), _std(filt_step)),
        "raw_second_difference_std_m": _std(raw_accel),
        "filtered_second_difference_std_m": _std(filt_accel),
        "second_difference_std_reduction_pct": _pct_reduction(_std(raw_accel), _std(filt_accel)),
        "raw_median_frame_step_m": _median(raw_step),
        "filtered_median_frame_step_m": _median(filt_step),
    }


def asset_alignment_diagnostics(stops: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Report whether waypoint residuals look like a valid geometry check.

    This is deliberately diagnostic only. A three-point affine residual can
    interpolate three stop positions and therefore must not be used as evidence
    that the monocular localization is accurate.
    """

    if len(stops) < 3:
        return {"enabled": False, "reason": "fewer_than_three_stops"}
    src = np.asarray([stop["estimated_filtered_centroid"][:2] for stop in stops], dtype=np.float64)
    dst = np.asarray([stop["gt_world_ground"][:2] for stop in stops], dtype=np.float64)
    affine = _fit_affine(src, dst)
    affine_pred = _apply_affine(src, affine)
    similarity_pred, similarity = _fit_apply_similarity(src, dst, src)
    loo_affine = _leave_one_out(src, dst, mode="affine")
    loo_similarity = _leave_one_out(src, dst, mode="similarity")
    return {
        "enabled": True,
        "controls": [str(stop["name"]) for stop in stops],
        "affine_in_sample_residual_m": _rmse(affine_pred, dst),
        "similarity_in_sample_residual_m": _rmse(similarity_pred, dst),
        "affine_matrix_xy": affine.tolist(),
        "similarity": similarity,
        "affine_leave_one_out_errors_m": loo_affine,
        "similarity_leave_one_out_errors_m": loo_similarity,
        "interpretation": (
            "This is not a calibration path and is not applied to output coordinates. "
            "With only three controls, a full affine fit can interpolate exactly; "
            "leave-one-out errors are the useful warning about asset consistency."
        ),
    }


def _read_gt_records(path: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_items = _read_raw_gt_items(path)
    records: List[Dict[str, Any]] = []
    skipped = 0
    invalid = 0
    duplicate_frames = 0
    seen: set[int] = set()
    for item in raw_items:
        status = str(item.get("review_status", "ok")).strip().lower()
        if status in {"skip", "ignore", "bad", "draft"}:
            skipped += 1
            continue
        try:
            frame_id = int(item["frame_id"])
            bbox = (
                float(item["x1"]),
                float(item["y1"]),
                float(item["x2"]),
                float(item["y2"]),
            )
        except Exception:
            invalid += 1
            continue
        if not (np.isfinite(bbox).all() and bbox[2] > bbox[0] and bbox[3] > bbox[1]):
            invalid += 1
            continue
        if frame_id in seen:
            duplicate_frames += 1
        seen.add(frame_id)
        records.append(
            {
                "frame_id": frame_id,
                "bbox": bbox,
                "review_status": status or "ok",
                "occluded": _parse_optional_bool(item),
                "visibility": _parse_optional_float(item, ("visibility", "visible_fraction", "visibility_fraction")),
                "bbox_type": str(item.get("bbox_type", item.get("box_type", ""))),
            }
        )
    records.sort(key=lambda r: int(r["frame_id"]))
    if duplicate_frames:
        # Keep the last reviewed box per frame. Multiple target boxes are not
        # meaningful for this single-bin task unless the caller filters first.
        by_frame = {int(r["frame_id"]): r for r in records}
        records = [by_frame[fid] for fid in sorted(by_frame)]
    return records, {
        "raw_rows": len(raw_items),
        "usable_rows": len(records),
        "skipped_rows": skipped,
        "invalid_rows": invalid,
        "duplicate_frames_collapsed": duplicate_frames,
    }


def _read_raw_gt_items(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("frames", data if isinstance(data, list) else [])
        return [dict(item) for item in items]
    out: List[Dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(dict(row))
    return out


def _read_contact_gt_records(path: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("frames", data if isinstance(data, list) else [])
        raw = [dict(item) for item in items]
    else:
        with path.open("r", newline="", encoding="utf-8") as f:
            raw = [dict(row) for row in csv.DictReader(f)]

    records: List[Dict[str, Any]] = []
    skipped = 0
    invalid = 0
    for item in raw:
        status = str(item.get("review_status", "ok")).strip().lower()
        if status in {"skip", "ignore", "bad", "draft"}:
            skipped += 1
            continue
        try:
            frame_id = int(item["frame_id"])
            u = _first_float(item, ("x_contact_px", "u_contact_px", "u", "x", "pixel_u"))
            v = _first_float(item, ("y_contact_px", "v_contact_px", "v", "y", "pixel_v"))
        except Exception:
            invalid += 1
            continue
        if not (np.isfinite(u) and np.isfinite(v)):
            invalid += 1
            continue
        records.append(
            {
                "frame_id": frame_id,
                "u": float(u),
                "v": float(v),
                "review_status": status or "ok",
                "occluded": _parse_optional_bool(item),
                "visibility": _parse_optional_float(item, ("visibility", "visible_fraction", "visibility_fraction")),
                "notes": str(item.get("notes", "")),
            }
        )
    records.sort(key=lambda r: int(r["frame_id"]))
    return records, {
        "raw_rows": len(raw),
        "usable_rows": len(records),
        "skipped_rows": skipped,
        "invalid_rows": invalid,
    }


def _first_float(row: Dict[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        if key not in row:
            continue
        value = str(row.get(key, "")).strip()
        if value:
            return float(value)
    raise KeyError(keys[0])


def _subset_gt_stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    matched = [r for r in items if r.get("matched")]
    ious = np.asarray([float(r["iou"]) for r in matched if r.get("iou") is not None], dtype=np.float64)
    n = len(items)
    return {
        "gt_frames": n,
        "matched_frames": len(matched),
        "missing_track_frames": n - len(matched),
        "continuity_rate": None if n == 0 else float(len(matched) / n),
        "mean_iou": None if len(ious) == 0 else float(np.mean(ious)),
        "median_iou": None if len(ious) == 0 else float(np.median(ious)),
        "min_iou": None if len(ious) == 0 else float(np.min(ious)),
        "iou_over_0_5_rate": None if len(ious) == 0 else float(np.mean(ious >= 0.5)),
        "iou_over_0_6_rate": None if len(ious) == 0 else float(np.mean(ious >= 0.6)),
        "iou_over_0_75_rate": None if len(ious) == 0 else float(np.mean(ious >= 0.75)),
    }


def _worst_iou_frames(items: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    candidates = [r for r in items if r.get("matched") and r.get("iou") is not None]
    candidates.sort(key=lambda r: float(r["iou"]))
    return [
        {
            "frame_id": int(r["frame_id"]),
            "iou": float(r["iou"]),
            "occluded": r.get("occluded"),
            "pred_status": r.get("pred_status"),
            "pred_detector_source": r.get("pred_detector_source"),
        }
        for r in candidates[:n]
    ]


def _parse_optional_bool(row: Dict[str, Any]) -> bool | None:
    for key in ("occluded", "is_occluded", "gt_occluded", "occlusion"):
        if key not in row:
            continue
        value = str(row.get(key, "")).strip().lower()
        if value == "":
            continue
        if value in {"1", "true", "yes", "y", "occluded", "partial", "heavy"}:
            return True
        if value in {"0", "false", "no", "n", "visible", "none"}:
            return False
    visibility = _parse_optional_float(row, ("visibility", "visible_fraction", "visibility_fraction"))
    if visibility is not None:
        return bool(visibility < 0.99)
    return None


def _parse_optional_float(row: Dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in row:
            continue
        value = str(row.get(key, "")).strip()
        if value == "":
            continue
        try:
            out = float(value)
        except ValueError:
            continue
        if out > 1.0 and out <= 100.0:
            out /= 100.0
        return float(out)
    return None


def _gt_interpretation(gt_frames: int, gt_occluded_frames: int, passes_iou: bool, passes_occl: bool) -> str:
    if gt_frames == 0:
        return "No usable GT rows after filtering draft/skip/invalid annotations."
    parts = ["Evaluation uses externally supplied reviewed bbox annotations."]
    if passes_iou:
        parts.append("IoU contract passes on supplied GT.")
    else:
        parts.append("IoU contract does not pass on supplied GT or GT coverage is too weak.")
    if gt_occluded_frames == 0:
        parts.append("No GT rows are marked occluded, so real occlusion continuity is not evaluated.")
    elif passes_occl:
        parts.append("Occlusion continuity passes on supplied occluded GT frames.")
    else:
        parts.append("Occlusion continuity does not pass on supplied occluded GT frames.")
    return " ".join(parts)


def _fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    matrix, _, _, _ = np.linalg.lstsq(np.column_stack([src, np.ones(len(src))]), dst, rcond=None)
    return matrix


def _apply_affine(src: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.column_stack([src, np.ones(len(src))]) @ matrix


def _fit_apply_similarity(train_src: np.ndarray, train_dst: np.ndarray, eval_src: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
    mu_s = np.mean(train_src, axis=0)
    mu_d = np.mean(train_dst, axis=0)
    x = train_src - mu_s
    y = train_dst - mu_d
    cov = y.T @ x / max(1, len(train_src))
    u, singular, vt = np.linalg.svd(cov)
    d = np.eye(2)
    if np.linalg.det(u @ vt) < 0:
        d[-1, -1] = -1.0
    r = u @ d @ vt
    var = float(np.mean(np.sum(x * x, axis=1)))
    scale = float(np.trace(np.diag(singular) @ d) / max(var, 1e-12))
    t = mu_d - scale * (r @ mu_s)
    pred = (scale * (r @ eval_src.T)).T + t
    return pred, {"scale": scale, "rotation": r.tolist(), "translation": t.tolist()}


def _leave_one_out(src: np.ndarray, dst: np.ndarray, mode: str) -> Dict[str, float | None]:
    out: Dict[str, float | None] = {}
    for i in range(len(src)):
        train = np.asarray([j for j in range(len(src)) if j != i], dtype=int)
        if mode == "affine" and len(train) < 3:
            out[str(i)] = None
            continue
        if mode == "affine":
            pred = _apply_affine(src[i : i + 1], _fit_affine(src[train], dst[train]))[0]
        else:
            pred, _ = _fit_apply_similarity(src[train], dst[train], src[i : i + 1])
            pred = pred[0]
        out[str(i)] = float(np.linalg.norm(pred - dst[i]))
    return out


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def _std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    return float(np.std(arr)) if len(arr) else float("nan")


def _median(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    return float(np.median(arr)) if len(arr) else float("nan")


def _pct_reduction(raw: float, filtered: float) -> float:
    return float(100.0 * (1.0 - filtered / raw)) if raw > 0 else float("nan")


def _safe_np_mean(values: np.ndarray) -> float | None:
    return None if len(values) == 0 else float(np.mean(values))


def _safe_np_max(values: np.ndarray) -> float | None:
    return None if len(values) == 0 else float(np.max(values))
