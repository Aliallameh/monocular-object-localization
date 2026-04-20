"""Evaluation helpers for bbox annotations and trajectory smoothness."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

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

    gt = _read_gt(path)
    by_frame = {int(row["frame_id"]): row for row in rows}
    ious: List[float] = []
    misses = 0
    for frame_id, gt_bbox in gt.items():
        row = by_frame.get(frame_id)
        if row is None:
            misses += 1
            continue
        ious.append(bbox_iou(tuple(float(v) for v in row["bbox"]), gt_bbox))

    arr = np.asarray(ious, dtype=np.float64)
    return {
        "enabled": True,
        "gt_file": str(path),
        "gt_frames": len(gt),
        "matched_frames": int(len(arr)),
        "missing_track_frames": int(misses),
        "mean_iou": None if len(arr) == 0 else float(np.mean(arr)),
        "median_iou": None if len(arr) == 0 else float(np.median(arr)),
        "min_iou": None if len(arr) == 0 else float(np.min(arr)),
        "iou_over_0_6_rate": None if len(arr) == 0 else float(np.mean(arr >= 0.6)),
        "passes_hidden_contract_proxy": bool(len(arr) > 0 and np.mean(arr >= 0.6) >= 0.90 and np.mean(arr) >= 0.60),
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
        writer.writerow(["frame_id", "x1", "y1", "x2", "y2", "review_status"])
        for row in samples:
            x1, y1, x2, y2 = [float(v) for v in row["bbox"]]
            writer.writerow([int(row["frame_id"]), f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}", "draft"])
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
        "affine_in_sample_rmse_m": _rmse(affine_pred, dst),
        "similarity_in_sample_rmse_m": _rmse(similarity_pred, dst),
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


def _read_gt(path: Path) -> Dict[int, Tuple[float, float, float, float]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("frames", data if isinstance(data, list) else [])
        return {
            int(item["frame_id"]): (
                float(item["x1"]),
                float(item["y1"]),
                float(item["x2"]),
                float(item["y2"]),
            )
            for item in items
        }
    out: Dict[int, Tuple[float, float, float, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("review_status", "ok")).lower() in {"skip", "ignore"}:
                continue
            out[int(row["frame_id"])] = (
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
            )
    return out


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
