"""Plotting and stop-analysis helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from localizer import BIN_HEIGHT_M, CameraGeometry


def project_waypoints(waypoint_data: Dict[str, Any], camera: CameraGeometry) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for marker in waypoint_data.get("markers", []):
        p = camera.ground_intersection_from_pixel(float(marker["pixel_u"]), float(marker["pixel_v"]))
        out[str(marker["name"])] = {
            "name": str(marker["name"]),
            "color": str(marker.get("color", "black")),
            "approx_frame": int(marker["approx_frame"]),
            "pixel_u": float(marker["pixel_u"]),
            "pixel_v": float(marker["pixel_v"]),
            "world_ground": p,
            "world_centroid": p + np.array([0.0, 0.0, BIN_HEIGHT_M * 0.5], dtype=np.float64),
        }
    return out


def estimate_stops(
    frame_ids: np.ndarray,
    raw_world: np.ndarray,
    filtered_world: np.ndarray,
    waypoints: Dict[str, Dict[str, Any]],
    fps: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    stops: List[Dict[str, Any]] = []
    raw_std_values: List[float] = []
    filt_std_values: List[float] = []
    errors_xy: List[float] = []
    errors_3d: List[float] = []

    if len(frame_ids) < 2:
        return stops, {"rmse_xy_m": float("nan"), "rmse_3d_centroid_m": float("nan")}

    vel = np.zeros(len(frame_ids), dtype=np.float64)
    diffs = np.linalg.norm(np.diff(raw_world[:, :2], axis=0), axis=1)
    dt = 1.0 / max(1e-6, fps)
    vel[1:] = diffs / dt

    for name, waypoint in waypoints.items():
        approx = int(waypoint["approx_frame"])
        in_window = np.where(np.abs(frame_ids - approx) <= 25)[0]
        if len(in_window) == 0:
            nearest = int(np.argmin(np.abs(frame_ids - approx)))
            in_window = np.arange(max(0, nearest - 15), min(len(frame_ids), nearest + 16))
        if len(in_window) == 0:
            continue

        win_vel = vel[in_window]
        threshold = min(0.12, max(0.015, float(np.quantile(win_vel, 0.55)))) if len(win_vel) else 0.12
        stationary = in_window[win_vel <= threshold] if len(win_vel) else in_window
        if len(stationary) < 8:
            order = np.argsort(np.abs(frame_ids[in_window] - approx))
            stationary = in_window[order[: min(max(8, len(order) // 2), len(order))]]
        if len(stationary) > 35:
            order = np.argsort(np.abs(frame_ids[stationary] - approx))
            stationary = stationary[order[:35]]
        if len(stationary) == 0:
            continue

        est_filtered = np.median(filtered_world[stationary], axis=0)
        est_raw = np.median(raw_world[stationary], axis=0)
        gt_centroid = waypoint["world_centroid"]

        raw_xy = raw_world[stationary, :2]
        filt_xy = filtered_world[stationary, :2]
        raw_std = _radial_std(raw_xy)
        filt_std = _radial_std(filt_xy)
        raw_std_values.append(raw_std)
        filt_std_values.append(filt_std)

        err_xy = float(np.linalg.norm(est_filtered[:2] - waypoint["world_ground"][:2]))
        err_3d = float(np.linalg.norm(est_filtered - gt_centroid))
        errors_xy.append(err_xy)
        errors_3d.append(err_3d)

        stops.append(
            {
                "name": name,
                "approx_frame": approx,
                "frame_start": int(frame_ids[int(np.min(stationary))]),
                "frame_end": int(frame_ids[int(np.max(stationary))]),
                "num_frames": int(len(stationary)),
                "gt_world_ground": waypoint["world_ground"].tolist(),
                "gt_world_centroid": gt_centroid.tolist(),
                "estimated_raw_centroid": est_raw.tolist(),
                "estimated_filtered_centroid": est_filtered.tolist(),
                "error_xy_m": err_xy,
                "error_3d_centroid_m": err_3d,
                "raw_jitter_std_m": raw_std,
                "filtered_jitter_std_m": filt_std,
            }
        )

    rmse_xy = float(np.sqrt(np.mean(np.square(errors_xy)))) if errors_xy else float("nan")
    rmse_3d = float(np.sqrt(np.mean(np.square(errors_3d)))) if errors_3d else float("nan")
    raw_jitter = float(np.mean(raw_std_values)) if raw_std_values else float("nan")
    filt_jitter = float(np.mean(filt_std_values)) if filt_std_values else float("nan")
    reduction = float(100.0 * (1.0 - filt_jitter / raw_jitter)) if raw_jitter > 0 else float("nan")

    metrics = {
        "rmse_xy_m": rmse_xy,
        "rmse_3d_centroid_m": rmse_3d,
        "raw_jitter_std_m": raw_jitter,
        "filtered_jitter_std_m": filt_jitter,
        "jitter_reduction_pct": reduction,
    }
    return stops, metrics


def fit_waypoint_affine(stops: List[Dict[str, Any]]) -> np.ndarray | None:
    """Fit a 2D affine residual from estimated stops to waypoint stops.

    This is a post-run calibration correction, not a replacement for the
    physics-only coordinate stream. It is useful when the supplied video scale
    is inconsistent with the nominal camera/bin geometry but the tape markers
    are available as scene control points.
    """

    if len(stops) < 3:
        return None
    src = np.asarray([stop["estimated_filtered_centroid"][:2] for stop in stops], dtype=np.float64)
    dst = np.asarray([stop["gt_world_ground"][:2] for stop in stops], dtype=np.float64)
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        return None
    x = np.column_stack([src, np.ones(len(src), dtype=np.float64)])
    matrix, _, _, _ = np.linalg.lstsq(x, dst, rcond=None)
    return matrix


def apply_waypoint_affine(world_xyz: np.ndarray, matrix: np.ndarray | None) -> np.ndarray:
    if matrix is None:
        return world_xyz.copy()
    out = world_xyz.copy()
    x = np.column_stack([world_xyz[:, :2], np.ones(len(world_xyz), dtype=np.float64)])
    out[:, :2] = x @ matrix
    return out


def write_waypoint_calibrated_csv(
    path: str,
    frame_ids: np.ndarray,
    fps: float,
    strict_raw: np.ndarray,
    strict_filtered: np.ndarray,
    calibrated_raw: np.ndarray,
    calibrated_filtered: np.ndarray,
) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_id",
                "timestamp_ms",
                "x_world_raw_strict",
                "y_world_raw_strict",
                "x_world_filtered_strict",
                "y_world_filtered_strict",
                "x_world_raw_waypoint_calibrated",
                "y_world_raw_waypoint_calibrated",
                "x_world_filtered_waypoint_calibrated",
                "y_world_filtered_waypoint_calibrated",
                "z_world",
            ]
        )
        for i, frame_id in enumerate(frame_ids):
            timestamp_ms = int(round(1000.0 * int(frame_id) / max(1e-6, fps)))
            writer.writerow(
                [
                    int(frame_id),
                    timestamp_ms,
                    f"{strict_raw[i, 0]:.4f}",
                    f"{strict_raw[i, 1]:.4f}",
                    f"{strict_filtered[i, 0]:.4f}",
                    f"{strict_filtered[i, 1]:.4f}",
                    f"{calibrated_raw[i, 0]:.4f}",
                    f"{calibrated_raw[i, 1]:.4f}",
                    f"{calibrated_filtered[i, 0]:.4f}",
                    f"{calibrated_filtered[i, 1]:.4f}",
                    f"{calibrated_filtered[i, 2]:.4f}",
                ]
            )


def save_trajectory_plot(
    path: str,
    frame_ids: np.ndarray,
    raw_world: np.ndarray,
    filtered_world: np.ndarray,
    waypoints: Dict[str, Dict[str, Any]],
    stops: List[Dict[str, Any]],
    metrics: Dict[str, float],
) -> None:
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0])

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(raw_world[:, 0], raw_world[:, 1], color="0.72", linewidth=1.2, label="raw ground-plane")
    ax0.plot(filtered_world[:, 0], filtered_world[:, 1], color="#005f99", linewidth=2.2, label="Kalman filtered")
    for name, wp in waypoints.items():
        color = _marker_color(wp["color"])
        p = wp["world_ground"]
        ax0.scatter([p[0]], [p[1]], s=90, marker="x", color=color, linewidths=2.5, label=f"waypoint {name}")
    for stop in stops:
        p = np.asarray(stop["estimated_filtered_centroid"], dtype=np.float64)
        ax0.scatter([p[0]], [p[1]], s=70, marker="s", facecolor="none", edgecolor="black")
        ax0.text(p[0], p[1], f"  stop {stop['name']}", va="center", fontsize=9)
    ax0.set_title("World XY trajectory: raw vs Kalman filtered")
    ax0.set_xlabel("world X forward from pole (m)")
    ax0.set_ylabel("world Y left from pole (m)")
    ax0.axis("equal")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="best", fontsize=8, ncols=3)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(frame_ids, raw_world[:, 0], color="0.7", linewidth=1.0, label="raw X")
    ax1.plot(frame_ids, filtered_world[:, 0], color="#005f99", linewidth=1.6, label="filtered X")
    ax1.plot(frame_ids, raw_world[:, 1], color="#d0a20b", linewidth=1.0, alpha=0.65, label="raw Y")
    ax1.plot(frame_ids, filtered_world[:, 1], color="#b04f00", linewidth=1.6, label="filtered Y")
    ax1.set_title("Position traces")
    ax1.set_xlabel("frame")
    ax1.set_ylabel("metres")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8, ncols=2)

    ax2 = fig.add_subplot(gs[1, 1])
    vals = [metrics.get("raw_jitter_std_m", np.nan), metrics.get("filtered_jitter_std_m", np.nan)]
    ax2.bar(["raw", "filtered"], vals, color=["0.7", "#005f99"])
    ax2.set_title(f"Stationary jitter reduction: {metrics.get('jitter_reduction_pct', np.nan):.1f}%")
    ax2.set_ylabel("radial std in stop windows (m)")
    ax2.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_raw_vs_filtered_plot(
    path: str,
    frame_ids: np.ndarray,
    raw_world: np.ndarray,
    filtered_world: np.ndarray,
    states: List[str],
    sigma_world: np.ndarray | None,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    labels = ["X forward (m)", "Y left (m)", "Z up (m)"]
    raw_colors = ["0.68", "#c18f00", "0.55"]
    filt_colors = ["#005f99", "#b04f00", "#1b8f3a"]

    for dim, ax in enumerate(axes):
        ax.plot(frame_ids, raw_world[:, dim], color=raw_colors[dim], linewidth=1.0, label="raw")
        ax.plot(frame_ids, filtered_world[:, dim], color=filt_colors[dim], linewidth=1.7, label="filtered")
        if sigma_world is not None and len(sigma_world) == len(frame_ids):
            sigma = sigma_world[:, dim]
            valid = np.isfinite(sigma)
            if np.any(valid):
                ax.fill_between(
                    frame_ids[valid],
                    filtered_world[valid, dim] - sigma[valid],
                    filtered_world[valid, dim] + sigma[valid],
                    color=filt_colors[dim],
                    alpha=0.13,
                    linewidth=0,
                    label="+/- 1 sigma" if dim == 0 else None,
                )
        _shade_state_runs(ax, frame_ids, states)
        ax.set_ylabel(labels[dim])
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, ncols=3)

    axes[0].set_title("Raw vs Kalman-filtered world position with track-state intervals")
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _shade_state_runs(ax: Any, frame_ids: np.ndarray, states: List[str]) -> None:
    if not states or len(states) != len(frame_ids):
        return
    shade_colors = {
        "OCCLUDED": "#d62728",
        "REACQUIRED": "#9467bd",
        "SEARCHING": "#7f7f7f",
    }
    start = 0
    while start < len(states):
        state = states[start]
        end = start + 1
        while end < len(states) and states[end] == state:
            end += 1
        color = shade_colors.get(state)
        if color is not None:
            ax.axvspan(frame_ids[start], frame_ids[end - 1], color=color, alpha=0.10, linewidth=0)
        start = end


def _radial_std(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    centered = xy - np.mean(xy, axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def _marker_color(name: str) -> str:
    return {
        "green": "#1b8f3a",
        "orange": "#d47400",
        "red": "#c62828",
    }.get(name.lower(), "black")
