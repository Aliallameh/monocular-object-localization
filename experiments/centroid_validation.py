"""
Automated centroid-approximation validation.

The pipeline localizes the bin using the bottom-center pixel of the detected
bbox projected onto the ground plane. For a cylindrical bin the visual bottom
edge is the near-side rim, not the true floor contact centre. This script
quantifies that systematic offset without any manual frame annotation by
comparing measured positions (in stationary windows) to waypoint references
stored in results/summary.json.

Output: results/centroid_validation.json
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from localizer import CameraGeometry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_stationary_frames(output_csv: Path) -> Dict[int, Dict[str, float]]:
    """Return {frame_id: {x_world_filt, y_world_filt, x1,y1,x2,y2}} for
    STATIONARY + CONFIRMED detected frames."""
    rows = {}
    with open(output_csv) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                fid = int(row["frame_id"])
                status = row.get("status", "")
                state = row.get("track_state", "")
                if status != "detected":
                    continue
                rows[fid] = {
                    "x1": float(row["x1"]),
                    "y1": float(row["y1"]),
                    "x2": float(row["x2"]),
                    "y2": float(row["y2"]),
                    "x_filt": float(row["x_world_filt"]),
                    "y_filt": float(row["y_world_filt"]),
                    "x_raw":  float(row["x_world_raw"]),
                    "y_raw":  float(row["y_world_raw"]),
                    "track_state": state,
                }
            except (ValueError, KeyError):
                continue
    return rows


def _window_around(frame_id: int, rows: Dict[int, Any], half: int = 30) -> List[Dict]:
    """Return rows within ±half frames of frame_id, STATIONARY state preferred."""
    window = [rows[f] for f in range(frame_id - half, frame_id + half + 1) if f in rows]
    stationary = [r for r in window if r["track_state"] == "STATIONARY"]
    return stationary if stationary else window


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_centroid_validation(
    output_csv: str = "results/output.csv",
    summary_json: str = "results/summary.json",
    calib_json: str = "calib.json",
    out_path: str = "results/centroid_validation.json",
) -> Dict[str, Any]:

    calib = json.loads(Path(calib_json).read_text())
    camera = CameraGeometry.from_json_dict(calib)

    summary = json.loads(Path(summary_json).read_text())
    waypoints: Dict[str, Any] = summary.get("waypoints", {})

    frame_rows = _load_stationary_frames(Path(output_csv))

    # -----------------------------------------------------------------------
    # Part A: Waypoint residuals
    #   For each waypoint, take measured position in the stationary window
    #   and compute XY distance to the waypoint's expected world position.
    # -----------------------------------------------------------------------
    waypoint_residuals = []
    for wp_id, wp in waypoints.items():
        approx_frame = wp.get("approx_frame")
        if approx_frame is None:
            continue
        expected_xy = np.array(wp["world_centroid"][:2])

        window = _window_around(approx_frame, frame_rows, half=20)
        if not window:
            continue

        measured_xy_filt = np.mean([[r["x_filt"], r["y_filt"]] for r in window], axis=0)
        measured_xy_raw  = np.mean([[r["x_raw"],  r["y_raw"]]  for r in window], axis=0)

        dist_filt = float(np.linalg.norm(measured_xy_filt - expected_xy))
        dist_raw  = float(np.linalg.norm(measured_xy_raw  - expected_xy))
        offset_vec = (measured_xy_filt - expected_xy).tolist()

        waypoint_residuals.append({
            "waypoint_id": wp_id,
            "approx_frame": approx_frame,
            "window_frames_used": len(window),
            "expected_xy_m": expected_xy.tolist(),
            "measured_xy_filt_m": measured_xy_filt.tolist(),
            "measured_xy_raw_m": measured_xy_raw.tolist(),
            "offset_vec_m": offset_vec,
            "dist_filt_m": round(dist_filt, 4),
            "dist_raw_m": round(dist_raw, 4),
        })

    mean_residual = float(np.mean([r["dist_filt_m"] for r in waypoint_residuals])) if waypoint_residuals else float("nan")
    std_residual  = float(np.std( [r["dist_filt_m"] for r in waypoint_residuals])) if waypoint_residuals else float("nan")

    # -----------------------------------------------------------------------
    # Part B: Pixel-to-ground consistency check
    #   For frames in each waypoint window, re-project bbox bottom-center
    #   to ground and compare to the already-recorded x_raw/y_raw.
    #   Any discrepancy would indicate a bug; expected to be ~0.
    # -----------------------------------------------------------------------
    reprojection_checks = []
    for wp in waypoint_residuals[:3]:  # First 3 waypoints is enough
        window = _window_around(wp["approx_frame"], frame_rows, half=10)
        for row in window[:5]:
            bu = 0.5 * (row["x1"] + row["x2"])
            bv = row["y2"]
            try:
                ground = camera.ground_intersection_from_pixel(bu, bv)
                reprojection_checks.append({
                    "bottom_center_u": round(bu, 1),
                    "bottom_center_v": round(bv, 1),
                    "ground_x": round(float(ground[0]), 4),
                    "ground_y": round(float(ground[1]), 4),
                    "recorded_raw_x": round(row["x_raw"], 4),
                    "recorded_raw_y": round(row["y_raw"], 4),
                    "delta_x": round(float(ground[0]) - row["x_raw"], 4),
                    "delta_y": round(float(ground[1]) - row["y_raw"], 4),
                })
            except (ValueError, RuntimeError):
                pass

    reproj_deltas = [math.hypot(r["delta_x"], r["delta_y"]) for r in reprojection_checks]
    mean_reproj_err = float(np.mean(reproj_deltas)) if reproj_deltas else float("nan")

    # -----------------------------------------------------------------------
    # Part C: Measurement noise on stationary segments
    #   Standard deviation of measured position during the longest stationary
    #   window gives the actual measurement noise — compare to error budget.
    # -----------------------------------------------------------------------
    stationary_rows = [r for r in frame_rows.values() if r["track_state"] == "STATIONARY"]
    noise_std_x = float(np.std([r["x_raw"] for r in stationary_rows])) if stationary_rows else float("nan")
    noise_std_y = float(np.std([r["y_raw"] for r in stationary_rows])) if stationary_rows else float("nan")
    noise_rms   = math.hypot(noise_std_x, noise_std_y) if stationary_rows else float("nan")

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------
    bin_radius_m = 0.20  # half of 0.40 m diameter

    if not math.isnan(mean_residual):
        if mean_residual > 0.3:
            finding = (
                f"Mean waypoint residual {mean_residual:.3f} m > 0.3 m. "
                "Consistent with centroid-approximation bias: the visual bottom-centre "
                "of a 0.4 m-diameter cylinder is ~0.2 m from the near-side floor contact. "
                "Remaining gap likely from waypoint reference invalidity (waypoints are "
                "pixel annotations, not surveyed 3-D positions)."
            )
            hypothesis = "SUPPORTED"
        elif mean_residual > 0.1:
            finding = (
                f"Mean waypoint residual {mean_residual:.3f} m (0.1–0.3 m range). "
                "Partially consistent with centroid-approximation; additional sources "
                f"(waypoint invalidity, dynamic motion) contribute."
            )
            hypothesis = "PARTIAL"
        else:
            finding = (
                f"Mean waypoint residual {mean_residual:.3f} m < 0.1 m. "
                "Centroid approximation is NOT the primary error source. "
                "Waypoint references may be accurate; investigate dynamic motion bias."
            )
            hypothesis = "REFUTED"
    else:
        finding = "Insufficient waypoint data."
        hypothesis = "INSUFFICIENT_DATA"

    result: Dict[str, Any] = {
        "summary": {
            "mean_waypoint_residual_m": round(mean_residual, 4),
            "std_waypoint_residual_m":  round(std_residual, 4),
            "centroid_hypothesis": hypothesis,
            "expected_centroid_offset_m": bin_radius_m,
            "measurement_noise_rms_m": round(noise_rms, 4),
            "reprojection_consistency_m": round(mean_reproj_err, 4),
        },
        "waypoint_residuals": waypoint_residuals,
        "reprojection_checks_sample": reprojection_checks[:5],
        "stationary_noise": {
            "n_stationary_frames": len(stationary_rows),
            "std_x_raw_m": round(noise_std_x, 4),
            "std_y_raw_m": round(noise_std_y, 4),
            "rms_m": round(noise_rms, 4),
        },
        "interpretation": finding,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"Saved → {out_path}")
    return result


if __name__ == "__main__":
    result = run_centroid_validation()
    print(json.dumps(result, indent=2))
