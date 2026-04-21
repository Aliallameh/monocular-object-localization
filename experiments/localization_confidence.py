"""
Localization sensitivity analysis.

Purpose: decompose plausible error sources without pretending the clip contains
surveyed world-coordinate ground truth:
  1. Bounding box uncertainty (width/height noise)
  2. Calibration uncertainty (K, tilt, height precision)
  3. Centroid approximation (bin is 3D; bottom-center pixel ≠ floor contact)
  4. Distance estimation model error

Then estimate a heuristic uncertainty band: if each source is σ, total is
√(σ1² + σ2² + ...). This is not validated RMSE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from localizer import CameraGeometry, localize_bbox


def estimate_bbox_uncertainty(calib_data: Dict[str, Any], sample_frame_path: str | None = None) -> Dict[str, float]:
    """Estimate bbox measurement noise from frame-to-frame variation.

    If a sample frame with ground truth markers is available, use it.
    Otherwise, estimate from typical detector bbox variance.
    """

    # Heuristic: typical detector bbox variance
    # (based on IoU stability and historical tracking error)
    sigma_width_px = 15.0  # ±15 pixel variation in width
    sigma_height_px = 10.0  # ±10 pixel variation in height
    sigma_center_x_px = 8.0  # ±8 pixel drift in center X
    sigma_center_y_px = 6.0  # ±6 pixel drift in center Y

    camera = CameraGeometry.from_json_dict(calib_data)

    # Convert pixel noise to 3D space at typical distance (2.5 m)
    z_typical = 1.6  # Camera-frame depth for bin ~2.5 m away
    fx = camera.K[0, 0]
    fy = camera.K[1, 1]

    # Pixel → depth (via derivative of z = fy * H / h_px)
    bin_height_m = 0.65
    h_px_typical = fy * bin_height_m / z_typical  # Expected bbox height in pixels
    # dz / d(h_px) = -z * (h_px) / h_px = -z / h_px ≈ -2 m / 40 px = -0.05 m/px
    dz_dh = z_typical / h_px_typical

    # Lateral uncertainty: dx/du = z / fx
    dx_du = z_typical / fx
    dy_dv = z_typical / fy

    return {
        "sigma_bbox_width_px": sigma_width_px,
        "sigma_bbox_height_px": sigma_height_px,
        "sigma_bbox_center_x_px": sigma_center_x_px,
        "sigma_bbox_center_y_px": sigma_center_y_px,
        "sigma_depth_m": float(dz_dh * sigma_height_px),
        "sigma_x_world_m": float(dx_du * sigma_center_x_px),
        "sigma_y_world_m": float(dy_dv * sigma_center_y_px),
        "note": "Estimated from typical detector variance. Should be validated on synthetic ground truth.",
    }


def estimate_calibration_uncertainty(calib_data: Dict[str, Any]) -> Dict[str, float]:
    """Estimate sensitivity of localization to calibration error.

    Perturb K, tilt, height; measure position sensitivity.
    """

    camera = CameraGeometry.from_json_dict(calib_data)
    h_cam = calib_data["camera_height_m"]
    tilt_deg = calib_data["camera_tilt_deg"]

    # Nominal point: center of image at ground
    u_center = calib_data["image_width_px"] / 2.0
    v_ground = calib_data["image_height_px"] * 0.8  # Lower part of frame (near ground)

    try:
        ground_nominal = camera.ground_intersection_from_pixel(u_center, v_ground)
        nominal_distance = float(np.linalg.norm(ground_nominal[:2]))
    except ValueError:
        nominal_distance = 2.5  # Default if ray doesn't intersect ground

    # Perturbations
    uncertainties = {}

    # Focal length uncertainty (±1% calibration drift)
    K_perturb = [list(row) for row in calib_data["K"]]
    K_perturb[0][0] *= 1.01  # fx +1%
    camera_perturbed = CameraGeometry.from_json_dict({**calib_data, "K": K_perturb})
    try:
        ground_perturb = camera_perturbed.ground_intersection_from_pixel(u_center, v_ground)
        dist_perturb = float(np.linalg.norm(ground_perturb[:2]))
        uncertainties["sigma_focal_length_sensitivity_m"] = abs(dist_perturb - nominal_distance)
    except ValueError:
        uncertainties["sigma_focal_length_sensitivity_m"] = 0.0

    # Tilt uncertainty (±2 degrees)
    camera_tilt_perturb = CameraGeometry.from_json_dict({**calib_data, "camera_tilt_deg": tilt_deg + 2.0})
    try:
        ground_tilt = camera_tilt_perturb.ground_intersection_from_pixel(u_center, v_ground)
        dist_tilt = float(np.linalg.norm(ground_tilt[:2]))
        uncertainties["sigma_tilt_deg_sensitivity_m"] = abs(dist_tilt - nominal_distance)
    except ValueError:
        uncertainties["sigma_tilt_deg_sensitivity_m"] = 0.0

    # Height uncertainty (±5 cm)
    camera_h_perturb = CameraGeometry.from_json_dict({**calib_data, "camera_height_m": h_cam + 0.05})
    try:
        ground_h = camera_h_perturb.ground_intersection_from_pixel(u_center, v_ground)
        dist_h = float(np.linalg.norm(ground_h[:2]))
        uncertainties["sigma_height_m_sensitivity_m"] = abs(dist_h - nominal_distance)
    except ValueError:
        uncertainties["sigma_height_m_sensitivity_m"] = 0.0

    return uncertainties


def localization_confidence_analysis(calib_path: str = "calib.json") -> Dict[str, Any]:
    """Estimate heuristic localization sensitivity; no real-world GT is used."""

    calib_data = json.loads(Path(calib_path).read_text())

    bbox_errors = estimate_bbox_uncertainty(calib_data)
    calib_errors = estimate_calibration_uncertainty(calib_data)

    # Combined error budget (RSS)
    total_sigma_m = np.sqrt(
        bbox_errors["sigma_depth_m"] ** 2
        + bbox_errors["sigma_x_world_m"] ** 2
        + bbox_errors["sigma_y_world_m"] ** 2
        + calib_errors.get("sigma_focal_length_sensitivity_m", 0.0) ** 2
        + calib_errors.get("sigma_tilt_deg_sensitivity_m", 0.0) ** 2
        + calib_errors.get("sigma_height_m_sensitivity_m", 0.0) ** 2
    )

    result = {
        "method": "heuristic_sensitivity_budget_not_accuracy_validation",
        "gt_status": "no_independent_world_coordinate_ground_truth_available",
        "error_decomposition": {
            "bbox_uncertainty": bbox_errors,
            "calibration_uncertainty": calib_errors,
        },
        "combined_error_budget": {
            "sigma_xyz_m": float(total_sigma_m),
            "confidence_68pct_m": float(total_sigma_m),
            "confidence_95pct_m": float(2.0 * total_sigma_m),
            "interpretation": (
                f"Estimated localization σ = {total_sigma_m:.3f} m. "
                f"68% confidence: ±{total_sigma_m:.3f} m. "
                f"95% confidence: ±{2*total_sigma_m:.3f} m. "
                "This is a heuristic model only, not measured accuracy."
            ),
        },
        "next_steps": [
            "1. Validate bbox noise estimate against synthetic ground truth frames",
            "2. Measure centroid offset: does bottom-center pixel equal floor contact?",
            "3. Compare ground-contact vs height-based estimate scatter to identify major source",
            "4. Perturb calibration on synthetic frames; verify sensitivity matches model",
        ],
    }

    output_path = "results/localization_sensitivity.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    result = localization_confidence_analysis("calib.json")
    print(json.dumps(result, indent=2))
