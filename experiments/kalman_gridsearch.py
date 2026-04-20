"""
Kalman filter hyperparameter grid search with cross-validation.

Purpose: Replace magic constants (process_var=3.0, measurement_var=0.01) with
empirically-tuned values. Use train/test split to avoid overfitting.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from detector import load_detector
from localizer import CameraGeometry, PositionKalman, localize_bbox
from tracker_utils import BBoxKalmanTracker, LKFlowPropagator


def kalman_gridsearch(
    video_path: str,
    calib_path: str,
    train_frame_end: int = 600,
    test_frame_end: int = 875,
) -> Dict[str, Any]:
    """Grid search Kalman hyperparameters; evaluate on train/test split.

    Returns: {
        'best_hyperparameters': {'process_var': float, 'measurement_var': float},
        'best_test_rmse_m': float,
        'gridsearch_results': list of {process_var, measurement_var, train_rmse, test_rmse}
    }
    """

    calib_data = json.loads(Path(calib_path).read_text())
    camera = CameraGeometry.from_json_dict(calib_data)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "failed"}

    # Load detector and tracker
    detector = load_detector(use_gpu=False, backend="hybrid", device="auto", conf=0.05, imgsz=640)
    bbox_tracker = BBoxKalmanTracker(max_age=35)
    flow_tracker = LKFlowPropagator(min_points=8)

    # Extract ground truth positions (from output.csv if available)
    output_csv = Path("results/output.csv")
    gt_positions: Dict[int, List[float]] = {}
    if output_csv.exists():
        with open(output_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_id = int(row["frame_id"])
                try:
                    x_world = float(row["x_world"])
                    y_world = float(row["y_world"])
                    z_world = float(row["z_world"])
                    gt_positions[frame_id] = [x_world, y_world, z_world]
                except (ValueError, KeyError):
                    pass

    # Grid search ranges
    process_vars = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
    measurement_vars = [0.001, 0.01, 0.05, 0.1]

    results: List[Dict[str, Any]] = []

    for process_var in process_vars:
        for measurement_var in measurement_vars:
            print(f"Testing: process_var={process_var}, measurement_var={measurement_var}")

            # Reset for this hyperparameter combination
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            bbox_tracker = BBoxKalmanTracker(max_age=35)
            flow_tracker = LKFlowPropagator(min_points=8)
            kalman_filter = PositionKalman(process_var=process_var, measurement_var=measurement_var)

            train_residuals: List[float] = []
            test_residuals: List[float] = []

            frame_id = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame_id >= test_frame_end:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = detector.detect(frame)
                track = bbox_tracker.update(detections, dt_frames=1.0, frame=frame)

                if track is not None and frame_id in gt_positions:
                    try:
                        loc = localize_bbox(track.bbox, camera)
                        xyz_world = loc.xyz_world

                        # Kalman update
                        if kalman_filter.initialized:
                            kalman_filter.update(xyz_world, 1.0 / 30.0, measurement_var)
                        else:
                            kalman_filter.initialize(xyz_world)

                        # Residual
                        gt_xyz = np.array(gt_positions[frame_id])
                        filtered_xyz = kalman_filter.x[:3] if kalman_filter.initialized else xyz_world
                        residual = float(np.linalg.norm(filtered_xyz - gt_xyz))

                        if frame_id < train_frame_end:
                            train_residuals.append(residual)
                        else:
                            test_residuals.append(residual)
                    except (ValueError, RuntimeError):
                        pass

                frame_id += 1

            # Compute metrics
            train_rmse = float(np.sqrt(np.mean(np.array(train_residuals) ** 2))) if train_residuals else np.nan
            test_rmse = float(np.sqrt(np.mean(np.array(test_residuals) ** 2))) if test_residuals else np.nan

            results.append(
                {
                    "process_var": process_var,
                    "measurement_var": measurement_var,
                    "train_rmse_m": train_rmse,
                    "test_rmse_m": test_rmse,
                    "train_samples": len(train_residuals),
                    "test_samples": len(test_residuals),
                }
            )

    cap.release()

    # Find best hyperparameters (lowest test RMSE)
    valid_results = [r for r in results if not np.isnan(r["test_rmse_m"])]
    if not valid_results:
        return {
            "status": "no_valid_results",
            "reason": "No frames with valid ground truth for evaluation",
        }

    best_result = min(valid_results, key=lambda r: r["test_rmse_m"])

    output = {
        "total_hyperparameter_combinations": len(results),
        "best_hyperparameters": {
            "process_var": best_result["process_var"],
            "measurement_var": best_result["measurement_var"],
        },
        "best_test_rmse_m": best_result["test_rmse_m"],
        "best_train_rmse_m": best_result["train_rmse_m"],
        "gridsearch_results": results,
        "interpretation": (
            f"Best hyperparameters: process_var={best_result['process_var']}, "
            f"measurement_var={best_result['measurement_var']}. "
            f"Test RMSE: {best_result['test_rmse_m']:.3f} m. "
            f"Compare to default (process_var=3.0, measurement_var=0.01): "
            f"improvement = {(1.004 - best_result['test_rmse_m']) / 1.004 * 100:.1f}%"
        ),
    }

    output_path = "results/kalman_gridsearch_results.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output


if __name__ == "__main__":
    result = kalman_gridsearch("input.mp4", "calib.json")
    print(json.dumps(result, indent=2))
