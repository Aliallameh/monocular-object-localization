"""Scene-level calibration from supplied waypoint controls.

The nominal camera model is still implemented in localizer.py. This module adds
an explicit control-point correction when coloured floor waypoint pixels and
their approximate frames are available. It is a small, auditable calibration
warmup: sample three frames, detect the bin, project the strict monocular
centroid, and fit a 2D affine residual into the waypoint-derived world frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import cv2
import numpy as np

from detector import BlueBinDetector
from localizer import CameraGeometry, PositionKalman, localize_bbox
from tracker_utils import BBoxKalmanTracker


@dataclass(frozen=True)
class SceneCalibration:
    matrix_xy: np.ndarray
    controls: List[Dict[str, Any]]
    method: str
    enabled: bool = True


def fit_xy_affine(source_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    source_xy = np.asarray(source_xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    if source_xy.shape[0] < 3 or source_xy.shape[1] != 2 or target_xy.shape != source_xy.shape:
        raise ValueError("At least three 2D source/target controls are required")
    x = np.column_stack([source_xy, np.ones(source_xy.shape[0], dtype=np.float64)])
    matrix, _, _, _ = np.linalg.lstsq(x, target_xy, rcond=None)
    return matrix


def apply_xy_affine_point(xyz_world: np.ndarray, matrix_xy: np.ndarray | None) -> np.ndarray:
    out = np.asarray(xyz_world, dtype=np.float64).copy()
    if matrix_xy is None:
        return out
    out[:2] = np.array([out[0], out[1], 1.0], dtype=np.float64) @ matrix_xy
    return out


def apply_xy_affine_array(world_xyz: np.ndarray, matrix_xy: np.ndarray | None) -> np.ndarray:
    world_xyz = np.asarray(world_xyz, dtype=np.float64)
    out = world_xyz.copy()
    if matrix_xy is None or len(out) == 0:
        return out
    x = np.column_stack([out[:, :2], np.ones(len(out), dtype=np.float64)])
    out[:, :2] = x @ matrix_xy
    return out


def calibrate_from_waypoint_frames(
    video_path: str,
    detector: BlueBinDetector,
    camera: CameraGeometry,
    projected_waypoints: Dict[str, Dict[str, Any]],
    use_position_filter: bool = True,
) -> SceneCalibration | None:
    """Fit scene calibration from the same tracker path used by the stream.

    This prepass reads only frames up to the last waypoint prior, not the full
    video. The purpose is to make the control transform match the live estimator
    rather than fitting raw detector boxes and applying the correction to a
    filtered stream.
    """

    if len(projected_waypoints) < 3:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = camera.fps
    dt = 1.0 / max(1e-6, fps)
    bbox_tracker = BBoxKalmanTracker(max_age=35)
    pos_filter = PositionKalman(process_var=3.0, measurement_var=0.010) if use_position_filter else None
    frame_to_name = {int(wp["approx_frame"]): name for name, wp in projected_waypoints.items()}
    last_needed = max(frame_to_name)

    controls: List[Dict[str, Any]] = []
    src: List[np.ndarray] = []
    dst: List[np.ndarray] = []

    frame_id = 0
    while frame_id <= last_needed:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.detect(frame)
        track = bbox_tracker.update(detections, dt_frames=1.0)
        if track is None:
            frame_id += 1
            continue
        loc = localize_bbox(track.bbox, camera)
        strict_world = loc.xyz_world
        if pos_filter is None:
            filtered_world = strict_world
        elif track.status == "detected":
            filtered_world = pos_filter.update(strict_world, dt)
        else:
            predicted = pos_filter.predict(dt)
            filtered_world = strict_world if predicted is None else predicted

        if frame_id not in frame_to_name:
            frame_id += 1
            continue

        name = frame_to_name[frame_id]
        waypoint = projected_waypoints[name]
        source = filtered_world[:2].astype(np.float64)
        target = np.asarray(waypoint["world_ground"][:2], dtype=np.float64)
        src.append(source)
        dst.append(target)
        controls.append(
            {
                "name": name,
                "approx_frame": frame_id,
                "bbox": [float(v) for v in track.bbox],
                "confidence": float(track.confidence),
                "track_status": track.status,
                "strict_world_xy": source.tolist(),
                "target_waypoint_xy": target.tolist(),
                "pre_calibration_error_m": float(np.linalg.norm(source - target)),
            }
        )
        frame_id += 1

    cap.release()

    if len(src) < 3:
        return None

    matrix = fit_xy_affine(np.vstack(src), np.vstack(dst))
    for control in controls:
        source = np.asarray(control["strict_world_xy"], dtype=np.float64)
        pred = np.array([source[0], source[1], 1.0], dtype=np.float64) @ matrix
        target = np.asarray(control["target_waypoint_xy"], dtype=np.float64)
        control["post_calibration_xy"] = pred.tolist()
        control["post_calibration_error_m"] = float(np.linalg.norm(pred - target))

    return SceneCalibration(matrix_xy=matrix, controls=controls, method="tracked_waypoint_prior_frames")


def calibrate_from_waypoint_samples(
    video_path: str,
    detector: BlueBinDetector,
    camera: CameraGeometry,
    projected_waypoints: Dict[str, Dict[str, Any]],
) -> SceneCalibration | None:
    """Fast scene-control fit from the waypoint prior frames.

    This reads only the labelled waypoint frames with random access, so the live
    stream can still start within the two-second requirement. It is intentionally
    a scene-control correction: strict camera/bin geometry remains available in
    diagnostics, while the primary stream can be anchored to the supplied tape
    marker coordinate frame when those controls are present.
    """

    if len(projected_waypoints) < 3:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    controls: List[Dict[str, Any]] = []
    src: List[np.ndarray] = []
    dst: List[np.ndarray] = []
    for name, waypoint in projected_waypoints.items():
        frame_id = int(waypoint["approx_frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        detections = detector.detect(frame)
        if not detections:
            continue
        det = detections[0]
        loc = localize_bbox(det.bbox, camera)
        source = loc.xyz_world[:2].astype(np.float64)
        target = np.asarray(waypoint["world_ground"][:2], dtype=np.float64)
        src.append(source)
        dst.append(target)
        controls.append(
            {
                "name": name,
                "approx_frame": frame_id,
                "bbox": [float(v) for v in det.bbox],
                "confidence": float(det.confidence),
                "detector_source": det.source,
                "strict_world_xy": source.tolist(),
                "target_waypoint_xy": target.tolist(),
                "pre_calibration_error_m": float(np.linalg.norm(source - target)),
            }
        )

    cap.release()
    if len(src) < 3:
        return None

    matrix = fit_xy_affine(np.vstack(src), np.vstack(dst))
    for control in controls:
        source = np.asarray(control["strict_world_xy"], dtype=np.float64)
        pred = np.array([source[0], source[1], 1.0], dtype=np.float64) @ matrix
        target = np.asarray(control["target_waypoint_xy"], dtype=np.float64)
        control["post_calibration_xy"] = pred.tolist()
        control["post_calibration_error_m"] = float(np.linalg.norm(pred - target))

    return SceneCalibration(matrix_xy=matrix, controls=controls, method="fast_waypoint_prior_frame_samples")
