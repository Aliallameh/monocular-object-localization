"""Monocular geometry and Kalman smoothing for bin localization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


BIN_DIAMETER_M = 0.40
BIN_HEIGHT_M = 0.65


@dataclass(frozen=True)
class Localization:
    xyz_cam: np.ndarray
    xyz_world: np.ndarray
    xyz_cam_height: np.ndarray
    xyz_world_height: np.ndarray
    xyz_cam_ground_model: np.ndarray
    xyz_world_ground_model: np.ndarray
    xyz_world_ground_contact: np.ndarray
    depth_delta_m: float
    world_xy_delta_m: float
    used_fallback: bool


@dataclass
class CameraGeometry:
    K: np.ndarray
    D: np.ndarray
    camera_height_m: float
    tilt_rad: float
    R_cw: np.ndarray
    t_cw: np.ndarray
    fps: float
    image_size: Tuple[int, int]

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "CameraGeometry":
        K = np.asarray(data["K"], dtype=np.float64)
        D = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1)
        cam_h = float(data["camera_height_m"])
        tilt_rad = float(np.deg2rad(data["camera_tilt_deg"]))
        R_cw, t_cw = build_camera_to_world(cam_h, tilt_rad)
        fps = float(data.get("fps", 30.0))
        image_size = (int(data.get("image_width_px", 0)), int(data.get("image_height_px", 0)))
        return cls(K, D, cam_h, tilt_rad, R_cw, t_cw, fps, image_size)

    def cam_to_world(self, xyz_cam: np.ndarray) -> np.ndarray:
        return self.R_cw @ xyz_cam + self.t_cw

    def world_to_cam(self, xyz_world: np.ndarray) -> np.ndarray:
        return self.R_cw.T @ (xyz_world - self.t_cw)

    def undistort_normalized(self, u: float, v: float) -> np.ndarray:
        pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
        out = cv2.undistortPoints(pts, self.K, self.D)
        return np.array([out[0, 0, 0], out[0, 0, 1]], dtype=np.float64)

    def undistort_ideal_pixel(self, u: float, v: float) -> np.ndarray:
        pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
        out = cv2.undistortPoints(pts, self.K, self.D, P=self.K)
        return np.array([out[0, 0, 0], out[0, 0, 1]], dtype=np.float64)

    def ray_cam_from_pixel(self, u: float, v: float) -> np.ndarray:
        xy = self.undistort_normalized(u, v)
        ray = np.array([xy[0], xy[1], 1.0], dtype=np.float64)
        return ray / np.linalg.norm(ray)

    def ground_intersection_from_pixel(self, u: float, v: float) -> np.ndarray:
        ray_world = self.R_cw @ self.ray_cam_from_pixel(u, v)
        denom = ray_world[2]
        if denom >= -1e-6:
            raise ValueError("Pixel ray does not intersect the ground in front of the camera")
        lam = -self.t_cw[2] / denom
        if lam <= 0.0:
            raise ValueError("Ground-plane intersection is behind the camera")
        point = self.t_cw + lam * ray_world
        point[2] = 0.0
        return point


def build_camera_to_world(camera_height_m: float, tilt_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Return R, t where P_world = R @ P_cam + t.

    With zero pitch:
      cam +Z -> world +X, cam +X -> world -Y, cam +Y -> world -Z.

    calib.json stores downward tilt as a negative number. A -15 degree camera
    pitch means the optical axis has world Z component -sin(15 deg), so the
    world-Y rotation applied after the axis swap uses -tilt_rad.
    """

    axis_swap = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    pitch_down = -float(tilt_rad)
    c, s = np.cos(pitch_down), np.sin(pitch_down)
    ry = np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )
    R_cw = ry @ axis_swap
    t_cw = np.array([0.0, 0.0, float(camera_height_m)], dtype=np.float64)

    optical_world = R_cw @ np.array([0.0, 0.0, 1.0])
    if optical_world[0] <= 0.0 or optical_world[2] >= 0.0:
        raise ValueError("Camera transform sign check failed")
    return R_cw, t_cw


def height_based_centroid(bbox: tuple[float, float, float, float], camera: CameraGeometry) -> np.ndarray:
    """Estimate centroid in camera frame from known bin height and bbox height.

    This is the direct pinhole-distance estimator required by the assessment:
      Z = fy * H / h_px
      X = x_norm * Z
      Y = y_norm * Z
    It is logged for every frame. The default output uses the ground-contact
    model below because a fixed, calibrated camera over a flat floor gives a
    stronger world-frame constraint than bbox height alone.
    """

    x1, y1, x2, y2 = bbox
    uc = 0.5 * (x1 + x2)
    vc = 0.5 * (y1 + y2)
    top = camera.undistort_ideal_pixel(uc, y1)
    bottom = camera.undistort_ideal_pixel(uc, y2)
    h_px = max(1.0, abs(bottom[1] - top[1]))
    z = camera.K[1, 1] * BIN_HEIGHT_M / h_px

    xy = camera.undistort_normalized(uc, vc)
    x = xy[0] * z
    y = xy[1] * z
    return np.array([x, y, z], dtype=np.float64)


def estimate_height_based_centroid(bbox: tuple[float, float, float, float], camera: CameraGeometry) -> np.ndarray:
    """Backward-compatible alias for the explicit height-based estimator."""

    return height_based_centroid(bbox, camera)


def ground_contact_centroid(
    bbox: tuple[float, float, float, float],
    camera: CameraGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate centroid using bottom-center ray intersection with ground.

    Returns (centroid_cam, centroid_world, ground_contact_world). The centroid
    is lifted by half the known bin height from the ground contact point.
    """

    x1, _, x2, y2 = bbox
    bottom_u = 0.5 * (x1 + x2)
    ground = camera.ground_intersection_from_pixel(bottom_u, y2)
    world = ground + np.array([0.0, 0.0, BIN_HEIGHT_M * 0.5], dtype=np.float64)
    cam = camera.world_to_cam(world)
    return cam, world, ground


def localize_bbox(bbox: tuple[float, float, float, float], camera: CameraGeometry) -> Localization:
    height_cam = height_based_centroid(bbox, camera)
    height_world = camera.cam_to_world(height_cam)

    used_fallback = False
    try:
        cam, world, ground = ground_contact_centroid(bbox, camera)
    except ValueError:
        used_fallback = True
        cam = height_cam
        world = height_world
        ground = np.array([world[0], world[1], 0.0], dtype=np.float64)

    depth_delta = float(abs(cam[2] - height_cam[2]))
    world_xy_delta = float(np.linalg.norm(world[:2] - height_world[:2]))
    if not np.isfinite(cam).all() or cam[2] <= 0.0:
        used_fallback = True
        cam = height_cam
        world = height_world
        ground = np.array([world[0], world[1], 0.0], dtype=np.float64)
        depth_delta = 0.0
        world_xy_delta = 0.0

    return Localization(
        xyz_cam=cam,
        xyz_world=world,
        xyz_cam_height=height_cam,
        xyz_world_height=height_world,
        xyz_cam_ground_model=cam,
        xyz_world_ground_model=world,
        xyz_world_ground_contact=ground,
        depth_delta_m=depth_delta,
        world_xy_delta_m=world_xy_delta,
        used_fallback=used_fallback,
    )


def synthetic_geometry_validation(camera: CameraGeometry) -> Dict[str, Any]:
    """Validate localization math on synthetic projected bins.

    This is not video ground truth. It projects virtual bins through the supplied
    calibration, reconstructs them through both estimators, and reports numerical
    consistency only for the camera model and implementation.
    """

    cases = [
        (3.0, 0.0),
        (5.0, 0.0),
        (7.0, 0.0),
        (7.0, 1.0),
        (9.0, -1.0),
    ]
    records: List[Dict[str, Any]] = []
    for world_x, world_y in cases:
        bottom_world = np.array([world_x, world_y, 0.0], dtype=np.float64)
        top_world = np.array([world_x, world_y, BIN_HEIGHT_M], dtype=np.float64)
        centroid_world = np.array([world_x, world_y, BIN_HEIGHT_M * 0.5], dtype=np.float64)
        bottom_cam = camera.world_to_cam(bottom_world)
        top_cam = camera.world_to_cam(top_world)
        centroid_cam = camera.world_to_cam(centroid_world)
        if bottom_cam[2] <= 0.0 or top_cam[2] <= 0.0 or centroid_cam[2] <= 0.0:
            continue

        bottom_px = _project_cam_point(camera, bottom_cam)
        top_px = _project_cam_point(camera, top_cam)
        width_px = max(8.0, float(camera.K[0, 0] * BIN_DIAMETER_M / max(centroid_cam[2], 1e-6)))
        u = 0.5 * (bottom_px[0] + top_px[0])
        bbox = (
            float(u - 0.5 * width_px),
            float(min(top_px[1], bottom_px[1])),
            float(u + 0.5 * width_px),
            float(max(top_px[1], bottom_px[1])),
        )

        loc = localize_bbox(bbox, camera)
        height_world = loc.xyz_world_height
        ground_world = loc.xyz_world
        records.append(
            {
                "world_centroid_true": centroid_world.tolist(),
                "bbox_px": [float(v) for v in bbox],
                "ground_contact_est_world": ground_world.tolist(),
                "height_based_est_world": height_world.tolist(),
                "ground_contact_error_m": float(np.linalg.norm(ground_world - centroid_world)),
                "height_based_error_m": float(np.linalg.norm(height_world - centroid_world)),
                "height_based_depth_error_m": float(abs(loc.xyz_cam_height[2] - centroid_cam[2])),
            }
        )

    ground_errors = np.asarray([r["ground_contact_error_m"] for r in records], dtype=np.float64)
    height_errors = np.asarray([r["height_based_error_m"] for r in records], dtype=np.float64)
    return {
        "type": "synthetic_camera_model_check_not_video_gt",
        "case_count": len(records),
        "ground_contact_rmse_m": _rmse(ground_errors),
        "height_based_rmse_m": _rmse(height_errors),
        "max_ground_contact_error_m": float(np.max(ground_errors)) if len(ground_errors) else None,
        "max_height_based_error_m": float(np.max(height_errors)) if len(height_errors) else None,
        "cases": records,
    }


def _project_cam_point(camera: CameraGeometry, xyz_cam: np.ndarray) -> np.ndarray:
    pts, _ = cv2.projectPoints(
        np.asarray(xyz_cam, dtype=np.float64).reshape(1, 1, 3),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        camera.K,
        camera.D,
    )
    return np.array([pts[0, 0, 0], pts[0, 0, 1]], dtype=np.float64)


def _rmse(errors: np.ndarray) -> float | None:
    if len(errors) == 0:
        return None
    return float(np.sqrt(np.mean(errors * errors)))


class PositionKalman:
    """Constant-velocity Kalman filter over world position.

    State: [x, y, z, vx, vy, vz] in metres and metres/second.
    """

    def __init__(self, process_var: float = 1.0, measurement_var: float = 0.025) -> None:
        self.process_var = float(process_var)
        self.measurement_var = float(measurement_var)
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None

    @property
    def initialized(self) -> bool:
        return self.x is not None and self.P is not None

    def initialize(self, xyz: np.ndarray) -> np.ndarray:
        self.x = np.zeros(6, dtype=np.float64)
        self.x[:3] = xyz
        self.P = np.diag([0.08, 0.08, 0.05, 2.0, 2.0, 1.0]).astype(np.float64)
        return self.x[:3].copy()

    def predict(self, dt: float) -> np.ndarray | None:
        if self.x is None or self.P is None:
            return None
        F, Q = self._transition(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x[:3].copy()

    def predict_measurement(self, dt: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Return predicted measurement mean/covariance without mutating state."""

        if self.x is None or self.P is None:
            return None
        F, Q = self._transition(dt)
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q
        H = np.zeros((3, 6), dtype=np.float64)
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        return H @ x_pred, H @ P_pred @ H.T

    def innovation_mahalanobis(self, xyz_meas: np.ndarray, dt: float, measurement_var: float | None = None) -> float:
        """Squared Mahalanobis distance of a candidate world measurement."""

        predicted = self.predict_measurement(dt)
        if predicted is None:
            return 0.0
        mean, S0 = predicted
        r_var = self.measurement_var if measurement_var is None else float(measurement_var)
        S = S0 + np.eye(3, dtype=np.float64) * max(1e-6, r_var)
        innov = np.asarray(xyz_meas, dtype=np.float64) - mean
        try:
            return float(innov.T @ np.linalg.solve(S, innov))
        except np.linalg.LinAlgError:
            return float(innov.T @ np.linalg.pinv(S) @ innov)

    def update(self, xyz_meas: np.ndarray, dt: float, measurement_var: float | None = None) -> np.ndarray:
        if self.x is None or self.P is None:
            return self.initialize(xyz_meas)

        self.predict(dt)
        assert self.x is not None and self.P is not None
        H = np.zeros((3, 6), dtype=np.float64)
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        r_var = self.measurement_var if measurement_var is None else float(measurement_var)
        R = np.eye(3, dtype=np.float64) * max(1e-6, r_var)
        y = xyz_meas - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        # Joseph form keeps the covariance positive semi-definite after many
        # frames and mixed high/low-confidence measurements.
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        return self.x[:3].copy()

    def covariance_xyz(self) -> np.ndarray | None:
        if self.P is None:
            return None
        return self.P[:3, :3].copy()

    def sigma_xyz(self) -> np.ndarray | None:
        cov = self.covariance_xyz()
        if cov is None:
            return None
        return np.sqrt(np.maximum(np.diag(cov), 0.0))

    def stationary_probability(self) -> float:
        """Heuristic probability that the filtered target is near-stationary."""

        if self.x is None:
            return 0.0
        speed_xy = float(np.linalg.norm(self.x[3:5]))
        return float(np.clip(1.0 - speed_xy / 0.35, 0.0, 1.0))

    def _transition(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        dt = max(1e-4, float(dt))
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = F[1, 4] = F[2, 5] = dt

        q = self.process_var
        q_pos = 0.25 * dt**4 * q
        q_cross = 0.5 * dt**3 * q
        q_vel = dt**2 * q
        Q = np.array(
            [
                [q_pos, 0.0, 0.0, q_cross, 0.0, 0.0],
                [0.0, q_pos, 0.0, 0.0, q_cross, 0.0],
                [0.0, 0.0, q_pos, 0.0, 0.0, q_cross],
                [q_cross, 0.0, 0.0, q_vel, 0.0, 0.0],
                [0.0, q_cross, 0.0, 0.0, q_vel, 0.0],
                [0.0, 0.0, q_cross, 0.0, 0.0, q_vel],
            ],
            dtype=np.float64,
        )
        return F, Q
