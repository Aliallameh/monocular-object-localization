"""Monocular geometry and Kalman smoothing for bin localization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import cv2
import numpy as np


BIN_DIAMETER_M = 0.40
BIN_HEIGHT_M = 0.65


@dataclass(frozen=True)
class Localization:
    xyz_cam: np.ndarray
    xyz_world: np.ndarray
    xyz_cam_height: np.ndarray
    xyz_world_ground: np.ndarray
    depth_delta_m: float
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


def estimate_height_based_centroid(bbox: tuple[float, float, float, float], camera: CameraGeometry) -> np.ndarray:
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


def localize_bbox(bbox: tuple[float, float, float, float], camera: CameraGeometry) -> Localization:
    x1, y1, x2, y2 = bbox
    bottom_u = 0.5 * (x1 + x2)
    bottom_v = y2
    height_cam = estimate_height_based_centroid(bbox, camera)

    used_fallback = False
    try:
        ground = camera.ground_intersection_from_pixel(bottom_u, bottom_v)
        world = ground + np.array([0.0, 0.0, BIN_HEIGHT_M * 0.5], dtype=np.float64)
        cam = camera.world_to_cam(world)
    except ValueError:
        used_fallback = True
        cam = height_cam
        world = camera.cam_to_world(cam)
        ground = np.array([world[0], world[1], 0.0], dtype=np.float64)

    depth_delta = float(abs(cam[2] - height_cam[2]))
    if not np.isfinite(cam).all() or cam[2] <= 0.0:
        used_fallback = True
        cam = height_cam
        world = camera.cam_to_world(cam)
        ground = np.array([world[0], world[1], 0.0], dtype=np.float64)

    return Localization(
        xyz_cam=cam,
        xyz_world=world,
        xyz_cam_height=height_cam,
        xyz_world_ground=ground,
        depth_delta_m=depth_delta,
        used_fallback=used_fallback,
    )


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
