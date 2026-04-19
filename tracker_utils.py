"""Single-target association and bbox Kalman filtering."""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Optional

from detector import BBox, Detection


@dataclass
class TrackResult:
    bbox: BBox
    confidence: float
    status: str
    age: int
    matched_detection: Optional[Detection]


def bbox_to_cxcywh(bbox: BBox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array(
        [(x1 + x2) * 0.5, (y1 + y2) * 0.5, max(1.0, x2 - x1), max(1.0, y2 - y1)],
        dtype=np.float64,
    )


def cxcywh_to_bbox(z: np.ndarray) -> BBox:
    cx, cy, w, h = z[:4]
    w = max(1.0, float(w))
    h = max(1.0, float(h))
    return (float(cx - 0.5 * w), float(cy - 0.5 * h), float(cx + 0.5 * w), float(cy + 0.5 * h))


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0.0 else float(inter / denom)


class BBoxKalmanTracker:
    """Constant-velocity Kalman filter for one bin bbox.

    State: [cx, cy, w, h, vx, vy, vw, vh] in pixel units per frame.
    """

    def __init__(self, max_age: int = 25) -> None:
        self.max_age = max_age
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.age = 0
        self.last_conf = 0.0
        self.appearance = AppearanceModel()

    @property
    def active(self) -> bool:
        return self.x is not None and self.P is not None and self.age <= self.max_age

    def update(
        self,
        detections: Iterable[Detection],
        dt_frames: float = 1.0,
        frame: Optional[np.ndarray] = None,
    ) -> Optional[TrackResult]:
        dets = list(detections)
        if self.x is None:
            if not dets:
                return None
            best = dets[0]
            self._init(best.bbox, best.confidence)
            if frame is not None:
                self.appearance.update(frame, best.bbox, best.confidence)
            return TrackResult(best.bbox, best.confidence, "detected", 0, best)

        predicted_bbox = self.predict(dt_frames)
        best_det, best_score, partial = self._associate(predicted_bbox, dets, frame)

        if (best_det is None or best_score < 0.18) and self.age >= self.max_age and dets:
            reacq = self._reacquisition_candidate(dets, frame)
            if reacq is not None:
                self._init(reacq.bbox, reacq.confidence)
                if frame is not None:
                    self.appearance.update(frame, reacq.bbox, reacq.confidence)
                return TrackResult(reacq.bbox, reacq.confidence, "detected", 0, reacq)

        if best_det is None or best_score < 0.18:
            self.age += 1
            if self.age > self.max_age:
                if self.x is not None:
                    self.x[4:] *= 0.15
            conf = max(0.01, self.last_conf * (0.78 ** self.age))
            return TrackResult(predicted_bbox, conf, "occluded", self.age, None)

        if partial:
            # Do not let a person-sized occluder shrink the physical bin box.
            self.age += 1
            z_det = bbox_to_cxcywh(best_det.bbox)
            z_pred = bbox_to_cxcywh(predicted_bbox)
            blended = np.array([z_det[0], z_det[1], z_pred[2], z_pred[3]], dtype=np.float64)
            self._measurement_update(blended, max(0.08, best_det.confidence * 0.25))
            bbox = cxcywh_to_bbox(self.x[:4])
            conf = max(0.05, best_det.confidence * 0.45)
            self.last_conf = conf
            return TrackResult(bbox, conf, "occluded", self.age, best_det)

        if best_det.source == "lk_optical_flow":
            self.age += 1
            self._measurement_update(bbox_to_cxcywh(best_det.bbox), max(0.08, best_det.confidence * 0.45))
            bbox = cxcywh_to_bbox(self.x[:4])
            conf = max(0.05, best_det.confidence)
            self.last_conf = conf
            return TrackResult(bbox, conf, "occluded", self.age, best_det)

        self.age = 0
        self._measurement_update(bbox_to_cxcywh(best_det.bbox), best_det.confidence)
        bbox = cxcywh_to_bbox(self.x[:4])
        self.last_conf = best_det.confidence
        if frame is not None:
            self.appearance.update(frame, bbox, best_det.confidence)
        return TrackResult(bbox, best_det.confidence, "detected", 0, best_det)

    def predict(self, dt_frames: float = 1.0) -> BBox:
        if self.x is None or self.P is None:
            raise RuntimeError("Tracker has not been initialized")

        dt = max(1e-3, float(dt_frames))
        F = np.eye(8, dtype=np.float64)
        F[0, 4] = dt
        F[1, 5] = dt
        F[2, 6] = dt
        F[3, 7] = dt

        q_pos = 8.0
        q_size = 3.0
        Q = np.diag([q_pos, q_pos, q_size, q_size, 2.0, 2.0, 0.9, 0.9]).astype(np.float64)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.x[2] = max(8.0, self.x[2])
        self.x[3] = max(8.0, self.x[3])
        return cxcywh_to_bbox(self.x[:4])

    def _init(self, bbox: BBox, conf: float) -> None:
        z = bbox_to_cxcywh(bbox)
        self.x = np.zeros(8, dtype=np.float64)
        self.x[:4] = z
        self.P = np.diag([30.0, 30.0, 20.0, 20.0, 120.0, 120.0, 40.0, 40.0]).astype(np.float64)
        self.age = 0
        self.last_conf = conf

    def _measurement_update(self, z: np.ndarray, conf: float) -> None:
        if self.x is None or self.P is None:
            raise RuntimeError("Tracker has not been initialized")

        H = np.zeros((4, 8), dtype=np.float64)
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
        sigma = 22.0 - 16.0 * float(np.clip(conf, 0.0, 1.0))
        R = np.diag([sigma * sigma, sigma * sigma, 0.8 * sigma * sigma, 0.8 * sigma * sigma])

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ H) @ self.P
        self.x[2] = max(8.0, self.x[2])
        self.x[3] = max(8.0, self.x[3])

    def _associate(
        self,
        predicted_bbox: BBox,
        detections: list[Detection],
        frame: Optional[np.ndarray],
    ) -> tuple[Optional[Detection], float, bool]:
        if not detections:
            return None, 0.0, False

        pred_z = bbox_to_cxcywh(predicted_bbox)
        pred_area = pred_z[2] * pred_z[3]
        pred_diag = max(1.0, float(np.hypot(pred_z[2], pred_z[3])))

        best_det: Optional[Detection] = None
        best_score = -1.0
        best_partial = False

        for det in detections:
            det_z = bbox_to_cxcywh(det.bbox)
            det_area = det_z[2] * det_z[3]
            iou = bbox_iou(predicted_bbox, det.bbox)
            center_dist = float(np.hypot(det_z[0] - pred_z[0], det_z[1] - pred_z[1]))
            center_score = max(0.0, 1.0 - center_dist / (1.25 * pred_diag))
            area_ratio = det_area / max(1.0, pred_area)
            area_score = max(0.0, 1.0 - abs(np.log(max(area_ratio, 1e-3))))
            appearance_score = self.appearance.similarity(frame, det.bbox) if frame is not None else 0.5
            source_prior = _source_prior(det.source)

            score = (
                0.34 * iou
                + 0.22 * center_score
                + 0.15 * area_score
                + 0.17 * appearance_score
                + 0.08 * det.confidence
                + 0.04 * source_prior
            )
            plausible = iou > 0.015 or center_dist < 1.05 * pred_diag
            if self.appearance.ready and appearance_score < 0.16 and det.source not in {"lk_optical_flow"}:
                plausible = plausible and (iou > 0.18 or center_dist < 0.45 * pred_diag)
            if plausible and score > best_score:
                best_det = det
                best_score = score
                best_partial = area_ratio < 0.55 and iou > 0.03

        return best_det, float(best_score), bool(best_partial)

    def _reacquisition_candidate(self, detections: list[Detection], frame: Optional[np.ndarray]) -> Optional[Detection]:
        strong = [
            det
            for det in detections
            if det.source != "lk_optical_flow" and det.confidence >= 0.24
        ]
        if not strong:
            return None
        source_bonus = {
            "blue_hsv_shape": 0.18,
            "dark_rect_shape": 0.08,
            "edge_shape": -0.10,
        }
        return max(
            strong,
            key=lambda d: (
                0.55 * d.confidence
                + 0.30 * self.appearance.similarity(frame, d.bbox)
                + source_bonus.get(d.source, 0.0)
            ),
        )


class AppearanceModel:
    """Adaptive HSV-HS histogram for single-target identity.

    This is intentionally lightweight. It gives the tracker an identity memory
    without adding a neural re-ID dependency, and it adapts to whichever target
    was actually acquired instead of baking in "blue bin" as the identity.
    """

    def __init__(self, bins_h: int = 24, bins_s: int = 16) -> None:
        self.bins_h = bins_h
        self.bins_s = bins_s
        self.hist: Optional[np.ndarray] = None
        self.samples = 0

    @property
    def ready(self) -> bool:
        return self.hist is not None and self.samples >= 3

    def update(self, frame: np.ndarray, bbox: BBox, conf: float) -> None:
        hist = self._extract(frame, bbox)
        if hist is None:
            return
        alpha = float(np.clip(0.08 + 0.20 * conf, 0.08, 0.28))
        if self.hist is None:
            self.hist = hist
        else:
            self.hist = (1.0 - alpha) * self.hist + alpha * hist
            s = float(np.sum(self.hist))
            if s > 0:
                self.hist = self.hist / s
        self.samples += 1

    def similarity(self, frame: Optional[np.ndarray], bbox: BBox) -> float:
        if frame is None or self.hist is None:
            return 0.5
        hist = self._extract(frame, bbox)
        if hist is None:
            return 0.0
        score = cv2.compareHist(self.hist.astype(np.float32), hist.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
        return float(np.clip(1.0 - score, 0.0, 1.0))

    def _extract(self, frame: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in _clip_bbox(bbox, w_img, h_img)]
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        pad_x = max(1, int(0.08 * (x2 - x1)))
        pad_y = max(1, int(0.08 * (y2 - y1)))
        roi = frame[y1 + pad_y : y2 - pad_y, x1 + pad_x : x2 - pad_x]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 20, 25], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
        hist = cv2.calcHist([hsv], [0, 1], mask, [self.bins_h, self.bins_s], [0, 180, 0, 256])
        s = float(np.sum(hist))
        if s <= 1e-6:
            return None
        return (hist / s).astype(np.float32)


def _source_prior(source: str) -> float:
    if source.startswith("yolo_world"):
        return 1.0
    return {
        "blue_hsv_shape": 0.90,
        "dark_rect_shape": 0.70,
        "motion_foreground": 0.42,
        "edge_shape": 0.25,
        "lk_optical_flow": 0.30,
    }.get(source, 0.35)


class LKFlowPropagator:
    """Sparse optical-flow bbox propagation for short detector dropouts.

    This is deliberately small and inspectable. It does not replace detector
    measurements; it supplies a low-confidence candidate when the appearance
    detector is blurred, occluded, or temporarily misses the target.
    """

    def __init__(self, min_points: int = 8) -> None:
        self.min_points = int(min_points)
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_bbox: Optional[BBox] = None
        self.prev_points: Optional[np.ndarray] = None
        self.last_tracked_points: Optional[np.ndarray] = None

    def reset(self, gray: np.ndarray, bbox: BBox) -> None:
        self.prev_gray = gray.copy()
        self.prev_bbox = _clip_bbox(bbox, gray.shape[1], gray.shape[0])
        self.prev_points = self._points_in_bbox(gray, self.prev_bbox)
        self.last_tracked_points = None

    def update_reference(self, gray: np.ndarray, bbox: BBox) -> None:
        self.reset(gray, bbox)

    def accept_prediction(self, gray: np.ndarray, bbox: BBox) -> None:
        """Advance LK state after an accepted flow prediction without re-seeding.

        Re-detecting features inside a purely predicted/occluded box can latch on
        to an occluder or background. For detector gaps, carry forward only the
        points that actually tracked from the previous frame; the next real
        detector hit will reset the feature template.
        """

        if self.last_tracked_points is None or len(self.last_tracked_points) < self.min_points:
            return
        self.prev_gray = gray.copy()
        self.prev_bbox = _clip_bbox(bbox, gray.shape[1], gray.shape[0])
        self.prev_points = self.last_tracked_points.astype(np.float32).reshape(-1, 1, 2)

    def predict(self, gray: np.ndarray) -> tuple[Optional[BBox], float, int]:
        self.last_tracked_points = None
        if self.prev_gray is None or self.prev_bbox is None or self.prev_points is None:
            return None, 0.0, 0
        if len(self.prev_points) < self.min_points:
            return None, 0.0, int(len(self.prev_points))

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if next_pts is None or status is None:
            return None, 0.0, 0

        status = status.reshape(-1).astype(bool)
        old = self.prev_points.reshape(-1, 2)[status]
        new = next_pts.reshape(-1, 2)[status]
        if len(new) < self.min_points:
            return None, 0.0, int(len(new))
        self.last_tracked_points = new.astype(np.float32).reshape(-1, 1, 2)

        delta = np.median(new - old, axis=0)
        x1, y1, x2, y2 = self.prev_bbox
        moved = (x1 + delta[0], y1 + delta[1], x2 + delta[0], y2 + delta[1])
        moved = _clip_bbox(moved, gray.shape[1], gray.shape[0])

        residual = np.linalg.norm((new - old) - delta[None, :], axis=1)
        residual_med = float(np.median(residual)) if len(residual) else 999.0
        point_score = min(1.0, len(new) / 35.0)
        residual_score = max(0.0, 1.0 - residual_med / 12.0)
        quality = float(np.clip(0.65 * point_score + 0.35 * residual_score, 0.0, 1.0))
        return moved, quality, int(len(new))

    def _points_in_bbox(self, gray: np.ndarray, bbox: BBox) -> np.ndarray:
        x1, y1, x2, y2 = [int(round(v)) for v in _clip_bbox(bbox, gray.shape[1], gray.shape[0])]
        mask = np.zeros_like(gray)
        pad_x = max(2, int(0.04 * max(1, x2 - x1)))
        pad_y = max(2, int(0.04 * max(1, y2 - y1)))
        mask[max(0, y1 + pad_y) : min(gray.shape[0], y2 - pad_y), max(0, x1 + pad_x) : min(gray.shape[1], x2 - pad_x)] = 255
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=80,
            qualityLevel=0.01,
            minDistance=6,
            blockSize=5,
            mask=mask,
        )
        if pts is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return pts.astype(np.float32)


def _clip_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    x1 = float(np.clip(x1, 0, max(0, width - 2)))
    y1 = float(np.clip(y1, 0, max(0, height - 2)))
    x2 = float(np.clip(x2, x1 + 1, max(1, width - 1)))
    y2 = float(np.clip(y2, y1 + 1, max(1, height - 1)))
    return x1, y1, x2, y2
