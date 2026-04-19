"""Detector backends for the Skyscouter bin assessment.

The default backend is deterministic and CPU-only: no downloaded weights, no
test-set training, and no false claim that COCO has a garbage-bin class. It uses
several inspectable proposal cues rather than a single model label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    confidence: float
    area_px: float
    source: str = "blue_hsv_shape"


class HybridBinDetector:
    """Classical multi-cue detector for the assessment videos.

    This is still not a universal trash-can recognizer. It is a reviewable
    detector stack for the available fixed-camera clips:

    - saturated-blue body proposal for canonical input.mp4,
    - dark-rectangular proposal for the development sample,
    - edge/shape fallback for blurred or desaturated frames.

    Identity and temporal consistency are handled by the single-target tracker,
    so this class intentionally returns candidate proposals rather than making
    a hard multi-object decision by itself.
    """

    def __init__(self) -> None:
        self.lower_blue = np.array([88, 45, 35], dtype=np.uint8)
        self.upper_blue = np.array([124, 255, 255], dtype=np.uint8)
        self.min_area_px = 550.0
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=180,
            varThreshold=24,
            detectShadows=True,
        )
        self.frame_index = 0

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        candidates: List[Detection] = []
        candidates.extend(self._detect_blue(frame_bgr))
        candidates.extend(self._detect_dark_rectangular_bin(frame_bgr))
        candidates.extend(self._detect_edge_shape(frame_bgr))
        candidates.extend(self._detect_motion_regions(frame_bgr))
        self.frame_index += 1
        return _non_max_suppression(candidates, iou_thresh=0.45)

    def metadata(self) -> Dict[str, object]:
        return {
            "backend": "hybrid_classical_cpu",
            "uses_native_pretrained_class": False,
            "uses_gpu": False,
            "proposal_sources": ["blue_hsv_shape", "dark_rect_shape", "edge_shape", "motion_foreground"],
            "notes": (
                "CPU-only multi-cue proposal detector. It avoids the false COCO "
                "trash-can-class assumption and is paired with temporal association."
            ),
        }

    def _detect_blue(self, frame_bgr: np.ndarray) -> List[Detection]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        # Remove tiny speckles, then reconnect the body around the white label
        # and specular highlights.
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 17))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Detection] = []
        h_img, w_img = frame_bgr.shape[:2]

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area_px:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if h <= 0 or w <= 0:
                continue

            aspect = w / float(h)
            if aspect < 0.45 or aspect > 3.25:
                continue

            x1, y1, x2, y2 = self._expand_to_full_bin(frame_bgr, (x, y, x + w, y + h))
            x1 = float(np.clip(x1, 0, w_img - 1))
            y1 = float(np.clip(y1, 0, h_img - 1))
            x2 = float(np.clip(x2, x1 + 1, w_img - 1))
            y2 = float(np.clip(y2, y1 + 1, h_img - 1))

            rect_area = max(1.0, float(w * h))
            fill = float(np.clip(area / rect_area, 0.0, 1.0))
            aspect_score = 1.0 - min(abs(aspect - 1.55) / 1.55, 1.0)
            size_score = min(area / 22000.0, 1.0)
            conf = 0.45 + 0.23 * fill + 0.22 * aspect_score + 0.10 * size_score
            conf = float(np.clip(conf, 0.05, 0.99))
            detections.append(Detection((x1, y1, x2, y2), conf, area))

        return detections

    def _detect_motion_regions(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Generic static-camera foreground proposals.

        This is not an identity detector. It is a safety net for videos where
        the bin is not blue/dark enough for the hand-authored cues but still
        moves against a mostly static camera. The tracker/appearance/world gate
        decides whether a motion blob belongs to the active target.
        """

        fg = self.bg_subtractor.apply(frame_bgr)
        if self.frame_index < 8:
            return []
        # Shadows are encoded near 127 by MOG2; keep only confident foreground.
        mask = cv2.inRange(fg, 200, 255)
        h_img, w_img = frame_bgr.shape[:2]
        mask[: int(0.25 * h_img), :] = 0
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 19)),
            iterations=2,
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 1800.0 or area > 220000.0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 35 or h < 45:
                continue
            aspect = w / float(h)
            if aspect < 0.28 or aspect > 3.8:
                continue
            fill = float(np.clip(area / max(1.0, w * h), 0.0, 1.0))
            y_score = float(np.clip((y + h - 0.32 * h_img) / (0.68 * h_img), 0.0, 1.0))
            area_score = float(np.clip(area / 35000.0, 0.0, 1.0))
            conf = float(np.clip(0.16 + 0.17 * fill + 0.09 * y_score + 0.06 * area_score, 0.05, 0.50))
            x1 = float(np.clip(x, 0, w_img - 1))
            y1 = float(np.clip(y, 0, h_img - 1))
            x2 = float(np.clip(x + w, x1 + 1, w_img - 1))
            y2 = float(np.clip(y + h, y1 + 1, h_img - 1))
            detections.append(Detection((x1, y1, x2, y2), conf, area, source="motion_foreground"))
        return detections

    def _detect_dark_rectangular_bin(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Fallback for the gray sample video shipped with the assessment.

        The bin there is a dark rectangular container on a gray floor. A fixed
        low-intensity threshold isolates the bin outline/body while ignoring the
        lighter person occluder in most frames. The tracker performs association,
        so this function can return plausible candidates rather than solve
        identity by itself.
        """

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 0, 58)
        mask[:250, :] = 0
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
            iterations=2,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Detection] = []
        h_img, w_img = frame_bgr.shape[:2]

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 1200.0 or area > 42000.0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if h < 55 or w < 35:
                continue
            aspect = w / float(h)
            if aspect < 0.32 or aspect > 1.25:
                continue
            if y < 250:
                continue

            pad_x = int(round(0.025 * w))
            pad_top = int(round(0.03 * h))
            x1 = float(np.clip(x - pad_x, 0, w_img - 1))
            y1 = float(np.clip(y - pad_top, 0, h_img - 1))
            x2 = float(np.clip(x + w + pad_x, x1 + 1, w_img - 1))
            y2 = float(np.clip(y + h, y1 + 1, h_img - 1))

            fill = float(np.clip(area / max(1.0, w * h), 0.0, 1.0))
            aspect_score = 1.0 - min(abs(aspect - 0.58) / 0.58, 1.0)
            size_score = min(area / 18000.0, 1.0)
            conf = 0.46 + 0.20 * fill + 0.24 * aspect_score + 0.10 * size_score
            conf = float(np.clip(conf, 0.05, 0.96))
            detections.append(Detection((x1, y1, x2, y2), conf, area, source="dark_rect_shape"))

        return detections

    def _detect_edge_shape(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Low-confidence geometry fallback independent of exact bin color.

        This path is deliberately conservative. It looks for compact rectangular
        foreground-like structures in the lower half of the frame and gives them
        low confidence, so the tracker can use them only when they agree with the
        predicted target state.
        """

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 45, 130)
        edges[: int(0.28 * edges.shape[0]), :] = 0
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (19, 13)),
            iterations=1,
        )

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Detection] = []
        h_img, w_img = frame_bgr.shape[:2]
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 2500.0 or area > 280000.0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 45 or h < 45:
                continue
            aspect = w / float(h)
            if aspect < 0.45 or aspect > 3.6:
                continue
            if y + h < int(0.38 * h_img):
                continue

            rect_area = max(1.0, float(w * h))
            fill = float(np.clip(area / rect_area, 0.0, 1.0))
            aspect_score = 1.0 - min(abs(aspect - 1.45) / 1.45, 1.0)
            y_score = float(np.clip((y + h - 0.35 * h_img) / (0.65 * h_img), 0.0, 1.0))
            conf = float(np.clip(0.18 + 0.18 * fill + 0.12 * aspect_score + 0.08 * y_score, 0.05, 0.52))
            x1 = float(np.clip(x, 0, w_img - 1))
            y1 = float(np.clip(y, 0, h_img - 1))
            x2 = float(np.clip(x + w, x1 + 1, w_img - 1))
            y2 = float(np.clip(y + h, y1 + 1, h_img - 1))
            detections.append(Detection((x1, y1, x2, y2), conf, area, source="edge_shape"))
        return detections

    def _expand_to_full_bin(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> BBox:
        """Expand blue-body box to include lid overhang and wheel contact.

        The ground-plane localizer needs the bottom contact point, but the wheels
        are black and not part of the blue mask. This small local search looks
        for compact dark blobs directly under the blue body and extends the box
        only when such wheel-like evidence is present.
        """

        x1, y1, x2, y2 = bbox
        body_w = max(1, x2 - x1)
        body_h = max(1, y2 - y1)
        h_img, w_img = frame_bgr.shape[:2]

        pad_x = int(round(0.025 * body_w))
        top_pad = int(round(0.025 * body_h))
        out_x1 = max(0, x1 - pad_x)
        out_x2 = min(w_img - 1, x2 + pad_x)
        out_y1 = max(0, y1 - top_pad)
        out_y2 = y2

        sx1 = max(0, x1 - int(0.08 * body_w))
        sx2 = min(w_img, x2 + int(0.08 * body_w))
        sy1 = max(0, y2 - int(0.12 * body_h))
        sy2 = min(h_img, y2 + max(22, int(0.30 * body_h)))
        if sx2 <= sx1 or sy2 <= sy1:
            return out_x1, out_y1, out_x2, min(h_img - 1, out_y2)

        roi = frame_bgr[sy1:sy2, sx1:sx2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        dark = cv2.inRange(gray, 0, 78)
        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        wheel_bottom = out_y2
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 8:
                continue
            wx, wy, ww, wh = cv2.boundingRect(contour)
            abs_x1 = sx1 + wx
            abs_y1 = sy1 + wy
            abs_y2 = abs_y1 + wh
            compact_width = ww <= max(8, int(0.22 * body_w))
            compact_height = wh <= max(8, int(0.45 * body_h))
            below_body = abs_y2 >= y2 - int(0.03 * body_h)
            horizontally_plausible = (x1 - 0.06 * body_w) <= abs_x1 <= (x2 + 0.06 * body_w)
            if compact_width and compact_height and below_body and horizontally_plausible:
                wheel_bottom = max(wheel_bottom, abs_y2)

        # If wheels were not individually detected, keep a conservative extension
        # because the blue body stops above the floor contact in this video.
        conservative_bottom = y2 + int(round(0.10 * body_h))
        out_y2 = max(wheel_bottom, conservative_bottom)
        out_y2 = min(h_img - 1, out_y2)
        return out_x1, out_y1, out_x2, out_y2


class YoloWorldBinDetector:
    """Optional open-vocabulary detector backend.

    This path is intentionally optional: it is the more general detector choice,
    but it depends on torch/ultralytics weights and can violate the assessment's
    first-output latency on a clean CPU machine. The default hybrid detector
    remains the deterministic fallback.
    """

    def __init__(
        self,
        weights: str | None = None,
        device: str = "cpu",
        conf: float = 0.05,
        imgsz: int = 640,
    ) -> None:
        try:
            from ultralytics import YOLOWorld
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("ultralytics YOLOWorld is not installed") from exc

        self.weights = weights or _default_yolo_world_weights()
        self.device = device
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.prompts = ["trash bin", "garbage bin", "trash can", "blue trash can", "wheelie bin"]
        self.model = YOLOWorld(self.weights)
        self.model.set_classes(self.prompts)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return detections
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return detections

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        cls_ids = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
        h_img, w_img = frame_bgr.shape[:2]
        for box, conf, cls_id in zip(xyxy, confs, cls_ids):
            x1, y1, x2, y2 = [float(v) for v in box]
            x1 = float(np.clip(x1, 0, w_img - 1))
            y1 = float(np.clip(y1, 0, h_img - 1))
            x2 = float(np.clip(x2, x1 + 1, w_img - 1))
            y2 = float(np.clip(y2, y1 + 1, h_img - 1))
            area = max(1.0, (x2 - x1) * (y2 - y1))
            prompt = self.prompts[int(cls_id)] if 0 <= int(cls_id) < len(self.prompts) else "open_vocab_bin"
            detections.append(Detection((x1, y1, x2, y2), float(conf), float(area), source=f"yolo_world:{prompt}"))
        return _non_max_suppression(detections, iou_thresh=0.55)

    def metadata(self) -> Dict[str, object]:
        return {
            "backend": "yolo_world_open_vocabulary",
            "weights": self.weights,
            "device": self.device,
            "imgsz": self.imgsz,
            "confidence_threshold": self.conf,
            "uses_native_pretrained_class": False,
            "uses_gpu": self.device.startswith("cuda"),
            "prompts": self.prompts,
            "notes": (
                "Open-vocabulary target prompts are used because standard COCO "
                "detectors do not expose a garbage-bin/trash-can class."
            ),
        }


class DetectorCascade:
    """Run an optional learned detector first, then add hybrid proposals."""

    def __init__(self, primary: YoloWorldBinDetector, fallback: HybridBinDetector) -> None:
        self.primary = primary
        self.fallback = fallback

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        detections = list(self.primary.detect(frame_bgr))
        detections.extend(self.fallback.detect(frame_bgr))
        return _non_max_suppression(detections, iou_thresh=0.50)

    def metadata(self) -> Dict[str, object]:
        return {
            "backend": "cascade_yolo_world_plus_hybrid",
            "primary": self.primary.metadata(),
            "fallback": self.fallback.metadata(),
        }


def _non_max_suppression(detections: List[Detection], iou_thresh: float) -> List[Detection]:
    if not detections:
        return []
    ordered = sorted(detections, key=lambda d: (d.confidence, d.area_px), reverse=True)
    kept: List[Detection] = []
    for det in ordered:
        if all(_bbox_iou(det.bbox, prev.bbox) < iou_thresh for prev in kept):
            kept.append(det)
    return kept


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0.0 else float(inter / denom)


def _default_yolo_world_weights() -> str:
    candidates = [
        Path("yolov8s-worldv2.pt"),
        Path("projectA/yolov8s-worldv2.pt"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return "yolov8s-worldv2.pt"


def load_detector(
    use_gpu: bool = False,
    backend: str = "hybrid",
    device: str = "auto",
    conf: float = 0.05,
    imgsz: int = 640,
) -> object:
    backend = backend.lower()
    requested_device = device.lower()
    if requested_device == "auto":
        requested_device = "cuda:0" if use_gpu else "cpu"

    if backend == "hybrid":
        if use_gpu:
            print(
                "[detector] --gpu requested, but hybrid backend is CPU-only; no CUDA device is used.",
                flush=True,
            )
        print("[detector] using hybrid classical proposal backend", flush=True)
        return HybridBinDetector()

    fallback = HybridBinDetector()
    try:
        learned = YoloWorldBinDetector(device=requested_device, conf=conf, imgsz=imgsz)
        if backend == "yolo_world":
            print(f"[detector] using YOLO-World open-vocabulary backend on device={requested_device}", flush=True)
            return learned
        if backend == "auto":
            print(
                f"[detector] using YOLO-World + hybrid cascade on device={requested_device}",
                flush=True,
            )
            return DetectorCascade(learned, fallback)
    except Exception as exc:
        if backend == "yolo_world":
            raise
        print(
            f"[detector] YOLO-World unavailable ({exc!r}); falling back to hybrid classical backend",
            flush=True,
        )

    if use_gpu:
        print(
            "[detector] --gpu requested, but active fallback backend is CPU-only; no CUDA device is used.",
            flush=True,
        )
    print("[detector] using hybrid classical proposal backend", flush=True)
    return HybridBinDetector()


BlueBinDetector = HybridBinDetector
