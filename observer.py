"""Visualization-only observer overlay for the tracking pipeline.

The observer consumes already-computed tracker/localizer state and renders a
review video plus event log. It must never feed labels or measurements back into
the detector, tracker, Kalman filter, waypoint analysis, or output CSV.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np


@dataclass
class MotionEstimate:
    label: str
    speed_mps: float
    heading_deg: float | None
    heading_delta_deg: float | None


class MotionStateEstimator:
    """Classify reviewer-facing motion state from filtered world positions."""

    def __init__(
        self,
        stationary_speed_mps: float = 0.055,
        moving_speed_mps: float = 0.14,
        direction_change_deg: float = 32.0,
    ) -> None:
        self.stationary_speed_mps = float(stationary_speed_mps)
        self.moving_speed_mps = float(moving_speed_mps)
        self.direction_change_deg = float(direction_change_deg)
        self.prev_xy: np.ndarray | None = None
        self.prev_heading_deg: float | None = None
        self.speed_ema: float = 0.0

    def update(self, obs: Dict[str, Any], dt: float) -> MotionEstimate:
        status = str(obs.get("status", "")).lower()
        track_state = str(obs.get("track_state", "")).upper()
        xy = _xy(obs.get("world"))

        if status in {"searching", ""} or xy is None:
            return MotionEstimate("IDLE", 0.0, None, None)
        if track_state == "OCCLUDED" or status == "occluded":
            speed, heading, delta = self._kinematics(xy, dt)
            return MotionEstimate("OCCLUDED", speed, heading, delta)
        if track_state == "REACQUIRED":
            speed, heading, delta = self._kinematics(xy, dt)
            return MotionEstimate("REACQUIRED", speed, heading, delta)
        if float(obs.get("conf", 0.0)) < 0.35:
            speed, heading, delta = self._kinematics(xy, dt)
            return MotionEstimate("LOW_CONFIDENCE", speed, heading, delta)

        speed, heading, delta = self._kinematics(xy, dt)
        if speed <= self.stationary_speed_mps:
            label = "STATIONARY"
        elif delta is not None and speed >= self.moving_speed_mps and abs(delta) >= self.direction_change_deg:
            label = "CHANGING_DIRECTION"
        elif speed >= self.moving_speed_mps:
            label = "MOVING"
        else:
            label = "SETTLING"
        return MotionEstimate(label, speed, heading, delta)

    def _kinematics(self, xy: np.ndarray, dt: float) -> tuple[float, float | None, float | None]:
        dt = max(1e-6, float(dt))
        if self.prev_xy is None:
            self.prev_xy = xy.copy()
            return 0.0, None, None

        delta_xy = xy - self.prev_xy
        inst_speed = float(np.linalg.norm(delta_xy) / dt)
        self.speed_ema = 0.72 * self.speed_ema + 0.28 * inst_speed
        heading = None
        heading_delta = None
        if np.linalg.norm(delta_xy) > 1e-4:
            heading = math.degrees(math.atan2(float(delta_xy[1]), float(delta_xy[0])))
            if self.prev_heading_deg is not None:
                heading_delta = _wrap_angle_deg(heading - self.prev_heading_deg)
            self.prev_heading_deg = heading
        self.prev_xy = xy.copy()
        return self.speed_ema, heading, heading_delta


class VideoObserver:
    """Render a live-style annotated video and event log."""

    def __init__(
        self,
        output_video: str | Path,
        output_json: str | Path,
        fps: float,
        display: bool = False,
        max_trail: int = 180,
    ) -> None:
        self.output_video = Path(output_video)
        self.output_json = Path(output_json)
        self.fps = float(fps)
        self.dt = 1.0 / max(1e-6, self.fps)
        self.display = bool(display)
        self.max_trail = int(max_trail)
        self.motion = MotionStateEstimator()
        self.writer: cv2.VideoWriter | None = None
        self.trail: List[tuple[int, np.ndarray, str]] = []
        self.events: List[Dict[str, Any]] = []
        self.current_event: Dict[str, Any] | None = None
        self.frames_rendered = 0
        self.state_counts: Dict[str, int] = {}

    def process(self, frame_bgr: np.ndarray, obs: Dict[str, Any]) -> None:
        motion = self.motion.update(obs, self.dt)
        obs = dict(obs)
        obs["motion_label"] = motion.label
        obs["speed_mps"] = motion.speed_mps
        obs["heading_deg"] = motion.heading_deg
        obs["heading_delta_deg"] = motion.heading_delta_deg

        self._update_events(obs)
        world_xy = _xy(obs.get("world"))
        if world_xy is not None:
            self.trail.append((int(obs["frame_id"]), world_xy, motion.label))
            if len(self.trail) > self.max_trail:
                self.trail = self.trail[-self.max_trail :]

        annotated = self._draw(frame_bgr, obs)
        self._ensure_writer(annotated)
        assert self.writer is not None
        self.writer.write(annotated)
        self.frames_rendered += 1
        self.state_counts[motion.label] = self.state_counts.get(motion.label, 0) + 1

        if self.display:
            cv2.imshow("Skyscouter Observer", annotated)
            cv2.waitKey(1)

    def close(self) -> Dict[str, Any]:
        if self.current_event is not None:
            self.events.append(self.current_event)
            self.current_event = None
        if self.writer is not None:
            self.writer.release()
        if self.display:
            cv2.destroyWindow("Skyscouter Observer")

        report = {
            "schema_version": 1,
            "video": str(self.output_video),
            "frames_rendered": self.frames_rendered,
            "fps": self.fps,
            "state_counts": self.state_counts,
            "events": self.events,
            "integrity": (
                "Visualization-only observer. Labels are derived from tracker/Kalman outputs "
                "and are never fed back into the pipeline."
            ),
        }
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        self.output_json.write_text(json.dumps(_to_jsonable(report), indent=2), encoding="utf-8")
        return report

    def _ensure_writer(self, frame: np.ndarray) -> None:
        if self.writer is not None:
            return
        self.output_video.parent.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.output_video), fourcc, self.fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open observer video writer: {self.output_video}")
        self.writer = writer

    def _update_events(self, obs: Dict[str, Any]) -> None:
        label = str(obs["motion_label"])
        frame_id = int(obs["frame_id"])
        timestamp_ms = int(obs.get("timestamp_ms", 0))
        if self.current_event is None:
            self.current_event = _new_event(label, frame_id, timestamp_ms, obs)
            return
        if self.current_event["label"] == label:
            self.current_event["end_frame"] = frame_id
            self.current_event["end_timestamp_ms"] = timestamp_ms
            self.current_event["max_speed_mps"] = max(
                float(self.current_event["max_speed_mps"]),
                float(obs.get("speed_mps", 0.0)),
            )
            return
        self.events.append(self.current_event)
        self.current_event = _new_event(label, frame_id, timestamp_ms, obs)

    def _draw(self, frame_bgr: np.ndarray, obs: Dict[str, Any]) -> np.ndarray:
        out = frame_bgr.copy()
        h, w = out.shape[:2]
        label = str(obs.get("motion_label", "IDLE"))
        color = _state_color(label)

        bbox = obs.get("bbox")
        if bbox is not None:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
            cv2.drawMarker(
                out,
                (int(round(0.5 * (x1 + x2))), y2),
                (255, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=24,
                thickness=2,
            )
            _text(out, f"{label}  conf={float(obs.get('conf', 0.0)):.2f}", (x1, max(28, y1 - 10)), color, 0.72)

        panel_w = min(650, w - 30)
        cv2.rectangle(out, (18, 18), (18 + panel_w, 214), (12, 18, 24), -1)
        cv2.rectangle(out, (18, 18), (18 + panel_w, 214), color, 2)
        lines = [
            "SKYSCOUTER LIVE OBSERVER",
            f"frame {int(obs.get('frame_id', -1)):04d}  t={int(obs.get('timestamp_ms', 0))} ms",
            f"state={label}  track={obs.get('track_state', '')}  source={_short(obs.get('detector_source', ''), 22)}",
            f"speed={float(obs.get('speed_mps', 0.0)):.2f} m/s  heading={_fmt_angle(obs.get('heading_deg'))}",
            f"world=({_fmt(obs, 'world', 0)}, {_fmt(obs, 'world', 1)}, {_fmt(obs, 'world', 2)}) m",
            f"cam=({_fmt(obs, 'cam', 0)}, {_fmt(obs, 'cam', 1)}, {_fmt(obs, 'cam', 2)}) m",
            f"sigma=({_fmt(obs, 'sigma', 0)}, {_fmt(obs, 'sigma', 1)}, {_fmt(obs, 'sigma', 2)}) m  age={obs.get('occlusion_age', 0)}",
        ]
        y = 48
        for i, line in enumerate(lines):
            scale = 0.62 if i else 0.70
            _text(out, line, (32, y), (230, 240, 245), scale, bg=False)
            y += 24

        self._draw_minimap(out, (w - 350, h - 260, 330, 230), obs)
        self._draw_status_strip(out, label)
        return out

    def _draw_minimap(self, out: np.ndarray, rect: tuple[int, int, int, int], obs: Dict[str, Any]) -> None:
        x, y, w, h = rect
        if x < 0 or y < 0:
            return
        cv2.rectangle(out, (x, y), (x + w, y + h), (14, 20, 26), -1)
        cv2.rectangle(out, (x, y), (x + w, y + h), (210, 215, 220), 1)
        _text(out, "WORLD XY TRAIL", (x + 12, y + 24), (230, 240, 245), 0.55, bg=False)

        points = [p for _, p, _ in self.trail]
        current = _xy(obs.get("world"))
        if current is not None:
            points.append(current)
        if not points:
            return
        arr = np.asarray(points, dtype=np.float64)
        min_xy = np.min(arr, axis=0)
        max_xy = np.max(arr, axis=0)
        span = np.maximum(max_xy - min_xy, np.array([0.5, 0.5], dtype=np.float64))
        pad = 0.18 * span
        min_xy -= pad
        max_xy += pad
        span = np.maximum(max_xy - min_xy, np.array([1e-3, 1e-3], dtype=np.float64))

        def map_pt(p: np.ndarray) -> tuple[int, int]:
            u = x + 20 + int(round((p[0] - min_xy[0]) / span[0] * (w - 40)))
            v = y + h - 24 - int(round((p[1] - min_xy[1]) / span[1] * (h - 55)))
            return u, v

        prev = None
        for _, p, state in self.trail:
            cur = map_pt(p)
            if prev is not None:
                cv2.line(out, prev, cur, _state_color(state), 2)
            prev = cur
        if current is not None:
            cv2.circle(out, map_pt(current), 6, _state_color(str(obs.get("motion_label", ""))), -1)

    def _draw_status_strip(self, out: np.ndarray, label: str) -> None:
        h, w = out.shape[:2]
        color = _state_color(label)
        cv2.rectangle(out, (0, h - 8), (w, h), color, -1)


def _new_event(label: str, frame_id: int, timestamp_ms: int, obs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "label": label,
        "start_frame": frame_id,
        "end_frame": frame_id,
        "start_timestamp_ms": timestamp_ms,
        "end_timestamp_ms": timestamp_ms,
        "max_speed_mps": float(obs.get("speed_mps", 0.0)),
        "start_world": obs.get("world"),
    }


def _xy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape[0] < 2 or not np.isfinite(arr[:2]).all():
        return None
    return arr[:2]


def _wrap_angle_deg(value: float) -> float:
    return float((value + 180.0) % 360.0 - 180.0)


def _state_color(label: str) -> tuple[int, int, int]:
    return {
        "IDLE": (170, 170, 170),
        "STATIONARY": (80, 210, 80),
        "SETTLING": (80, 190, 220),
        "MOVING": (0, 190, 255),
        "CHANGING_DIRECTION": (0, 120, 255),
        "OCCLUDED": (70, 80, 255),
        "REACQUIRED": (255, 180, 40),
        "LOW_CONFIDENCE": (80, 80, 255),
    }.get(label, (220, 220, 220))


def _text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.65,
    bg: bool = True,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    x, y = org
    if bg:
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(img, (x - 4, y - th - 5), (x + tw + 5, y + baseline + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _fmt(obs: Dict[str, Any], key: str, idx: int) -> str:
    value = obs.get(key)
    if value is None:
        return "--"
    try:
        val = float(value[idx])
    except Exception:
        return "--"
    return "--" if not np.isfinite(val) else f"{val:.2f}"


def _fmt_angle(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return "--"
    return "--" if not np.isfinite(val) else f"{val:.0f} deg"


def _short(value: Any, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "~"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return None if not np.isfinite(out) else out
    return value
