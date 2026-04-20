"""Runtime configuration loading with small YAML/JSON support."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "detector": {
        "backend": "hybrid",
        "device": "auto",
        "conf": 0.05,
        "imgsz": 640,
    },
    "tracker": {
        "bbox_max_age": 35,
        "lk_min_points": 8,
    },
    "kalman": {
        "enabled": True,
        "process_var": 3.0,
        "measurement_var": 0.010,
    },
    "scene_control": {
        "auto_for_input_mp4": False,
        "use_scene_control": False,
        "strict_geometry": False,
        "slow_scene_calibrate": False,
    },
    "stress": {
        "loops": 4,
        "seed": 20260418,
    },
}


def load_runtime_config(path: str | None = None) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if path:
        loaded = _read_config(Path(path))
        _deep_update(cfg, loaded)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    backend = cfg["detector"]["backend"]
    if backend not in {"hybrid", "auto", "yolo_world"}:
        raise ValueError(f"Unsupported detector.backend: {backend}")
    if int(cfg["detector"]["imgsz"]) <= 0:
        raise ValueError("detector.imgsz must be positive")
    if float(cfg["detector"]["conf"]) < 0.0:
        raise ValueError("detector.conf must be non-negative")
    if int(cfg["tracker"]["bbox_max_age"]) < 1:
        raise ValueError("tracker.bbox_max_age must be >= 1")
    if int(cfg["tracker"]["lk_min_points"]) < 1:
        raise ValueError("tracker.lk_min_points must be >= 1")
    if float(cfg["kalman"]["process_var"]) <= 0.0:
        raise ValueError("kalman.process_var must be positive")
    if float(cfg["kalman"]["measurement_var"]) <= 0.0:
        raise ValueError("kalman.measurement_var must be positive")


def _read_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = _parse_simple_yaml(text)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return data


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Parse the small YAML subset used by configs/default.yaml.

    This avoids adding PyYAML to the required assessment dependencies. It
    supports nested mappings via two-space indentation and scalar values.
    """

    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Invalid config line: {raw!r}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> Any:
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none"}:
        return None
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
