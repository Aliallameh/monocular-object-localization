"""Run provenance helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np


def build_run_manifest(
    *,
    command_args: Dict[str, Any],
    video_path: str,
    calib_path: str,
    output_path: str,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "created_unix_s": time.time(),
        "git": git_info(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "libraries": {
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "inputs": {
            "video": file_record(video_path),
            "calibration": file_record(calib_path),
        },
        "command_args": command_args,
        "outputs": {
            "output_csv": file_record(output_path),
            "trajectory_png": file_record("trajectory.png"),
            "trajectory_raw_vs_filtered_png": file_record("trajectory_raw_vs_filtered.png"),
            "trajectory_strict_png": file_record("trajectory_strict.png"),
            "summary_json": file_record("results/summary.json"),
        },
        "run_metrics": {
            "frames_processed": summary.get("frames_processed"),
            "detector_hit_rate": summary.get("detector_hit_rate"),
            "tracker_output_rate": summary.get("tracker_output_rate"),
            "occluded_frames": summary.get("occluded_frames"),
            "p95_processing_ms_per_frame": summary.get("p95_processing_ms_per_frame"),
            "rmse_xy_m": summary.get("metrics", {}).get("rmse_xy_m"),
            "strict_rmse_xy_m": summary.get("strict_metrics", {}).get("rmse_xy_m"),
        },
    }


def write_run_manifest(path: str | Path, manifest: Dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(path)


def file_record(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "bytes": p.stat().st_size,
        "sha256": sha256_file(p),
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def git_info() -> Dict[str, Any]:
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "is_dirty": bool(run_git(["status", "--short"])),
        "remote": run_git(["remote", "get-url", "origin"]),
    }


def run_git(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(["git", *args], check=False, text=True, capture_output=True)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()
