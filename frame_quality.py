"""Frame quality diagnostics used for tracking confidence and QA."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameQuality:
    blur_laplacian_var: float
    brightness_mean: float
    contrast_std: float
    is_blurry: bool
    is_low_light: bool


def assess_frame_quality(frame_bgr: np.ndarray) -> FrameQuality:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    return FrameQuality(
        blur_laplacian_var=blur,
        brightness_mean=brightness,
        contrast_std=contrast,
        is_blurry=blur < 18.0,
        is_low_light=brightness < 35.0,
    )
