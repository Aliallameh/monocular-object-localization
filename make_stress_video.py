"""Generate a longer, harsher stress video from the assessment clip.

The output is not meant to be photorealistic training data. It is a deterministic
QA asset: repeated source frames with controlled blur, noise, occluding
person-like silhouettes, brightness shifts, camera shake, and blue distractors.
The manifest records what was applied to every generated frame.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


BBox = Tuple[float, float, float, float]
SAFE_WAYPOINT_FRAMES = (45, 195, 345)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reproducible harsh stress-test video")
    parser.add_argument("--video", default="input.mp4", help="Source video")
    parser.add_argument("--bbox-csv", default="results/output.csv", help="Existing tracker CSV with x1,y1,x2,y2")
    parser.add_argument("--output", default="stress_long_harsh.mp4", help="Output MP4 path")
    parser.add_argument("--manifest", default="results/stress_long_manifest.csv", help="Per-frame perturbation manifest")
    parser.add_argument("--loops", type=int, default=4, help="Number of source-video passes")
    parser.add_argument("--seed", type=int, default=20260418, help="Deterministic RNG seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    bbox_trace = load_bbox_trace(args.bbox_csv)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    out_path = Path(args.output)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer for {out_path}")

    frame_cache: Dict[int, np.ndarray] = {}
    total = frame_count * args.loops
    with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
        manifest = csv.writer(mf)
        manifest.writerow(
            [
                "out_frame",
                "src_frame",
                "loop",
                "brightness_gain",
                "brightness_bias",
                "noise_sigma",
                "gaussian_blur",
                "motion_blur",
                "shake_dx",
                "shake_dy",
                "target_occluders",
                "distractor_people",
                "blue_distractors",
                "protected_waypoint_window",
            ]
        )

        for out_frame in range(total):
            loop = out_frame // frame_count
            local = out_frame % frame_count
            src_frame = local if loop % 2 == 0 else frame_count - 1 - local
            frame = read_frame(cap, args.video, src_frame, frame_cache)
            bbox = bbox_trace.get(src_frame)
            frame, meta = perturb_frame(frame, bbox, out_frame, src_frame, loop, rng, width, height)
            writer.write(frame)
            manifest.writerow(
                [
                    out_frame,
                    src_frame,
                    loop,
                    f"{meta['brightness_gain']:.4f}",
                    f"{meta['brightness_bias']:.2f}",
                    f"{meta['noise_sigma']:.2f}",
                    int(meta["gaussian_blur"]),
                    int(meta["motion_blur"]),
                    f"{meta['shake_dx']:.2f}",
                    f"{meta['shake_dy']:.2f}",
                    int(meta["target_occluders"]),
                    int(meta["distractor_people"]),
                    int(meta["blue_distractors"]),
                    int(meta["protected_waypoint_window"]),
                ]
            )
            if out_frame == 0 or (out_frame + 1) % 250 == 0 or out_frame + 1 == total:
                print(f"[stress-video] wrote {out_frame + 1}/{total} frames", flush=True)

    writer.release()
    cap.release()

    summary = {
        "source_video": args.video,
        "output_video": str(out_path),
        "manifest": str(manifest_path),
        "source_frames": frame_count,
        "output_frames": total,
        "fps": fps,
        "duration_seconds": total / fps,
        "loops": args.loops,
        "seed": args.seed,
        "purpose": (
            "Reproducible QA stress video with synthetic occlusion, blur, noise, "
            "brightness shifts, camera shake, and distractors. Not photorealistic GT."
        ),
    }
    summary_path = manifest_path.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[stress-video] wrote {out_path}", flush=True)
    print(f"[stress-video] wrote {manifest_path} and {summary_path}", flush=True)


def load_bbox_trace(path: str) -> Dict[int, BBox]:
    trace: Dict[int, BBox] = {}
    csv_path = Path(path)
    if not csv_path.exists():
        return trace
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_id = int(row["frame_id"])
                vals = tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))
            except Exception:
                continue
            if vals[2] > vals[0] and vals[3] > vals[1]:
                trace[frame_id] = vals  # type: ignore[assignment]
    return trace


def read_frame(cap: cv2.VideoCapture, video: str, frame_id: int, cache: Dict[int, np.ndarray]) -> np.ndarray:
    if frame_id in cache:
        return cache[frame_id].copy()
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_id} from {video}")
    if len(cache) < 96:
        cache[frame_id] = frame.copy()
    return frame


def perturb_frame(
    frame: np.ndarray,
    bbox: BBox | None,
    out_frame: int,
    src_frame: int,
    loop: int,
    rng: np.random.Generator,
    width: int,
    height: int,
) -> Tuple[np.ndarray, Dict[str, float | int | bool]]:
    protected = loop == 0 and any(abs(src_frame - wp) <= 8 for wp in SAFE_WAYPOINT_FRAMES)
    severity = min(1.0, 0.18 + 0.24 * loop)
    if protected:
        severity = min(severity, 0.12)

    gain = 1.0 + 0.16 * severity * np.sin(0.027 * out_frame) + rng.normal(0.0, 0.025 * severity)
    bias = 18.0 * severity * np.sin(0.013 * out_frame + 1.7) + rng.normal(0.0, 3.0 * severity)
    out = cv2.convertScaleAbs(frame, alpha=float(gain), beta=float(bias))

    shake_dx = 0.0
    shake_dy = 0.0
    if loop >= 1:
        shake_dx = float(rng.normal(0.0, 2.0 + 2.0 * severity))
        shake_dy = float(rng.normal(0.0, 1.4 + 1.6 * severity))
        m = np.array([[1.0, 0.0, shake_dx], [0.0, 1.0, shake_dy]], dtype=np.float32)
        out = cv2.warpAffine(out, m, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    target_occ = 0
    if bbox is not None and not protected:
        phase = (out_frame * 7 + loop * 41) % 120
        if loop >= 1 and 16 <= phase <= 52:
            alpha = 0.60 + 0.17 * severity
            cover = 0.22 + 0.42 * severity
            draw_person_occluder(out, bbox, phase / 120.0, alpha=alpha, cover=cover)
            target_occ += 1
        if loop >= 2 and 76 <= phase <= 102:
            shifted = shift_bbox(bbox, -0.34, 0.04)
            draw_person_occluder(out, shifted, phase / 120.0, alpha=0.55, cover=0.30)
            target_occ += 1

    distractor_people = 0
    for i in range(1 + int(loop >= 2)):
        if loop >= 1 and ((out_frame + 37 * i) % (52 - 5 * loop)) < (18 + 3 * loop):
            x = int((width * (0.16 + 0.22 * i) + 5.5 * out_frame * (1 + i)) % width)
            y = int(height * (0.43 + 0.08 * ((i + loop) % 3)))
            draw_free_person(out, x, y, scale=0.70 + 0.18 * loop, alpha=0.42 + 0.08 * loop)
            distractor_people += 1

    blue_distractors = 0
    if loop >= 2:
        for i in range(2):
            if (out_frame + 19 * i) % 90 < 46:
                draw_blue_distractor(out, out_frame, i, width, height, severity)
                blue_distractors += 1

    gaussian_blur = 0
    if loop >= 1 and not protected:
        if out_frame % 95 < 26 + 8 * loop:
            k = 3 + 2 * min(3, loop)
            out = cv2.GaussianBlur(out, (k, k), 0)
            gaussian_blur = k

    motion_blur = 0
    if loop >= 2 and not protected and out_frame % 140 < 48:
        motion_blur = 9 + 2 * loop
        out = apply_motion_blur(out, motion_blur, horizontal=(out_frame // 20) % 2 == 0)

    noise_sigma = 4.0 + 11.0 * severity
    if protected:
        noise_sigma = min(noise_sigma, 4.0)
    noise = rng.normal(0.0, noise_sigma, out.shape).astype(np.float32)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if loop >= 3 and not protected and out_frame % 210 < 72:
        out = jpeg_like_degrade(out, scale=0.52)

    meta: Dict[str, float | int | bool] = {
        "brightness_gain": float(gain),
        "brightness_bias": float(bias),
        "noise_sigma": float(noise_sigma),
        "gaussian_blur": int(gaussian_blur),
        "motion_blur": int(motion_blur),
        "shake_dx": float(shake_dx),
        "shake_dy": float(shake_dy),
        "target_occluders": int(target_occ),
        "distractor_people": int(distractor_people),
        "blue_distractors": int(blue_distractors),
        "protected_waypoint_window": bool(protected),
    }
    return out, meta


def draw_person_occluder(frame: np.ndarray, bbox: BBox, phase: float, alpha: float, cover: float) -> None:
    x1, y1, x2, y2 = bbox
    w = max(10.0, x2 - x1)
    h = max(10.0, y2 - y1)
    cx = int(x1 + w * (0.12 + 0.76 * phase))
    cy = int(y1 + h * 0.52)
    person_h = int(h * (1.28 + cover))
    person_w = int(w * (0.18 + 0.34 * cover))
    draw_free_person(frame, cx, cy - int(0.08 * h), scale=max(0.35, person_h / 290.0), alpha=alpha, width_hint=person_w)


def draw_free_person(
    frame: np.ndarray,
    cx: int,
    cy: int,
    scale: float,
    alpha: float,
    width_hint: int | None = None,
) -> None:
    overlay = frame.copy()
    width = width_hint if width_hint is not None else int(64 * scale)
    body_h = int(165 * scale)
    head_r = max(7, int(18 * scale))
    shoulder = max(12, width // 2)
    color = (26, 29, 31)
    edge = (54, 56, 58)

    head_c = (cx, cy - body_h // 2 - head_r)
    cv2.circle(overlay, head_c, head_r, color, -1, lineType=cv2.LINE_AA)
    body = np.array(
        [
            [cx - shoulder, cy - body_h // 2],
            [cx + shoulder, cy - body_h // 2],
            [cx + width // 3, cy + body_h // 3],
            [cx + width // 6, cy + body_h // 2],
            [cx - width // 6, cy + body_h // 2],
            [cx - width // 3, cy + body_h // 3],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(overlay, [body], color, lineType=cv2.LINE_AA)
    cv2.line(overlay, (cx - shoulder, cy - body_h // 3), (cx - width, cy + body_h // 6), edge, max(4, int(7 * scale)))
    cv2.line(overlay, (cx + shoulder, cy - body_h // 3), (cx + width, cy + body_h // 7), edge, max(4, int(7 * scale)))
    leg_y = cy + body_h // 2
    cv2.line(overlay, (cx - width // 7, leg_y), (cx - width // 2, leg_y + int(64 * scale)), color, max(5, int(9 * scale)))
    cv2.line(overlay, (cx + width // 7, leg_y), (cx + width // 2, leg_y + int(64 * scale)), color, max(5, int(9 * scale)))
    cv2.addWeighted(overlay, float(np.clip(alpha, 0.0, 1.0)), frame, 1.0 - float(np.clip(alpha, 0.0, 1.0)), 0, dst=frame)


def draw_blue_distractor(frame: np.ndarray, out_frame: int, idx: int, width: int, height: int, severity: float) -> None:
    overlay = frame.copy()
    w = int(width * (0.055 + 0.012 * idx))
    h = int(height * (0.115 + 0.018 * idx))
    x = int((width * (0.68 + 0.11 * idx) - 3.3 * out_frame * (idx + 1)) % width)
    y = int(height * (0.54 + 0.11 * ((out_frame // 75 + idx) % 2)))
    color = (190, 76, 16)
    cv2.rectangle(overlay, (x, y), (min(width - 1, x + w), min(height - 1, y + h)), color, -1)
    cv2.rectangle(overlay, (x - 3, y - 10), (min(width - 1, x + w + 3), y + 3), color, -1)
    cv2.circle(overlay, (x + w // 5, min(height - 1, y + h + 5)), max(3, w // 12), (20, 20, 20), -1)
    cv2.circle(overlay, (x + 4 * w // 5, min(height - 1, y + h + 5)), max(3, w // 12), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.28 + 0.18 * severity, frame, 0.72 - 0.18 * severity, 0, dst=frame)


def apply_motion_blur(frame: np.ndarray, k: int, horizontal: bool) -> np.ndarray:
    k = max(3, int(k) | 1)
    kernel = np.zeros((k, k), dtype=np.float32)
    if horizontal:
        kernel[k // 2, :] = 1.0 / k
    else:
        kernel[:, k // 2] = 1.0 / k
    return cv2.filter2D(frame, -1, kernel)


def jpeg_like_degrade(frame: np.ndarray, scale: float) -> np.ndarray:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(2, int(w * scale)), max(2, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def shift_bbox(bbox: BBox, frac_x: float, frac_y: float) -> BBox:
    x1, y1, x2, y2 = bbox
    dx = (x2 - x1) * frac_x
    dy = (y2 - y1) * frac_y
    return x1 + dx, y1 + dy, x2 + dx, y2 + dy


if __name__ == "__main__":
    main()
