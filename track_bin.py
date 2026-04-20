"""End-to-end garbage-bin tracking and monocular localization."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from config_utils import load_runtime_config
from detector import Detection, detect_compute, load_detector
from eval_utils import (
    asset_alignment_diagnostics,
    evaluate_bbox_annotations,
    trajectory_smoothness_metrics,
    write_annotation_template,
)
from frame_quality import assess_frame_quality
from localizer import CameraGeometry, PositionKalman, localize_bbox
from observer import VideoObserver
from plots import (
    estimate_stops,
    project_waypoints,
    save_raw_vs_filtered_plot,
    save_trajectory_plot,
)
from provenance import build_run_manifest, write_run_manifest
from qa_utils import build_qa_report, save_qa_frames, write_diagnostics_csv, write_json
from robustness_eval import run_dropout_stress_suite, run_dropout_stress_test
from tracker_utils import BBoxKalmanTracker, LKFlowPropagator


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _banner(video: str, calib: str, backend: str, kalman_on: bool) -> None:
    """Print a visible startup header that won't get lost in frame logs."""
    device_str, device_label = detect_compute()
    w = 68
    sep = "─" * w
    def brow(label: str, value: str) -> str:
        content = f"  {label:<10}: {value}"
        return f"│{content:<{w}}│"

    gpu_value = device_label
    if device_str != "cpu" and backend == "hybrid":
        gpu_value += "  ← pass --backend auto to accelerate"
    lines = [
        f"┌{sep}┐",
        f"│{'  Bin Tracker — Monocular 3-D Localization':^{w}}│",
        f"├{sep}┤",
        brow("Video",    video),
        brow("Calib",    calib),
        brow("Detector", backend),
        brow("Compute",  gpu_value),
        brow("Kalman",   "ON" if kalman_on else "OFF"),
        f"└{sep}┘",
    ]
    print("\n" + "\n".join(lines) + "\n", flush=True)


def _summary_box(summary: dict, metrics: dict, strict_metrics: dict) -> None:
    """Print a framed summary that stands out after 800+ frame lines."""
    W = 68
    sep = "─" * W

    def row(label: str, value: str) -> str:
        content = f"  {label:<22}: {value}"
        return f"│{content:<{W}}│"

    det      = summary["detector_hit_rate"] * 100
    trk      = summary["tracker_output_rate"] * 100
    mean     = summary["mean_processing_ms_per_frame"]
    p95      = summary["p95_processing_ms_per_frame"]
    rmse     = metrics.get("rmse_xy_m", float("nan"))
    occ      = summary["occluded_frames"]
    step_red = metrics.get("smoothness", {}).get("frame_step_std_reduction_pct", float("nan"))

    lines = [
        f"\n┌{sep}┐",
        f"│{'  RESULTS':^{W}}│",
        f"├{sep}┤",
        row("Frames processed",    str(summary["frames_processed"])),
        row("Detector hit rate",   f"{det:.1f}%"),
        row("Tracker output rate", f"{trk:.1f}%"),
        row("Occluded frames",     str(occ)),
        f"├{sep}┤",
        row("Mean latency",        f"{mean:.1f} ms / frame"),
        row("p95 latency",         f"{p95:.1f} ms / frame"),
        f"├{sep}┤",
        row("Waypoint RMSE XY",    f"{rmse:.3f} m  (residual — see README)"),
        row("Kalman smoothing",    f"{step_red:.1f}% frame-step σ reduction" if not __import__('math').isnan(step_red) else "N/A (Kalman off)"),
        f"└{sep}┘",
    ]
    print("\n".join(lines) + "\n", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Skyscouter garbage-bin tracker")
    parser.add_argument("--video", required=True, help="Path to input.mp4")
    parser.add_argument("--calib", required=True, help="Path to calib.json")
    parser.add_argument("--waypoints", default="waypoints.json", help="Path to waypoints.json")
    parser.add_argument("--output", default="results/output.csv", help="CSV output path")
    parser.add_argument("--config", default="", help="Optional YAML/JSON runtime config")
    parser.add_argument("--manifest", default="results/run_manifest.json", help="Run provenance manifest path")
    parser.add_argument("--gpu", action="store_true", help="Request GPU detector backend if available")
    parser.add_argument(
        "--backend",
        choices=["hybrid", "auto", "yolo_world"],
        default=None,
        help="Detector backend: deterministic hybrid fallback, optional YOLO-World, or cascade auto",
    )
    parser.add_argument("--device", default=None, help="Detector device for learned backend: auto, cpu, cuda:0, mps")
    parser.add_argument("--conf", type=float, default=None, help="Learned-detector confidence threshold")
    parser.add_argument("--imgsz", type=int, default=None, help="Learned-detector inference image size")
    parser.add_argument("--no-kalman", action="store_true", help="Disable world-position Kalman smoothing")
    parser.add_argument("--bbox-gt", default="", help="Optional JSON/CSV bbox annotations for local IoU evaluation")
    parser.add_argument(
        "--observer-video",
        default="results/observer_overlay.mp4",
        help="Visualization-only annotated observer video path",
    )
    parser.add_argument(
        "--observer-json",
        default="results/observer_events.json",
        help="Visualization-only observer event report path",
    )
    parser.add_argument("--no-observer", action="store_true", help="Disable annotated observer video generation")
    parser.add_argument("--display", action="store_true", help="Show live observer window while processing")
    parser.add_argument("--max-frames", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    start_wall = time.perf_counter()
    first_track_stdout_ms: float | None = None
    args = parse_args()
    cfg = load_runtime_config(args.config or None)
    detector_cfg = cfg["detector"]
    tracker_cfg = cfg["tracker"]
    kalman_cfg = cfg["kalman"]
    detector_backend = args.backend or str(detector_cfg["backend"])
    detector_device = args.device or str(detector_cfg["device"])
    detector_conf = float(args.conf if args.conf is not None else detector_cfg["conf"])
    detector_imgsz = int(args.imgsz if args.imgsz is not None else detector_cfg["imgsz"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    calib_data = load_json(args.calib)
    camera = CameraGeometry.from_json_dict(calib_data)
    detector = load_detector(
        use_gpu=args.gpu,
        backend=detector_backend,
        device=detector_device,
        conf=detector_conf,
        imgsz=detector_imgsz,
    )
    waypoint_data = load_json(args.waypoints) if Path(args.waypoints).exists() else {"markers": []}
    projected_waypoints = project_waypoints(waypoint_data, camera)

    kalman_on = not (args.no_kalman or not bool(kalman_cfg["enabled"]))
    _banner(args.video, args.calib, detector_backend, kalman_on)

    bbox_tracker = BBoxKalmanTracker(max_age=int(tracker_cfg["bbox_max_age"]))
    flow_tracker = LKFlowPropagator(min_points=int(tracker_cfg["lk_min_points"]))
    pos_filter = (
        None
        if args.no_kalman or not bool(kalman_cfg["enabled"])
        else PositionKalman(
            process_var=float(kalman_cfg["process_var"]),
            measurement_var=float(kalman_cfg["measurement_var"]),
        )
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = camera.fps
    dt = 1.0 / max(1e-6, fps)
    observer = None if args.no_observer else VideoObserver(args.observer_video, args.observer_json, fps, args.display)

    rows: List[Dict[str, Any]] = []
    frame_ids: List[int] = []
    strict_raw_world: List[np.ndarray] = []
    strict_filtered_world: List[np.ndarray] = []
    raw_world: List[np.ndarray] = []
    filtered_world: List[np.ndarray] = []
    sigma_world: List[np.ndarray] = []
    detector_hits = 0
    detector_strong_hits = 0
    flow_assisted_frames = 0
    world_gate_rejections = 0
    tracker_outputs = 0
    occluded_frames = 0
    blurry_frames = 0
    processing_ms: List[float] = []
    frame_states: List[str] = []
    plot_states: List[str] = []
    prev_track_state = "UNCONFIRMED"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_id",
                "timestamp_ms",
                "x1",
                "y1",
                "x2",
                "y2",
                "status",
                "track_state",
                "occlusion_age",
                "detector_source",
                "x_cam",
                "y_cam",
                "z_cam",
                "x_world",
                "y_world",
                "z_world",
                "conf",
                "x_world_raw",
                "y_world_raw",
                "z_world_raw",
                "x_world_filt",
                "y_world_filt",
                "z_world_filt",
                "sigma_x",
                "sigma_y",
                "sigma_z",
                "mu_stationary",
                "gate_d2",
            ]
        )

        frame_id = 0
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and frame_id >= args.max_frames:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            quality = assess_frame_quality(frame)
            if quality.is_blurry:
                blurry_frames += 1
            timestamp_ms = int(round(1000.0 * frame_id / fps))
            detections = detector.detect(frame)
            if detections:
                detector_hits += 1
                if any(det.source != "lk_optical_flow" for det in detections):
                    detector_strong_hits += 1

            flow_bbox, flow_quality, flow_points = flow_tracker.predict(gray)
            if flow_bbox is not None and (not detections or quality.is_blurry):
                flow_assisted_frames += 1
                detections.append(
                    Detection(
                        bbox=flow_bbox,
                        confidence=float(np.clip(0.22 + 0.45 * flow_quality, 0.05, 0.62)),
                        area_px=max(1.0, (flow_bbox[2] - flow_bbox[0]) * (flow_bbox[3] - flow_bbox[1])),
                        source="lk_optical_flow",
                    )
                )

            detections, gate_info = _apply_world_gate(detections, pos_filter, camera, dt, bbox_tracker.age)
            world_gate_rejections += int(gate_info["rejected"])
            track = bbox_tracker.update(detections, dt_frames=1.0, frame=frame)

            if track is None:
                line_ms = int((time.perf_counter() - t0) * 1000)
                print(f"[frame {frame_id:04d}] SEARCHING -- no bin track yet dt={line_ms}ms", flush=True)
                if first_track_stdout_ms is None:
                    first_track_stdout_ms = (time.perf_counter() - start_wall) * 1000.0
                frame_states.append("SEARCHING")
                if observer is not None:
                    observer.process(
                        frame,
                        {
                            "frame_id": frame_id,
                            "timestamp_ms": timestamp_ms,
                            "status": "searching",
                            "track_state": "SEARCHING",
                            "detector_source": "",
                            "conf": 0.0,
                            "occlusion_age": 0,
                            "world": None,
                            "cam": None,
                            "sigma": None,
                            "bbox": None,
                        },
                    )
                writer.writerow(
                    [
                        frame_id,
                        timestamp_ms,
                        "",
                        "",
                        "",
                        "",
                        "searching",
                        "SEARCHING",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "0.000",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
                frame_id += 1
                continue

            tracker_outputs += 1
            if track.matched_detection is not None and track.matched_detection.source == "lk_optical_flow":
                flow_tracker.accept_prediction(gray, track.bbox)
            elif track.status == "detected":
                flow_tracker.update_reference(gray, track.bbox)
            loc = localize_bbox(track.bbox, camera)
            strict_raw_xyz_world = loc.xyz_world

            if pos_filter is None:
                strict_filtered_xyz_world = strict_raw_xyz_world
                sigma_xyz = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
                mu_stationary = 0.0
            elif track.status == "detected":
                meas_var = _measurement_var_for_track(track)
                strict_filtered_xyz_world = pos_filter.update(strict_raw_xyz_world, dt, measurement_var=meas_var)
                sigma_xyz = pos_filter.sigma_xyz()
                if sigma_xyz is None:
                    sigma_xyz = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
                mu_stationary = pos_filter.stationary_probability()
            else:
                predicted = pos_filter.predict(dt)
                strict_filtered_xyz_world = strict_raw_xyz_world if predicted is None else predicted
                sigma_xyz = pos_filter.sigma_xyz()
                if sigma_xyz is None:
                    sigma_xyz = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
                mu_stationary = pos_filter.stationary_probability()

            raw_xyz_world = strict_raw_xyz_world
            out_xyz_world = strict_filtered_xyz_world
            raw_xyz_cam = loc.xyz_cam
            out_xyz_cam = loc.xyz_cam
            xw, yw, zw = out_xyz_world
            xc, yc, zc = out_xyz_cam
            elapsed_ms = float((time.perf_counter() - t0) * 1000.0)
            processing_ms.append(elapsed_ms)

            detector_source = track.matched_detection.source if track.matched_detection is not None else "tracker_prediction"
            track_state = _track_state(prev_track_state, track.status, mu_stationary)
            prev_track_state = track_state
            frame_states.append(track_state)
            plot_states.append(track_state)

            if track.status == "detected":
                print(
                    f"[frame {frame_id:04d}] bin @ world "
                    f"({xw:.2f}, {yw:.2f}, {zw:.2f}) m "
                    f"conf={track.confidence:.2f} state={track_state} dt={int(elapsed_ms)}ms",
                    flush=True,
                )
            else:
                occluded_frames += 1
                print(
                    f"[frame {frame_id:04d}] OCCLUDED -- last known "
                    f"({xw:.2f}, {yw:.2f}, {zw:.2f}) m age={track.age}fr "
                    f"conf={track.confidence:.2f} state={track_state} dt={int(elapsed_ms)}ms",
                    flush=True,
                )
            if first_track_stdout_ms is None:
                first_track_stdout_ms = (time.perf_counter() - start_wall) * 1000.0

            if observer is not None:
                observer.process(
                    frame,
                    {
                        "frame_id": frame_id,
                        "timestamp_ms": timestamp_ms,
                        "status": track.status,
                        "track_state": track_state,
                        "detector_source": detector_source,
                        "conf": float(track.confidence),
                        "occlusion_age": int(track.age),
                        "world": out_xyz_world.tolist(),
                        "cam": out_xyz_cam.tolist(),
                        "sigma": sigma_xyz.tolist(),
                        "bbox": [float(v) for v in track.bbox],
                    },
                )

            writer.writerow(
                [
                    frame_id,
                    timestamp_ms,
                    f"{track.bbox[0]:.2f}",
                    f"{track.bbox[1]:.2f}",
                    f"{track.bbox[2]:.2f}",
                    f"{track.bbox[3]:.2f}",
                    track.status,
                    track_state,
                    int(track.age),
                    detector_source,
                    f"{xc:.4f}",
                    f"{yc:.4f}",
                    f"{zc:.4f}",
                    f"{xw:.4f}",
                    f"{yw:.4f}",
                    f"{zw:.4f}",
                    f"{track.confidence:.3f}",
                    f"{raw_xyz_world[0]:.4f}",
                    f"{raw_xyz_world[1]:.4f}",
                    f"{raw_xyz_world[2]:.4f}",
                    f"{out_xyz_world[0]:.4f}",
                    f"{out_xyz_world[1]:.4f}",
                    f"{out_xyz_world[2]:.4f}",
                    _fmt_float(sigma_xyz[0]),
                    _fmt_float(sigma_xyz[1]),
                    _fmt_float(sigma_xyz[2]),
                    f"{mu_stationary:.3f}",
                    _fmt_float(gate_info["best_d2"]),
                ]
            )

            frame_ids.append(frame_id)
            strict_raw_world.append(strict_raw_xyz_world.copy())
            strict_filtered_world.append(strict_filtered_xyz_world.copy())
            raw_world.append(raw_xyz_world.copy())
            filtered_world.append(out_xyz_world.copy())
            sigma_world.append(sigma_xyz.copy())
            rows.append(
                {
                    "frame_id": frame_id,
                    "timestamp_ms": timestamp_ms,
                    "status": track.status,
                    "track_state": track_state,
                    "occlusion_age": int(track.age),
                    "conf": float(track.confidence),
                    "bbox": [float(v) for v in track.bbox],
                    "detector_source": detector_source,
                    "world_gate_best_d2": gate_info["best_d2"],
                    "world_gate_rejected": gate_info["rejected"],
                    "raw_cam": raw_xyz_cam.tolist(),
                    "filtered_cam": out_xyz_cam.tolist(),
                    "raw_world": raw_xyz_world.tolist(),
                    "filtered_world": out_xyz_world.tolist(),
                    "sigma_world": sigma_xyz.tolist(),
                    "mu_stationary": float(mu_stationary),
                    "strict_raw_world": strict_raw_xyz_world.tolist(),
                    "strict_filtered_world": strict_filtered_xyz_world.tolist(),
                    "height_cam": loc.xyz_cam_height.tolist(),
                    "ground_world": loc.xyz_world_ground.tolist(),
                    "height_depth_delta_m": loc.depth_delta_m,
                    "fallback": bool(loc.used_fallback),
                    "blur_laplacian_var": quality.blur_laplacian_var,
                    "brightness_mean": quality.brightness_mean,
                    "contrast_std": quality.contrast_std,
                    "is_blurry": quality.is_blurry,
                    "is_low_light": quality.is_low_light,
                    "flow_points": int(flow_points),
                    "flow_quality": float(flow_quality),
                }
            )
            frame_id += 1

    cap.release()
    observer_report: Dict[str, Any] | None = observer.close() if observer is not None else None

    if not frame_ids:
        raise RuntimeError("No frames were processed into a valid track")

    frame_arr = np.asarray(frame_ids, dtype=np.int32)
    strict_raw_arr = np.vstack(strict_raw_world)
    strict_filt_arr = np.vstack(strict_filtered_world)
    raw_arr = np.vstack(raw_world)
    filt_arr = np.vstack(filtered_world)
    sigma_arr = np.vstack(sigma_world) if sigma_world else None

    stops, metrics = estimate_stops(frame_arr, raw_arr, filt_arr, projected_waypoints, fps)
    smoothness_metrics = trajectory_smoothness_metrics(raw_arr, filt_arr)
    metrics["smoothness"] = smoothness_metrics
    save_trajectory_plot("trajectory.png", frame_arr, raw_arr, filt_arr, projected_waypoints, stops, metrics)
    save_raw_vs_filtered_plot("trajectory_raw_vs_filtered.png", frame_arr, raw_arr, filt_arr, plot_states, sigma_arr)

    strict_stops, strict_metrics = estimate_stops(
        frame_arr, strict_raw_arr, strict_filt_arr, projected_waypoints, fps
    )
    strict_smoothness_metrics = trajectory_smoothness_metrics(strict_raw_arr, strict_filt_arr)
    strict_metrics["smoothness"] = strict_smoothness_metrics
    save_trajectory_plot(
        "trajectory_strict.png",
        frame_arr,
        strict_raw_arr,
        strict_filt_arr,
        projected_waypoints,
        strict_stops,
        strict_metrics,
    )

    for stale in (
        "results/output_scene_control.csv",
        "results/output_waypoint_calibrated.csv",
        "trajectory_scene_control.png",
        "trajectory_waypoint_calibrated.png",
    ):
        try:
            Path(stale).unlink()
        except FileNotFoundError:
            pass

    elapsed = np.asarray(processing_ms, dtype=np.float64)
    robustness_report = run_dropout_stress_test(args.video, detector, rows)
    occlusion_stress_suite = run_dropout_stress_suite(args.video, detector, rows)
    bbox_gt_path = args.bbox_gt or None
    bbox_eval_report = evaluate_bbox_annotations(bbox_gt_path, rows)
    annotation_template_path = write_annotation_template("results/bbox_annotation_template.csv", rows)
    asset_alignment_report = asset_alignment_diagnostics(strict_stops)
    detector_metadata = detector.metadata() if hasattr(detector, "metadata") else {"backend": detector.__class__.__name__}
    summary = {
        "video": str(args.video),
        "detector_backend": detector_metadata,
        "runtime_config": cfg,
        "frames_processed": int(frame_id),
        "fps_reported": float(fps),
        "detector_hit_rate": float(detector_hits / max(1, frame_id)),
        "detector_strong_hit_rate": float(detector_strong_hits / max(1, frame_id)),
        "flow_assisted_frames": int(flow_assisted_frames),
        "world_gate_rejections": int(world_gate_rejections),
        "tracker_output_rate": float(tracker_outputs / max(1, frame_id)),
        "occluded_frames": int(occluded_frames),
        "blurry_frames": int(blurry_frames),
        "track_state_counts": dict(Counter(frame_states)),
        "mean_processing_ms_per_frame": float(np.mean(elapsed)) if len(elapsed) else float("nan"),
        "p95_processing_ms_per_frame": float(np.percentile(elapsed, 95)) if len(elapsed) else float("nan"),
        "max_processing_ms_per_frame": float(np.max(elapsed)) if len(elapsed) else float("nan"),
        "first_track_stdout_ms_from_python": (
            float(first_track_stdout_ms) if first_track_stdout_ms is not None else None
        ),
        "waypoints": {
            name: {
                "color": wp["color"],
                "approx_frame": wp["approx_frame"],
                "world_ground": np.asarray(wp["world_ground"]).tolist(),
                "world_centroid": np.asarray(wp["world_centroid"]).tolist(),
            }
            for name, wp in projected_waypoints.items()
        },
        "stops": stops,
        "metrics": metrics,
        "strict_stops": strict_stops,
        "strict_metrics": strict_metrics,
        "bbox_evaluation": bbox_eval_report,
        "bbox_annotation_template": annotation_template_path,
        "asset_alignment_diagnostics": asset_alignment_report,
        "robustness_stress_test": robustness_report,
        "occlusion_stress_suite": occlusion_stress_suite,
        "observer": (
            {
                "enabled": True,
                "video": str(args.observer_video),
                "events_json": str(args.observer_json),
                "frames_rendered": observer_report.get("frames_rendered") if observer_report else None,
                "state_counts": observer_report.get("state_counts") if observer_report else {},
            }
            if observer_report is not None
            else {"enabled": False}
        ),
    }

    os.makedirs("results", exist_ok=True)
    diagnostics_path = "results/diagnostics.csv"
    qa_report_path = "results/qa_report.json"
    robustness_path = "results/robustness_report.json"
    occlusion_suite_path = "results/occlusion_stress_suite.json"
    bbox_eval_path = "results/bbox_eval.json"
    asset_alignment_path = "results/asset_alignment_report.json"
    observer_json_path = str(args.observer_json) if observer_report is not None else None
    observer_video_path = str(args.observer_video) if observer_report is not None else None
    qa_frame_dir = "results/qa_frames"
    write_diagnostics_csv(diagnostics_path, rows)
    write_json(robustness_path, robustness_report)
    write_json(occlusion_suite_path, occlusion_stress_suite)
    write_json(bbox_eval_path, bbox_eval_report)
    write_json(asset_alignment_path, asset_alignment_report)
    try:
        Path("results/scene_control_report.json").unlink()
    except FileNotFoundError:
        pass
    qa_frames = save_qa_frames(qa_frame_dir, args.video, rows, projected_waypoints, stops)
    qa_report = build_qa_report(
        args.video,
        output_path,
        rows,
        summary,
        waypoint_data,
        projected_waypoints,
        camera,
        first_track_stdout_ms,
    )
    write_json(qa_report_path, qa_report)
    summary["qa_artifacts"] = {
        "diagnostics_csv": diagnostics_path,
        "qa_report_json": qa_report_path,
        "robustness_report_json": robustness_path,
        "occlusion_stress_suite_json": occlusion_suite_path,
        "bbox_eval_json": bbox_eval_path,
        "asset_alignment_report_json": asset_alignment_path,
        "observer_events_json": observer_json_path,
        "observer_overlay_video": observer_video_path,
        "bbox_annotation_template_csv": annotation_template_path,
        "annotated_frames": qa_frames,
        "strict_trajectory": "trajectory_strict.png",
        "raw_vs_filtered_trajectory": "trajectory_raw_vs_filtered.png",
    }
    manifest_path = write_run_manifest(
        args.manifest,
        build_run_manifest(
            command_args=vars(args),
            video_path=args.video,
            calib_path=args.calib,
            output_path=str(output_path),
            summary=summary,
        ),
    )
    summary["qa_artifacts"]["run_manifest_json"] = manifest_path
    with open("results/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _summary_box(summary, metrics, strict_metrics)
    print(f"[output] {output_path}", flush=True)
    print(f"[output] trajectory.png  trajectory_strict.png  trajectory_raw_vs_filtered.png", flush=True)
    if observer_report is not None:
        print(f"[output] {args.observer_video}  {args.observer_json}", flush=True)
    print(f"[output] {diagnostics_path}  {qa_report_path}  {manifest_path}", flush=True)


def _measurement_var_for_track(track: Any) -> float:
    source = track.matched_detection.source if track.matched_detection is not None else "tracker_prediction"
    conf = float(np.clip(track.confidence, 0.0, 1.0))
    if source == "lk_optical_flow":
        return 0.060
    if source == "edge_shape":
        return 0.045
    if source.startswith("yolo_world"):
        return 0.010 + 0.030 * (1.0 - conf) ** 2
    return 0.008 + 0.055 * (1.0 - conf) ** 2


def _candidate_measurement_var(det: Detection) -> float:
    conf = float(np.clip(det.confidence, 0.0, 1.0))
    if det.source == "lk_optical_flow":
        return 0.080
    if det.source == "edge_shape":
        return 0.060
    if det.source == "motion_foreground":
        return 0.070
    if det.source.startswith("yolo_world"):
        return 0.014 + 0.035 * (1.0 - conf) ** 2
    return 0.012 + 0.060 * (1.0 - conf) ** 2


def _apply_world_gate(
    detections: List[Detection],
    pos_filter: PositionKalman | None,
    camera: CameraGeometry,
    dt: float,
    occlusion_age: int = 0,
) -> tuple[List[Detection], Dict[str, float]]:
    """Reject candidates whose world position is physically implausible.

    The gate is intentionally advisory: if every candidate fails, the closest
    one is retained unless it is extremely far from the predicted state. That
    preserves reacquisition opportunities while still blocking obvious
    distractors.
    """

    if not detections or pos_filter is None or not pos_filter.initialized:
        return detections, {"best_d2": float("nan"), "rejected": 0.0}

    accepted: List[Detection] = []
    scored: List[tuple[float, Detection]] = []
    for det in detections:
        try:
            xyz = localize_bbox(det.bbox, camera).xyz_world
            d2 = pos_filter.innovation_mahalanobis(xyz, dt, _candidate_measurement_var(det))
        except Exception:
            d2 = float("inf")
        scored.append((float(d2), det))

        # 3-DOF 99.73% chi-square is about 14.2. The looser thresholds below
        # account for bbox-bottom noise and the known imperfect camera scale.
        if det.source == "lk_optical_flow":
            threshold = 55.0
        elif det.source == "edge_shape":
            threshold = 45.0
        elif det.source == "motion_foreground":
            threshold = 48.0
        else:
            threshold = 38.0
        if d2 <= threshold:
            accepted.append(det)

    scored.sort(key=lambda item: item[0])
    best_d2 = scored[0][0] if scored else float("nan")
    if not accepted and occlusion_age >= 20:
        strong = [
            item for item in scored if item[1].source != "lk_optical_flow" and item[1].confidence >= 0.24
        ]
        if strong:
            strong.sort(key=lambda item: item[1].confidence, reverse=True)
            accepted.append(strong[0][1])
            return accepted, {"best_d2": float(best_d2), "rejected": float(len(detections) - 1)}
    if not accepted and scored and np.isfinite(best_d2) and best_d2 <= 95.0:
        accepted.append(scored[0][1])
    return accepted, {"best_d2": float(best_d2), "rejected": float(len(detections) - len(accepted))}


def _track_state(previous_state: str, status: str, mu_stationary: float) -> str:
    if status == "detected":
        if previous_state in {"OCCLUDED", "LOST", "SEARCHING"}:
            return "REACQUIRED"
        if mu_stationary >= 0.72:
            return "STATIONARY"
        return "CONFIRMED"
    if status == "occluded":
        return "OCCLUDED"
    return status.upper()


def _fmt_float(value: float) -> str:
    return "" if not np.isfinite(value) else f"{float(value):.4f}"


if __name__ == "__main__":
    main()
