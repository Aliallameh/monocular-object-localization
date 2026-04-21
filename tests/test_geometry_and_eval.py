from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
import subprocess

import numpy as np

from config_utils import load_runtime_config
from eval_utils import asset_alignment_diagnostics, evaluate_bbox_annotations, evaluate_contact_points
from localizer import BIN_HEIGHT_M, CameraGeometry, build_camera_to_world, localize_bbox
from observer import MotionStateEstimator
from tracker_utils import bbox_iou


class GeometryTests(unittest.TestCase):
    def test_camera_transform_sign_conventions(self) -> None:
        r_cw, t_cw = build_camera_to_world(1.35, np.deg2rad(-15.0))
        optical_world = r_cw @ np.array([0.0, 0.0, 1.0])
        self.assertGreater(optical_world[0], 0.0)
        self.assertLess(optical_world[2], 0.0)
        self.assertTrue(np.allclose(t_cw, [0.0, 0.0, 1.35]))

    def test_ground_intersection_and_centroid_height(self) -> None:
        data = {
            "K": [[1402.5, 0.0, 960.0], [0.0, 1402.5, 540.0], [0.0, 0.0, 1.0]],
            "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
            "camera_height_m": 1.35,
            "camera_tilt_deg": -15.0,
            "fps": 30.0,
            "image_width_px": 1920,
            "image_height_px": 1080,
        }
        camera = CameraGeometry.from_json_dict(data)
        point = camera.ground_intersection_from_pixel(960.0, 900.0)
        self.assertGreater(point[0], 0.0)
        self.assertAlmostEqual(point[2], 0.0, places=9)
        centroid = point + np.array([0.0, 0.0, BIN_HEIGHT_M / 2.0])
        self.assertAlmostEqual(centroid[2], 0.325, places=9)

    def test_synthetic_bin_localization_at_known_distances(self) -> None:
        """Round-trip geometry: forward-project a bin at known world positions,
        then localize_bbox() must recover x within 0.25 m.
        No video GT needed — this is a closed-form identity check."""
        calib = {
            "K": [[1402.5, 0.0, 960.0], [0.0, 1402.5, 540.0], [0.0, 0.0, 1.0]],
            "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
            "camera_height_m": 1.35,
            "camera_tilt_deg": -15.0,
            "fps": 30.0,
            "image_width_px": 1920,
            "image_height_px": 1080,
        }
        camera = CameraGeometry.from_json_dict(calib)

        def proj(xyz: np.ndarray) -> tuple:
            K = camera.K
            return K[0, 0] * xyz[0] / xyz[2] + K[0, 2], K[1, 1] * xyz[1] / xyz[2] + K[1, 2]

        for x_true in [3.0, 5.0, 7.0, 9.0]:
            cam_bot = camera.world_to_cam(np.array([x_true, 0.0, 0.0]))
            cam_top = camera.world_to_cam(np.array([x_true, 0.0, BIN_HEIGHT_M]))
            u_bot, v_bot = proj(cam_bot)
            u_top, v_top = proj(cam_top)
            u_cen = (u_bot + u_top) / 2.0
            bbox = (u_cen - 50, v_top, u_cen + 50, v_bot)
            loc = localize_bbox(bbox, camera)
            error = abs(loc.xyz_world[0] - x_true)
            self.assertLess(
                error, 0.25,
                f"Localization error {error:.4f} m > 0.25 m at x_true={x_true} m — geometry broken",
            )


class EvaluationTests(unittest.TestCase):
    def test_bbox_iou(self) -> None:
        self.assertAlmostEqual(bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)

    def test_bbox_annotation_evaluator_csv(self) -> None:
        rows = [
            {"frame_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0], "status": "detected", "detector_source": "blue"},
            {"frame_id": 2, "bbox": [10.0, 10.0, 20.0, 20.0], "status": "occluded", "detector_source": "lk_optical_flow"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            gt_path = Path(tmp) / "gt.csv"
            with gt_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["frame_id", "x1", "y1", "x2", "y2", "review_status", "occluded", "visibility"])
                writer.writerow([1, 0, 0, 10, 10, "ok", 0, 1.0])
                writer.writerow([2, 12, 12, 22, 22, "ok", 1, 0.5])
                writer.writerow([3, 0, 0, 1, 1, "skip", 1, 0.1])
            report = evaluate_bbox_annotations(gt_path, rows)
        self.assertTrue(report["enabled"])
        self.assertEqual(report["gt_frames"], 2)
        self.assertEqual(report["gt_occluded_frames"], 1)
        self.assertEqual(report["matched_frames"], 2)
        self.assertEqual(report["occluded_frames"]["matched_frames"], 1)
        self.assertEqual(report["detector_source_counts_on_gt_occluded"]["lk_optical_flow"], 1)
        self.assertGreater(report["mean_iou"], 0.5)

    def test_bbox_annotation_evaluator_json(self) -> None:
        rows = [{"frame_id": 7, "bbox": [1.0, 2.0, 11.0, 12.0]}]
        with tempfile.TemporaryDirectory() as tmp:
            gt_path = Path(tmp) / "gt.json"
            gt_path.write_text(json.dumps({"frames": [{"frame_id": 7, "x1": 1, "y1": 2, "x2": 11, "y2": 12}]}))
            report = evaluate_bbox_annotations(gt_path, rows)
        self.assertTrue(report["passes_hidden_contract_proxy"])

    def test_contact_point_evaluator_is_validation_only(self) -> None:
        calib = {
            "K": [[1402.5, 0.0, 960.0], [0.0, 1402.5, 540.0], [0.0, 0.0, 1.0]],
            "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
            "camera_height_m": 1.35,
            "camera_tilt_deg": -15.0,
            "fps": 30.0,
            "image_width_px": 1920,
            "image_height_px": 1080,
        }
        camera = CameraGeometry.from_json_dict(calib)
        ground = camera.ground_intersection_from_pixel(960.0, 900.0)
        centroid = ground + np.array([0.0, 0.0, BIN_HEIGHT_M / 2.0])
        rows = [
            {
                "frame_id": 10,
                "bbox": [900.0, 700.0, 1020.0, 900.0],
                "ground_world": ground.tolist(),
                "raw_world": centroid.tolist(),
                "filtered_world": centroid.tolist(),
                "detector_source": "unit_test",
                "track_state": "CONFIRMED",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            gt_path = Path(tmp) / "contact.csv"
            gt_path.write_text(
                "frame_id,x_contact_px,y_contact_px,review_status,occluded,visibility\n"
                "10,960,900,ok,0,1.0\n",
                encoding="utf-8",
            )
            report = evaluate_contact_points(gt_path, rows, camera)
        self.assertTrue(report["enabled"])
        self.assertEqual(report["matched_frames"], 1)
        self.assertAlmostEqual(report["mean_pixel_error"], 0.0)
        self.assertAlmostEqual(report["mean_ground_xy_error_m"], 0.0)
        self.assertIn("not used to calibrate", report["interpretation"])

    def test_import_cvat_xml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            xml_path = tmp_path / "annotations.xml"
            out_path = tmp_path / "out.csv"
            xml_path.write_text(
                """<annotations>
                <track id="3" label="object">
                  <box frame="5" outside="0" xtl="1" ytl="2" xbr="11" ybr="22" occluded="0" />
                  <box frame="6" outside="1" xtl="1" ytl="2" xbr="11" ybr="22" occluded="0" />
                </track>
                </annotations>""",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    "python3",
                    "tools/import_annotations.py",
                    "--input",
                    str(xml_path),
                    "--output",
                    str(out_path),
                    "--label",
                    "object",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            with out_path.open(newline="", encoding="utf-8") as f:
                imported = list(csv.DictReader(f))
        self.assertEqual(len(imported), 1)
        self.assertEqual(imported[0]["frame_id"], "5")
        self.assertEqual(imported[0]["x2"], "11.00")
        self.assertEqual(imported[0]["occluded"], "0")
        self.assertEqual(imported[0]["bbox_type"], "amodal_full_bin")

    def test_import_coco_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            json_path = tmp_path / "instances_default.json"
            out_path = tmp_path / "out.csv"
            json_path.write_text(
                json.dumps(
                    {
                        "images": [{"id": 10, "file_name": "frame_00042.jpg"}],
                        "categories": [{"id": 1, "name": "object"}],
                        "annotations": [{"image_id": 10, "category_id": 1, "bbox": [1, 2, 10, 20]}],
                    }
                ),
                encoding="utf-8",
            )
            subprocess.run(
                ["python3", "tools/import_annotations.py", "--input", str(json_path), "--output", str(out_path)],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            with out_path.open(newline="", encoding="utf-8") as f:
                imported = list(csv.DictReader(f))
        self.assertEqual(imported[0]["frame_id"], "42")
        self.assertEqual(imported[0]["x2"], "11.00")


class ConfigTests(unittest.TestCase):
    def test_default_config_loads(self) -> None:
        cfg = load_runtime_config()
        self.assertEqual(cfg["detector"]["backend"], "hybrid")
        self.assertTrue(cfg["kalman"]["enabled"])
        self.assertNotIn("scene_control", cfg)

    def test_default_yaml_has_no_scene_control_switch(self) -> None:
        cfg = load_runtime_config("configs/default.yaml")
        self.assertNotIn("scene_control", cfg)

    def test_asset_alignment_diagnostics_warns_on_three_point_fit(self) -> None:
        stops = [
            {
                "name": "A",
                "estimated_filtered_centroid": [0.0, 0.0, 0.325],
                "gt_world_ground": [1.0, 2.0, 0.0],
            },
            {
                "name": "B",
                "estimated_filtered_centroid": [1.0, 0.0, 0.325],
                "gt_world_ground": [3.0, 2.0, 0.0],
            },
            {
                "name": "C",
                "estimated_filtered_centroid": [0.0, 1.0, 0.325],
                "gt_world_ground": [1.0, 5.0, 0.0],
            },
        ]
        report = asset_alignment_diagnostics(stops)
        self.assertTrue(report["enabled"])
        self.assertIn("not a calibration path", report["interpretation"])

    def test_yaml_config_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            cfg_path.write_text(
                """
detector:
  backend: auto
  conf: 0.12
tracker:
  bbox_max_age: 12
""",
                encoding="utf-8",
            )
            cfg = load_runtime_config(str(cfg_path))
        self.assertEqual(cfg["detector"]["backend"], "auto")
        self.assertAlmostEqual(cfg["detector"]["conf"], 0.12)
        self.assertEqual(cfg["detector"]["imgsz"], 640)
        self.assertEqual(cfg["tracker"]["bbox_max_age"], 12)


class ObserverTests(unittest.TestCase):
    def test_motion_state_estimator_labels_stationary_and_moving(self) -> None:
        estimator = MotionStateEstimator(stationary_speed_mps=0.05, moving_speed_mps=0.12)
        first = estimator.update(
            {"status": "detected", "track_state": "CONFIRMED", "world": [0.0, 0.0, 0.325], "conf": 0.9},
            1.0,
        )
        second = estimator.update(
            {"status": "detected", "track_state": "CONFIRMED", "world": [0.01, 0.0, 0.325], "conf": 0.9},
            1.0,
        )
        moving = estimator.update(
            {"status": "detected", "track_state": "CONFIRMED", "world": [1.0, 0.0, 0.325], "conf": 0.9},
            1.0,
        )
        self.assertEqual(first.label, "STATIONARY")
        self.assertEqual(second.label, "STATIONARY")
        self.assertIn(moving.label, {"MOVING", "CHANGING_DIRECTION"})

    def test_motion_state_estimator_preserves_occlusion_label(self) -> None:
        estimator = MotionStateEstimator()
        estimator.update(
            {"status": "detected", "track_state": "CONFIRMED", "world": [0.0, 0.0, 0.325], "conf": 0.9},
            1.0,
        )
        out = estimator.update(
            {"status": "occluded", "track_state": "OCCLUDED", "world": [0.1, 0.0, 0.325], "conf": 0.2},
            1.0,
        )
        self.assertEqual(out.label, "OCCLUDED")


if __name__ == "__main__":
    unittest.main()
