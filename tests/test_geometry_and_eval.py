from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
import subprocess

import numpy as np

from eval_utils import evaluate_bbox_annotations
from localizer import BIN_HEIGHT_M, CameraGeometry, build_camera_to_world
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


class EvaluationTests(unittest.TestCase):
    def test_bbox_iou(self) -> None:
        self.assertAlmostEqual(bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)

    def test_bbox_annotation_evaluator_csv(self) -> None:
        rows = [
            {"frame_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0]},
            {"frame_id": 2, "bbox": [10.0, 10.0, 20.0, 20.0]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            gt_path = Path(tmp) / "gt.csv"
            with gt_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["frame_id", "x1", "y1", "x2", "y2", "review_status"])
                writer.writerow([1, 0, 0, 10, 10, "ok"])
                writer.writerow([2, 12, 12, 22, 22, "ok"])
            report = evaluate_bbox_annotations(gt_path, rows)
        self.assertTrue(report["enabled"])
        self.assertEqual(report["gt_frames"], 2)
        self.assertEqual(report["matched_frames"], 2)
        self.assertGreater(report["mean_iou"], 0.5)

    def test_bbox_annotation_evaluator_json(self) -> None:
        rows = [{"frame_id": 7, "bbox": [1.0, 2.0, 11.0, 12.0]}]
        with tempfile.TemporaryDirectory() as tmp:
            gt_path = Path(tmp) / "gt.json"
            gt_path.write_text(json.dumps({"frames": [{"frame_id": 7, "x1": 1, "y1": 2, "x2": 11, "y2": 12}]}))
            report = evaluate_bbox_annotations(gt_path, rows)
        self.assertTrue(report["passes_hidden_contract_proxy"])

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


if __name__ == "__main__":
    unittest.main()
