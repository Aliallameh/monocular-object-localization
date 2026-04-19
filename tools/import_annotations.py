"""Convert external annotation exports into this repo's bbox-GT CSV.

Supported inputs:

- CVAT XML for video/interpolation tasks
- CVAT XML for image tasks
- COCO JSON detection export
- ZIP files containing one of the above

Output schema is compatible with:

    bash run.sh --video input.mp4 --calib calib.json --bbox-gt annotations/bbox_gt.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from xml.etree import ElementTree as ET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import CVAT/COCO annotations into bbox GT CSV")
    parser.add_argument("--input", required=True, help="CVAT XML, COCO JSON, or ZIP export")
    parser.add_argument("--output", default="annotations/bbox_gt.csv", help="Output bbox GT CSV")
    parser.add_argument("--label", default="", help="Optional label/category filter")
    parser.add_argument("--track-id", type=int, default=None, help="Optional CVAT track id filter")
    parser.add_argument(
        "--duplicate-policy",
        choices=["largest", "first", "fail"],
        default="largest",
        help="How to collapse multiple boxes on the same frame",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    resolved = resolve_input(input_path)
    label_filter = args.label.strip() or None

    if resolved.suffix.lower() == ".xml":
        rows = read_cvat_xml(resolved, label_filter=label_filter, track_id=args.track_id)
    elif resolved.suffix.lower() == ".json":
        rows = read_coco_json(resolved, label_filter=label_filter)
    else:
        raise ValueError(f"Unsupported annotation file: {resolved}")

    collapsed, duplicate_count = collapse_duplicates(rows, args.duplicate_policy)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_path, collapsed)
    report_path = out_path.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(
            {
                "input": str(input_path),
                "resolved_file": str(resolved),
                "output": str(out_path),
                "label_filter": label_filter,
                "track_id_filter": args.track_id,
                "raw_boxes": len(rows),
                "output_frames": len(collapsed),
                "duplicate_frames": duplicate_count,
                "duplicate_policy": args.duplicate_policy,
                "labels": dict(Counter(str(row.get("label", "")) for row in rows)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[import] read {len(rows)} boxes from {resolved}", flush=True)
    print(f"[import] wrote {len(collapsed)} frame boxes to {out_path}", flush=True)
    print(f"[import] wrote report {report_path}", flush=True)


def resolve_input(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".zip":
        return path
    temp_dir = Path(tempfile.mkdtemp(prefix="annotation_export_"))
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(temp_dir)
    candidates = sorted(temp_dir.rglob("*.xml")) + sorted(temp_dir.rglob("*.json"))
    if not candidates:
        raise ValueError(f"No XML/JSON annotation file found in {path}")
    preferred = [
        p
        for p in candidates
        if p.name in {"annotations.xml", "instances_default.json", "instances_train.json", "instances_val.json"}
    ]
    return preferred[0] if preferred else candidates[0]


def read_cvat_xml(path: Path, label_filter: Optional[str], track_id: Optional[int]) -> List[Dict[str, object]]:
    root = ET.parse(path).getroot()
    rows: List[Dict[str, object]] = []

    # Video/interpolation export: <track id="..." label="..."><box frame="...">
    for track in root.findall("track"):
        tid = int(track.attrib.get("id", "-1"))
        label = track.attrib.get("label", "")
        if track_id is not None and tid != track_id:
            continue
        if label_filter is not None and label != label_filter:
            continue
        for box in track.findall("box"):
            if box.attrib.get("outside", "0") == "1":
                continue
            rows.append(
                {
                    "frame_id": int(box.attrib["frame"]),
                    "x1": float(box.attrib["xtl"]),
                    "y1": float(box.attrib["ytl"]),
                    "x2": float(box.attrib["xbr"]),
                    "y2": float(box.attrib["ybr"]),
                    "label": label,
                    "source": "cvat_track",
                    "track_id": tid,
                    "occluded": int(box.attrib.get("occluded", "0")),
                }
            )

    # Image/annotation export: <image id="..." name="..."><box ...>
    for image in root.findall("image"):
        frame_id = frame_id_from_image(image.attrib.get("name", ""), fallback=int(image.attrib.get("id", "0")))
        for box in image.findall("box"):
            label = box.attrib.get("label", "")
            if label_filter is not None and label != label_filter:
                continue
            rows.append(
                {
                    "frame_id": frame_id,
                    "x1": float(box.attrib["xtl"]),
                    "y1": float(box.attrib["ytl"]),
                    "x2": float(box.attrib["xbr"]),
                    "y2": float(box.attrib["ybr"]),
                    "label": label,
                    "source": "cvat_image",
                    "track_id": "",
                    "occluded": int(box.attrib.get("occluded", "0")),
                }
            )
    return rows


def read_coco_json(path: Path, label_filter: Optional[str]) -> List[Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    categories = {int(cat["id"]): str(cat.get("name", "")) for cat in data.get("categories", [])}
    images = {
        int(img["id"]): frame_id_from_image(str(img.get("file_name", "")), fallback=int(img.get("id", 0)))
        for img in data.get("images", [])
    }
    rows: List[Dict[str, object]] = []
    for ann in data.get("annotations", []):
        cat_id = int(ann.get("category_id", -1))
        label = categories.get(cat_id, str(cat_id))
        if label_filter is not None and label != label_filter:
            continue
        x, y, w, h = [float(v) for v in ann["bbox"]]
        rows.append(
            {
                "frame_id": images.get(int(ann["image_id"]), int(ann["image_id"])),
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
                "label": label,
                "source": "coco",
                "track_id": "",
                "occluded": int(ann.get("iscrowd", 0)),
            }
        )
    return rows


def frame_id_from_image(name: str, fallback: int) -> int:
    stem = Path(name).stem
    numbers = re.findall(r"\d+", stem)
    if numbers:
        return int(numbers[-1])
    return int(fallback)


def collapse_duplicates(rows: List[Dict[str, object]], policy: str) -> tuple[List[Dict[str, object]], int]:
    grouped: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["frame_id"])].append(row)
    out: List[Dict[str, object]] = []
    duplicate_count = 0
    for frame_id in sorted(grouped):
        group = grouped[frame_id]
        if len(group) > 1:
            duplicate_count += 1
        if len(group) > 1 and policy == "fail":
            raise ValueError(f"Multiple boxes on frame {frame_id}; use --label/--track-id or another duplicate policy")
        if policy == "first":
            chosen = group[0]
        else:
            chosen = max(group, key=area)
        out.append(chosen)
    return out, duplicate_count


def area(row: Dict[str, object]) -> float:
    return max(0.0, float(row["x2"]) - float(row["x1"])) * max(0.0, float(row["y2"]) - float(row["y1"]))


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "review_status",
                "label",
                "source",
                "track_id",
                "occluded",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "frame_id": int(row["frame_id"]),
                    "x1": f"{float(row['x1']):.2f}",
                    "y1": f"{float(row['y1']):.2f}",
                    "x2": f"{float(row['x2']):.2f}",
                    "y2": f"{float(row['y2']):.2f}",
                    "review_status": "ok",
                    "label": row.get("label", ""),
                    "source": row.get("source", ""),
                    "track_id": row.get("track_id", ""),
                    "occluded": row.get("occluded", 0),
                }
            )


if __name__ == "__main__":
    main()
