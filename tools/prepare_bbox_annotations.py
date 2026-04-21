"""Prepare a human bbox annotation packet from a tracker CSV and video.

This script does not create ground truth. It extracts review frames, overlays
the current predicted box as a draft, and writes both a CSV template and a small
offline HTML annotator. A reviewer can correct the boxes and then pass the CSV
back into:

    bash run.sh --video input.mp4 --calib calib.json --bbox-gt annotations/bbox_gt.csv
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare bbox GT annotation review assets")
    parser.add_argument("--video", required=True, help="Video to sample")
    parser.add_argument("--tracks", default="results/output.csv", help="Tracker output CSV with bbox columns")
    parser.add_argument("--out-dir", default="annotations/bbox_review", help="Output annotation packet directory")
    parser.add_argument("--samples", type=int, default=60, help="Number of frames to sample")
    parser.add_argument(
        "--include-frames",
        default="50,200,500",
        help="Comma-separated must-include frame IDs such as reviewed contact/occlusion frames",
    )
    parser.add_argument(
        "--include-ranges",
        default="",
        help="Comma-separated inclusive frame ranges, e.g. 420:470,610:650, for real occlusion review",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    image_dir = out_dir / "frames"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = read_tracks(args.tracks)
    if not rows:
        raise RuntimeError(f"No tracked bbox rows found in {args.tracks}")
    sample_ids = choose_samples(
        rows,
        args.samples,
        parse_frame_list(args.include_frames),
        parse_frame_ranges(args.include_ranges),
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    annotation_rows: List[Dict[str, object]] = []
    for frame_id in sample_ids:
        row = rows[frame_id]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        x1, y1, x2, y2 = [float(row[k]) for k in ("x1", "y1", "x2", "y2")]
        clean_name = f"frame_{frame_id:05d}.jpg"
        overlay_name = f"frame_{frame_id:05d}_draft.jpg"
        cv2.imwrite(str(image_dir / clean_name), frame)
        overlay = draw_box(frame.copy(), (x1, y1, x2, y2), row)
        cv2.imwrite(str(image_dir / overlay_name), overlay)
        annotation_rows.append(
            {
                "frame_id": frame_id,
                "image": f"frames/{clean_name}",
                "draft_image": f"frames/{overlay_name}",
                "x1": f"{x1:.2f}",
                "y1": f"{y1:.2f}",
                "x2": f"{x2:.2f}",
                "y2": f"{y2:.2f}",
                "review_status": "draft",
                "occluded": "",
                "visibility": "",
                "bbox_type": "amodal_full_bin",
                "notes": "",
            }
        )
    cap.release()

    csv_path = out_dir / "bbox_gt_template.csv"
    write_template(csv_path, annotation_rows)
    write_html(out_dir / "bbox_annotator.html", annotation_rows)
    write_manifest(out_dir / "manifest.json", args, annotation_rows)
    print(f"[annotate] wrote {csv_path}", flush=True)
    print(f"[annotate] wrote {out_dir / 'bbox_annotator.html'}", flush=True)
    print(f"[annotate] extracted {len(annotation_rows)} frames into {image_dir}", flush=True)


def read_tracks(path: str) -> Dict[int, Dict[str, str]]:
    rows: Dict[int, Dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"frame_id", "x1", "y1", "x2", "y2"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for row in reader:
            try:
                frame_id = int(row["frame_id"])
                vals = [float(row[k]) for k in ("x1", "y1", "x2", "y2")]
            except Exception:
                continue
            if vals[2] > vals[0] and vals[3] > vals[1]:
                rows[frame_id] = row
    return rows


def choose_samples(
    rows: Dict[int, Dict[str, str]],
    samples: int,
    include_frames: List[int],
    include_ranges: List[tuple[int, int]],
) -> List[int]:
    frame_ids = np.asarray(sorted(rows), dtype=int)
    if len(frame_ids) == 0:
        return []
    uniform = np.linspace(0, len(frame_ids) - 1, num=min(samples, len(frame_ids)), dtype=int)
    selected = {int(frame_ids[i]) for i in uniform}

    # Prioritize hard frames: occlusions, reacquisitions, low confidence, and large
    # world-gate innovations if those columns exist.
    hard: List[tuple[float, int]] = []
    for frame_id, row in rows.items():
        score = 0.0
        state = str(row.get("track_state", "")).upper()
        if state in {"OCCLUDED", "REACQUIRED", "SEARCHING"}:
            score += 3.0
        try:
            score += max(0.0, 0.8 - float(row.get("conf", "0"))) * 2.0
        except Exception:
            pass
        try:
            gate = float(row.get("gate_d2", "nan"))
            if np.isfinite(gate):
                score += min(3.0, gate / 25.0)
        except Exception:
            pass
        if score > 0:
            hard.append((score, frame_id))
    hard.sort(reverse=True)
    selected.update(frame_id for _, frame_id in hard[: max(10, samples // 3)])
    selected.update(frame_id for frame_id in include_frames if frame_id in rows)
    for start, end in include_ranges:
        selected.update(frame_id for frame_id in range(start, end + 1) if frame_id in rows)
    return sorted(selected)


def parse_frame_list(value: str) -> List[int]:
    out: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def parse_frame_ranges(value: str) -> List[tuple[int, int]]:
    out: List[tuple[int, int]] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            frame = int(part)
            out.append((frame, frame))
            continue
        a, b = part.split(":", 1)
        start, end = int(a), int(b)
        if end < start:
            start, end = end, start
        out.append((start, end))
    return out


def draw_box(frame, bbox, row):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 255), 3)
    label = f"draft f={row.get('frame_id')} conf={row.get('conf', '')} state={row.get('track_state', row.get('status', ''))}"
    cv2.rectangle(frame, (x1, max(0, y1 - 34)), (min(frame.shape[1] - 1, x1 + 620), y1), (0, 0, 0), -1)
    cv2.putText(frame, label, (x1 + 8, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    return frame


def write_template(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_id",
                "image",
                "draft_image",
                "x1",
                "y1",
                "x2",
                "y2",
                "review_status",
                "occluded",
                "visibility",
                "bbox_type",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path: Path, args: argparse.Namespace, rows: List[Dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "video": args.video,
                "tracks": args.tracks,
                "annotation_csv": "bbox_gt_template.csv",
                "html_tool": "bbox_annotator.html",
                "frames": len(rows),
                "instructions": (
                "Correct x1,y1,x2,y2 around the amodal/full bin extent when inferable. "
                "For real occlusion evidence, set occluded=1 and visibility to an estimated fraction. "
                "Set review_status=ok for usable GT, skip for frames that are too ambiguous."
            ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_html(path: Path, rows: List[Dict[str, object]]) -> None:
    payload = json.dumps(rows)
    rows_html = "\n".join(
        f"""
        <section class="card" data-frame="{row['frame_id']}">
          <div class="card-title">
            <h2>Frame {row['frame_id']}</h2>
            <span class="pill">draft box is only a starting point</span>
          </div>
          <div class="images">
            <figure>
              <figcaption>Draw / adjust GT box here</figcaption>
              <div class="canvas-wrap">
                <img src="{html.escape(str(row['image']))}" alt="clean frame {row['frame_id']}">
                <canvas data-frame="{row['frame_id']}"></canvas>
              </div>
            </figure>
            <figure>
              <figcaption>Draft tracker box</figcaption>
              <img src="{html.escape(str(row['draft_image']))}" alt="draft frame {row['frame_id']}">
            </figure>
          </div>
          <div class="grid">
            <label>x1 <input value="{row['x1']}" data-field="x1" readonly></label>
            <label>y1 <input value="{row['y1']}" data-field="y1" readonly></label>
            <label>x2 <input value="{row['x2']}" data-field="x2" readonly></label>
            <label>y2 <input value="{row['y2']}" data-field="y2" readonly></label>
            <label>status <select data-field="review_status"><option>ok</option><option selected>draft</option><option>skip</option></select></label>
            <label>occluded <select data-field="occluded"><option value=""></option><option value="0">0</option><option value="1">1</option></select></label>
            <label>visibility <input value="" data-field="visibility" placeholder="0.0-1.0"></label>
            <label>bbox type <select data-field="bbox_type"><option selected>amodal_full_bin</option><option>visible_extent</option></select></label>
            <label>notes <input value="" data-field="notes"></label>
          </div>
          <div class="actions">
            <button type="button" onclick="markOk(this)">OK</button>
            <button type="button" onclick="markOccluded(this)">Occluded OK</button>
            <button type="button" onclick="markSkip(this)">Skip</button>
            <button type="button" onclick="resetDraft(this)">Reset Draft</button>
          </div>
        </section>
        """
        for row in rows
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>BBox Annotation Review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; background: #f5f6f7; color: #111; }}
    header {{ position: sticky; top: 0; z-index: 3; background: #fff; border-bottom: 1px solid #cfd5dc; padding: 14px 22px; }}
    main {{ margin: 22px; }}
    button {{ border-radius: 6px; border: 1px solid #333; padding: 9px 14px; background: #111; color: white; cursor: pointer; }}
    .card {{ background: white; border: 1px solid #d0d4d8; border-radius: 8px; padding: 14px; margin: 16px 0; }}
    .card-title {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
    .pill {{ border: 1px solid #a9b2bc; border-radius: 999px; padding: 4px 8px; color: #303942; font-size: 12px; }}
    .images {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    figure {{ margin: 0; }}
    figcaption {{ font-size: 13px; font-weight: 700; margin: 0 0 6px; color: #29313a; }}
    img {{ width: 100%; height: auto; display: block; border: 1px solid #999; background: #ddd; }}
    .canvas-wrap {{ position: relative; }}
    .canvas-wrap img {{ position: relative; z-index: 1; }}
    .canvas-wrap canvas {{ position: absolute; inset: 0; z-index: 2; width: 100%; height: 100%; cursor: crosshair; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 12px; }}
    input, select {{ width: 100%; box-sizing: border-box; padding: 6px; }}
    input[readonly] {{ background: #eef1f4; color: #222; }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
    .warn {{ color: #8a3000; font-weight: 700; }}
    @media (max-width: 900px) {{ .images {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>BBox Annotation Review</h1>
    <p>Drag on the clean frame to draw the GT box. Mark usable rows as <strong>ok</strong>, mark real occlusion rows with <strong>occluded=1</strong>, then download CSV and run with <code>--bbox-gt</code>. For continuity evidence, use amodal/full-bin boxes when the bin extent is inferable; skip fully ambiguous frames.</p>
    <p class="warn">Draft rows are ignored by evaluation. Nothing counts until you set review_status=ok.</p>
    <button onclick="downloadCsv()">Download corrected CSV</button>
  </header>
  <main>
    {rows_html}
  </main>
  <script>
    const initialRows = {base64.b64encode(payload.encode()).decode()};
    function b64decode(s) {{ return decodeURIComponent(Array.prototype.map.call(atob(s), c => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2)).join('')); }}
    const rowByFrame = new Map(initialRows.map(r => [String(r.frame_id), r]));
    function fields(card) {{
      const out = {{}};
      card.querySelectorAll('[data-field]').forEach(el => out[el.dataset.field] = el);
      return out;
    }}
    function setBox(card, box) {{
      const f = fields(card);
      f.x1.value = box.x1.toFixed(2);
      f.y1.value = box.y1.toFixed(2);
      f.x2.value = box.x2.toFixed(2);
      f.y2.value = box.y2.toFixed(2);
      drawCanvas(card);
    }}
    function getBox(card) {{
      const f = fields(card);
      return {{x1:+f.x1.value, y1:+f.y1.value, x2:+f.x2.value, y2:+f.y2.value}};
    }}
    function drawCanvas(card) {{
      const img = card.querySelector('.canvas-wrap img');
      const canvas = card.querySelector('canvas');
      const ctx = canvas.getContext('2d');
      const w = img.naturalWidth || 1366;
      const h = img.naturalHeight || 768;
      canvas.width = w;
      canvas.height = h;
      ctx.clearRect(0, 0, w, h);
      const b = getBox(card);
      ctx.lineWidth = Math.max(3, w / 450);
      ctx.strokeStyle = '#ff00cc';
      ctx.fillStyle = 'rgba(255,0,204,0.10)';
      ctx.fillRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
      ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    }}
    function pointerToImage(e, canvas) {{
      const r = canvas.getBoundingClientRect();
      return {{
        x: (e.clientX - r.left) * canvas.width / r.width,
        y: (e.clientY - r.top) * canvas.height / r.height,
      }};
    }}
    document.querySelectorAll('.card').forEach(card => {{
      const canvas = card.querySelector('canvas');
      const img = card.querySelector('.canvas-wrap img');
      let start = null;
      img.addEventListener('load', () => drawCanvas(card));
      canvas.addEventListener('pointerdown', e => {{
        start = pointerToImage(e, canvas);
        canvas.setPointerCapture(e.pointerId);
      }});
      canvas.addEventListener('pointermove', e => {{
        if (!start) return;
        const p = pointerToImage(e, canvas);
        setBox(card, {{
          x1: Math.min(start.x, p.x),
          y1: Math.min(start.y, p.y),
          x2: Math.max(start.x, p.x),
          y2: Math.max(start.y, p.y),
        }});
      }});
      canvas.addEventListener('pointerup', e => {{
        if (!start) return;
        const f = fields(card);
        if (f.review_status.value === 'draft') f.review_status.value = 'ok';
        start = null;
        drawCanvas(card);
      }});
      drawCanvas(card);
    }});
    function cardFromButton(btn) {{ return btn.closest('.card'); }}
    function markOk(btn) {{
      const f = fields(cardFromButton(btn));
      f.review_status.value = 'ok';
      if (!f.occluded.value) f.occluded.value = '0';
    }}
    function markOccluded(btn) {{
      const f = fields(cardFromButton(btn));
      f.review_status.value = 'ok';
      f.occluded.value = '1';
      if (!f.visibility.value) f.visibility.value = '0.7';
    }}
    function markSkip(btn) {{
      fields(cardFromButton(btn)).review_status.value = 'skip';
    }}
    function resetDraft(btn) {{
      const card = cardFromButton(btn);
      const r = rowByFrame.get(card.dataset.frame);
      setBox(card, {{x1:+r.x1, y1:+r.y1, x2:+r.x2, y2:+r.y2}});
      fields(card).review_status.value = 'draft';
    }}
    function collectRows() {{
      return Array.from(document.querySelectorAll('.card')).map(card => {{
        const row = {{frame_id: card.dataset.frame, image: `frames/frame_${{String(card.dataset.frame).padStart(5, '0')}}.jpg`, draft_image: `frames/frame_${{String(card.dataset.frame).padStart(5, '0')}}_draft.jpg`}};
        card.querySelectorAll('[data-field]').forEach(el => row[el.dataset.field] = el.value);
        return row;
      }});
    }}
    function downloadCsv() {{
      const cols = ['frame_id','image','draft_image','x1','y1','x2','y2','review_status','occluded','visibility','bbox_type','notes'];
      const lines = [cols.join(',')];
      for (const row of collectRows()) {{
        lines.push(cols.map(c => `"${{String(row[c] ?? '').replaceAll('"', '""')}}"`).join(','));
      }}
      const blob = new Blob([lines.join('\\n') + '\\n'], {{type: 'text/csv'}});
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'bbox_gt_corrected.csv';
      a.click();
    }}
  </script>
</body>
</html>
"""
    # Keep the base64 payload parseable for offline debugging, even though the
    # current UI reads directly from the DOM.
    html_text = html_text.replace(f"const initialRows = {base64.b64encode(payload.encode()).decode()};", f"const initialRows = JSON.parse(b64decode('{base64.b64encode(payload.encode()).decode()}'));")
    path.write_text(html_text, encoding="utf-8")


if __name__ == "__main__":
    main()
