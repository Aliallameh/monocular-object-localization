# External Tool Assessment

This repo keeps the required runtime small, but external tools can speed up
annotation, validation, and detector upgrades. The rule is: use external systems
for evidence and model development, not as hidden dependencies in the assessment
command.

## Recommended Stack

| Need | Tool | Use In This Repo | Decision |
|---|---|---|---|
| Video bbox annotation | [CVAT](https://github.com/cvat-ai/cvat) | Human GT for `--bbox-gt` | Recommended |
| Dataset conversion | [CVAT COCO export](https://docs.cvat.ai/docs/dataset_management/formats/format-coco/) / Datumaro | Convert CVAT/COCO to CSV with `tools/import_annotations.py` | Recommended |
| Dataset QA/visualization | [FiftyOne CVAT integration](https://docs.voxel51.com/integrations/cvat.html) | Inspect GT/predictions across videos | Useful, optional |
| Lightweight annotation | Label Studio | Alternative if CVAT is too heavy | Optional |
| Detector generalization | YOLO-World / GroundingDINO | Open-vocabulary detector backend | Optional learned path |
| Segmentation-assisted boxes | SAM 2 | Generate masks/boxes for annotation acceleration | Optional, not runtime default |
| Multi-object tracking | [ByteTrack / BoT-SORT via Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/docs/en/modes/track.md) | Useful if multiple similar objects appear | Not default for single target |

## CVAT Evaluation

CVAT is the best fit for our immediate evidence gap: true bbox ground truth.
It is mature, supports video annotation/interpolation, and can export task data
as CVAT XML or COCO JSON. CVAT's own docs describe COCO support for bounding
boxes and track export via custom `track_id` attributes, and the annotation
download workflow supports CVAT-for-video/image plus common dataset formats.
It is too heavy to be a runtime dependency for this assessment, but excellent
as an annotation station.

Recommended workflow:

1. Run the tracker and generate draft boxes.
2. Create review frames:

   ```bash
   .venv/bin/python tools/prepare_bbox_annotations.py \
     --video input.mp4 \
     --tracks results/output.csv \
     --out-dir annotations/bbox_review \
     --samples 60
   ```

3. For a larger annotation job, import the video into CVAT and annotate the
   object track.
4. Export CVAT XML or COCO JSON.
5. Convert it:

   ```bash
   .venv/bin/python tools/import_annotations.py \
     --input annotations/cvat_export.xml \
     --output annotations/bbox_gt.csv
   ```

6. Evaluate:

   ```bash
   bash run.sh --video input.mp4 --calib calib.json --bbox-gt annotations/bbox_gt.csv
   ```

## Why Not Make CVAT A Dependency?

- It is an annotation platform, not a tracker/localizer library.
- It adds Docker/services/storage that are unnecessary for the required command.
- The evaluator should be able to run the repo with Python/OpenCV/NumPy only.
- Keeping CVAT external prevents "it works only inside my annotation stack"
  failure.

## Detector Upgrade Path

For field-like videos, the default classical backend should be treated as a
fallback. Use:

```bash
.venv/bin/python -m pip install -r requirements-learned.txt
bash run.sh --video <video.mp4> --calib calib.json --backend auto --device cpu
```

Then compare:

- detector hit rate,
- tracker output rate,
- occlusion/reacquisition counts,
- true bbox IoU from CVAT annotations,
- p95 latency.

If YOLO-World is not enough, the next candidates are GroundingDINO for
open-vocabulary detection and SAM 2 for segmentation-assisted annotation. Those
are good development/annotation tools, but they need a separate latency and
deployment assessment before becoming runtime dependencies.

## Tracking Libraries

ByteTrack, BoT-SORT, and OC-SORT are strong when there are many objects and
detector outputs are reliable. Ultralytics exposes BoT-SORT and ByteTrack
through tracker YAMLs, with BoT-SORT as the default tracker. This task is
single-target with metric localization, so the custom tracker remains easier to
audit. If future videos contain several similar objects crossing the scene, the
clean integration point is after detection and before `localize_bbox`: external
MOT would output the chosen track bbox, then this repo's geometry and validation
stay unchanged.
