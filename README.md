# Multi-Object Tracking Studio

A web demo comparing four modern tracking-by-detection algorithms on your own videos,
with a live annotated stream, FPS, and resource metrics.

| Algorithm | Year | Notes |
|-----------|------|-------|
| **ByteTrack** | 2022 | Low-confidence association |
| **BoT-SORT-ReID** | 2022 | Motion + camera compensation + appearance ReID |
| **OC-SORT** | 2023 | Observation-centric re-update |
| **Hybrid-SORT** | AAAI 2024 | Adds confidence/height cues (TCM) |

A shared **Ultralytics YOLOv8l-OIV7** detector (**601 Open Images V7 classes**) feeds every
tracker. Each algorithm is the **official reference implementation**, vendored under
`app/trackers/vendor/` and adapted to one common
`BaseTracker.update(detections, frame) -> tracks` interface (`app/trackers/*_adapter.py`).

## Object selection

After upload, the first frame is detected and shown on an interactive canvas
(`POST /api/first-frame`). You can:

- **Click** detection boxes to multi-select specific objects.
- **Drag** on empty space to draw a custom box around anything (even objects YOLO
  doesn't know). Drawn boxes are injected as high-confidence detections to *seed* a
  track in the chosen algorithm — the track then persists only while the detector keeps
  re-finding that object (an unknown object coasts on Kalman prediction, then drops).
- Toggle **scope** per run:
  - **Selected objects** (instance mode): only the picked/drawn instances are followed
    (matched to frame-1 tracks by IoU); objects entering later are ignored.
  - **Whole classes** (class mode): every object of the selected objects' classes is
    tracked, whenever it appears.
- Select nothing to track **everything**.

Picked detections + drawn boxes can be mixed in a single run. Boxes are labelled with
their class (`car #3`) via a per-frame IoU match back to the detections.

## Architecture

```
app/
  main.py       FastAPI: upload, first-frame detect, tracker list, WebSocket stream, download
  detector.py   YOLO wrapper -> Detection[]
  reid.py       OSNet appearance embedder (BoT-SORT-ReID)
  pipeline.py   detect -> track -> annotate loop; streams frames + saves mp4 / MOT txt
  monitor.py    rolling FPS (detect/track split) + CPU/RAM/GPU sampling
  trackers/
    base.py     interface, Detection/Track types, registry
    *_adapter.py
    vendor/     official tracker source (lightly patched, see below)
  static/       vanilla HTML/CSS/JS dashboard
```

## Setup & run

```bash
./setup.sh        # creates .venv and installs requirements
./run.sh          # http://localhost:8000
```

YOLO and ReID weights download automatically on first use into `weights/`.
Configuration is via env vars (see `app/config.py`): `YOLO_MODEL`, `DEVICE`,
`STREAM_MAX_WIDTH`, `JPEG_QUALITY`, etc.

**Detector notes.** On the GTX 1650, YOLOv8l-OIV7 runs ~16 detect-FPS at FP32 / 640px
(≈14 FPS end-to-end for ByteTrack/OC-SORT/Hybrid-SORT, ≈9 FPS for BoT-SORT-ReID). FP32 is
the default because **GTX 16-series (TU116/117) GPUs have crippled FP16 throughput (~4×
slower)** — set `YOLO_HALF=1` only on RTX/tensor-core cards. For more speed, drop to
`YOLO_MODEL=yolov8m-oiv7.pt` (still 601 classes, ~2× faster) or lower `YOLO_IMGSZ` (e.g. 512). Detection is restricted to the `person`
class by default (`KEEP_CLASSES` in `config.py`).

Results (annotated `.mp4` + MOT-format `.txt`) are written to `storage/results/`.

## Deviations from the papers (for feasibility on Python 3.13 / a 4 GB GPU)

These keep the **algorithm logic untouched** while making the reference code installable:

- `cython_bbox` (no Python 3.13 wheels) → a pure-NumPy `bbox_ious` (`vendor/_iou.py`).
- Unmaintained `lap` → `lapx` (drop-in, same `import lap`).
- Removed NumPy aliases (`np.float`) → builtin `float`.
- **BoT-SORT-ReID** uses a lightweight **OSNet** embedder instead of the paper's heavier
  FastReID stack, so it runs on a 4 GB card. The encoder is injected into the reference
  `BoTSORT` via its existing `self.encoder.inference(image, dets)` hook.
