from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORAGE = ROOT / "storage"
UPLOAD_DIR = STORAGE / "uploads"
RESULT_DIR = STORAGE / "results"
WEIGHTS_DIR = ROOT / "weights"

for _d in (UPLOAD_DIR, RESULT_DIR, WEIGHTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Detector — YOLOv8l trained on Open Images V7 (601 classes)
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8l-oiv7.pt")
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.1"))     # low: trackers do their own gating
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
# FP32 by default: GTX 16-series (TU116/117) have crippled FP16 throughput (~4x slower).
# Set YOLO_HALF=1 on RTX/tensor-core GPUs to speed up and save VRAM.
YOLO_HALF = os.getenv("YOLO_HALF", "0")
FIRSTFRAME_CONF = float(os.getenv("FIRSTFRAME_CONF", "0.25"))  # cleaner boxes for selection UI
# COCO class ids to keep. Empty set -> keep all classes.
KEEP_CLASSES: set[int] = set()

# ReID (BoT-SORT-ReID appearance model)
REID_MODEL = os.getenv("REID_MODEL", "osnet_x0_25_msmt17.pt")
# deep-person-reid model zoo: OSNet x0.25 trained on MSMT17 (softmax)
REID_GDRIVE_ID = os.getenv("REID_GDRIVE_ID", "1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF")

DEVICE = os.getenv("DEVICE", "auto")   # auto | cuda | cpu

# Streaming
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))
STREAM_MAX_WIDTH = int(os.getenv("STREAM_MAX_WIDTH", "960"))


def resolve_device() -> str:
    if DEVICE != "auto":
        return DEVICE
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
