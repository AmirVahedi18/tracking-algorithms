from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from . import config
from .trackers.base import Detection


class Detector:
    def __init__(self, model_path: str | None = None, device: str | None = None):
        self.device = device or config.resolve_device()
        path = model_path or config.YOLO_MODEL
        if not (config.WEIGHTS_DIR / path).exists() and "/" not in path:
            # Ultralytics auto-downloads known model names into cwd; keep them in weights/.
            self.model = YOLO(str(config.WEIGHTS_DIR / path))
        else:
            self.model = YOLO(path)
        self.model.to(self.device)
        self.names = self.model.names
        self.half = config.YOLO_HALF in ("1", "true", "True") and self.device == "cuda"

    def detect(self, frame: np.ndarray, conf: float | None = None) -> list[Detection]:
        kw = {"half": True} if self.half else {}
        res = self.model.predict(frame, imgsz=config.YOLO_IMGSZ,
                                 conf=conf if conf is not None else config.YOLO_CONF,
                                 device=self.device, verbose=False, **kw)[0]
        dets: list[Detection] = []
        if res.boxes is None:
            return dets
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        for box, s, c in zip(xyxy, conf, cls):
            if config.KEEP_CLASSES and c not in config.KEEP_CLASSES:
                continue
            dets.append(Detection(xyxy=box.astype(np.float32), score=float(s), cls=int(c)))
        return dets
