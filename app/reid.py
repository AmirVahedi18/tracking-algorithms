from __future__ import annotations

import cv2
import gdown
import numpy as np
import torch

from . import config
from .reid_models.osnet import osnet_x0_25


def _ensure_weights() -> str:
    path = config.WEIGHTS_DIR / config.REID_MODEL
    if not path.exists():
        print(f"[reid] downloading {config.REID_MODEL} ...")
        gdown.download(id=config.REID_GDRIVE_ID, output=str(path), quiet=False)
    return str(path)


class ReIDEncoder:
    """Appearance embedder exposing the interface BoT-SORT expects:
    `inference(image_bgr, dets_xyxy) -> np.ndarray[N, D]` (L2-normalized)."""

    def __init__(self, device: str | None = None, input_size=(256, 128)):
        self.device = device or config.resolve_device()
        self.input_size = input_size  # (h, w)
        self.model = osnet_x0_25(num_classes=1000, pretrained=False)
        state = torch.load(_ensure_weights(), map_location="cpu", weights_only=False)
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        clean = {k.replace("module.", ""): v for k, v in state.items()
                 if not k.replace("module.", "").startswith("classifier.")}
        self.model.load_state_dict(clean, strict=False)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def inference(self, image: np.ndarray, dets: np.ndarray) -> np.ndarray:
        if dets is None or len(dets) == 0:
            return np.empty((0, 512), dtype=np.float32)
        h, w = image.shape[:2]
        crops = []
        for x1, y1, x2, y2 in np.asarray(dets)[:, :4]:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 <= x1 or y2 <= y1:
                crops.append(np.zeros((*self.input_size, 3), np.uint8))
                continue
            crop = cv2.resize(image[y1:y2, x1:x2], (self.input_size[1], self.input_size[0]))
            crops.append(crop)
        batch = np.stack(crops).astype(np.float32)[..., ::-1] / 255.0  # BGR->RGB
        batch = (batch - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2))).float()
        feats = self.model(tensor.to(self.device))
        feats = torch.nn.functional.normalize(feats, dim=1)
        return feats.cpu().numpy().astype(np.float32)
