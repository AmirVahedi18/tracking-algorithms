from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable

import numpy as np


@dataclass
class Detection:
    xyxy: np.ndarray          # [x1, y1, x2, y2]
    score: float
    cls: int = 0


@dataclass
class Track:
    track_id: int
    xyxy: np.ndarray          # [x1, y1, x2, y2]
    score: float = 0.0
    cls: int = 0


class BaseTracker:
    """Common interface every vendored algorithm is adapted to."""

    #: shown in the UI
    display_name: str = "base"
    #: whether this tracker consumes appearance embeddings
    needs_reid: bool = False

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        raise NotImplementedError

    @staticmethod
    def _dets_to_array(detections: list[Detection]) -> np.ndarray:
        if not detections:
            return np.empty((0, 5), dtype=np.float32)
        rows = [np.concatenate([d.xyxy, [d.score]]) for d in detections]
        return np.asarray(rows, dtype=np.float32)


_REGISTRY: dict[str, dict] = {}


def register(key: str, name: str, factory: Callable[..., BaseTracker], **meta):
    _REGISTRY[key] = {"name": name, "factory": factory, **meta}


def available() -> list[dict]:
    return [{"key": k, "name": v["name"], "needs_reid": v.get("needs_reid", False)}
            for k, v in _REGISTRY.items()]


def build(key: str, **kwargs) -> BaseTracker:
    if key not in _REGISTRY:
        raise KeyError(f"unknown tracker '{key}', have {list(_REGISTRY)}")
    return _REGISTRY[key]["factory"](**kwargs)


def ns(**kwargs) -> SimpleNamespace:
    """Reference trackers expect an argparse-like namespace."""
    return SimpleNamespace(**kwargs)
