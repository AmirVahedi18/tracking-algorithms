from __future__ import annotations

import time
from typing import Iterator

import cv2
import numpy as np

from . import config
from .detector import Detector
from .monitor import Monitor
from .trackers import build
from .trackers.base import Detection, Track


def _color(track_id: int) -> tuple[int, int, int]:
    np.random.seed(track_id * 3 + 1)
    return tuple(int(x) for x in np.random.randint(64, 256, size=3))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _draw(frame: np.ndarray, tracks: list[Track], labels: dict[int, str]) -> np.ndarray:
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.xyxy]
        c = _color(t.track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        label = labels.get(t.track_id, f"#{t.track_id}")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), c, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


class TrackingPipeline:
    def __init__(self, tracker_key: str, detector: Detector, encoder=None):
        self.tracker_key = tracker_key
        self.detector = detector
        self.encoder = encoder
        self._names: dict[int, str] = {}   # track_id -> class label, sticky across frames

    def run(self, video_path: str, result_path: str | None = None,
            mot_path: str | None = None, selection: dict | None = None) -> Iterator[dict]:
        selection = selection or {}
        mode = selection.get("mode", "all")
        classes = set(selection.get("classes", []))
        seed_boxes = [np.asarray(b, dtype=np.float32) for b in selection.get("seed_boxes", [])]
        if mode == "instance" and not seed_boxes:
            mode = "all"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        tracker = build(self.tracker_key, frame_rate=int(round(fps_in)), encoder=self.encoder)
        monitor = Monitor()

        writer = None
        if result_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(result_path, fourcc, fps_in, (w, h))
        mot_file = open(mot_path, "w") if mot_path else None

        followed: set[int] = set()
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1

                t0 = time.perf_counter()
                dets = self.detector.detect(frame)
                if mode == "class" and classes:
                    dets = [d for d in dets if d.cls in classes]
                if frame_idx == 1 and seed_boxes:
                    dets = self._inject_seeds(dets, seed_boxes)
                t1 = time.perf_counter()
                tracks = tracker.update(dets, frame)
                t2 = time.perf_counter()

                if frame_idx == 1 and mode == "instance":
                    followed = self._match_followed(seed_boxes, tracks)

                self._label(tracks, dets)
                if mode == "instance":
                    tracks = [t for t in tracks if t.track_id in followed]

                monitor.tick((t1 - t0) * 1000, (t2 - t1) * 1000)
                annotated = _draw(frame.copy(), tracks, self._names)

                if writer is not None:
                    writer.write(annotated)
                if mot_file is not None:
                    for t in tracks:
                        x1, y1, x2, y2 = t.xyxy
                        mot_file.write(f"{frame_idx},{t.track_id},{x1:.2f},{y1:.2f},"
                                       f"{x2 - x1:.2f},{y2 - y1:.2f},{t.score:.4f},-1,-1,-1\n")

                yield {
                    "jpeg": self._encode(annotated),
                    "stats": monitor.stats(),
                    "detections": len(dets),
                    "tracks": len(tracks),
                    "frame": frame_idx,
                    "total": total,
                }
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if mot_file is not None:
                mot_file.close()

    def _inject_seeds(self, dets: list[Detection], seeds: list[np.ndarray],
                      iou_thr=0.6, score=0.99) -> list[Detection]:
        for sb in seeds:
            best, bi = 0.0, -1
            for i, d in enumerate(dets):
                iou = _iou(sb, d.xyxy)
                if iou > best:
                    best, bi = iou, i
            if best >= iou_thr:
                dets[bi].score = max(dets[bi].score, score)
            else:
                dets.append(Detection(xyxy=sb.astype(np.float32), score=score, cls=-1))
        return dets

    @staticmethod
    def _match_followed(seeds: list[np.ndarray], tracks: list[Track], iou_thr=0.3) -> set[int]:
        followed = set()
        for sb in seeds:
            best, bid = 0.0, None
            for t in tracks:
                iou = _iou(sb, t.xyxy)
                if iou > best:
                    best, bid = iou, t.track_id
            if bid is not None and best >= iou_thr:
                followed.add(bid)
        return followed

    def _label(self, tracks: list[Track], dets: list[Detection]):
        names = self.detector.names
        for t in tracks:
            best, bcls = 0.3, None
            for d in dets:
                iou = _iou(t.xyxy, d.xyxy)
                if iou > best:
                    best, bcls = iou, d.cls
            if bcls is not None:
                name = "object" if bcls < 0 else names.get(bcls, str(bcls))
                self._names[t.track_id] = f"{name} #{t.track_id}"
            elif t.track_id not in self._names:
                self._names[t.track_id] = f"#{t.track_id}"

    @staticmethod
    def _encode(frame: np.ndarray) -> bytes:
        h, w = frame.shape[:2]
        if w > config.STREAM_MAX_WIDTH:
            scale = config.STREAM_MAX_WIDTH / w
            frame = cv2.resize(frame, (config.STREAM_MAX_WIDTH, int(h * scale)))
        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        return buf.tobytes() if ok else b""
