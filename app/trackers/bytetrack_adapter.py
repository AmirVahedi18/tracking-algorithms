from __future__ import annotations

import numpy as np

from .base import BaseTracker, Detection, Track, register, ns
from .vendor.bytetrack.byte_tracker import BYTETracker


class ByteTrackAdapter(BaseTracker):
    display_name = "ByteTrack"

    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8,
                 mot20=False, frame_rate=30, **_):
        args = ns(track_thresh=track_thresh, track_buffer=track_buffer,
                  match_thresh=match_thresh, mot20=mot20)
        self.tracker = BYTETracker(args, frame_rate=frame_rate)

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        dets = self._dets_to_array(detections)
        h, w = frame.shape[:2]
        online = self.tracker.update(dets, (h, w), (h, w))
        out = []
        for t in online:
            x, y, bw, bh = t.tlwh
            out.append(Track(track_id=t.track_id,
                             xyxy=np.array([x, y, x + bw, y + bh]),
                             score=float(t.score)))
        return out


register("bytetrack", "ByteTrack (2022)", ByteTrackAdapter)
