from __future__ import annotations

import numpy as np

from .base import BaseTracker, Detection, Track, register, ns
from .vendor.botsort.bot_sort import BoTSORT


class BoTSortReIDAdapter(BaseTracker):
    display_name = "BoT-SORT-ReID"
    needs_reid = True

    def __init__(self, track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.6,
                 track_buffer=30, match_thresh=0.8, proximity_thresh=0.5,
                 appearance_thresh=0.25, cmc_method="sparseOptFlow", mot20=False,
                 frame_rate=30, encoder=None, **_):
        args = ns(track_high_thresh=track_high_thresh, track_low_thresh=track_low_thresh,
                  new_track_thresh=new_track_thresh, track_buffer=track_buffer,
                  match_thresh=match_thresh, proximity_thresh=proximity_thresh,
                  appearance_thresh=appearance_thresh, with_reid=encoder is not None,
                  cmc_method=cmc_method, name="botsort", ablation=False, mot20=mot20,
                  encoder=encoder)
        self.tracker = BoTSORT(args, frame_rate=frame_rate)

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        dets = self._dets_to_array(detections)
        online = self.tracker.update(dets, frame)
        out = []
        for t in online:
            if not t.is_activated:
                continue
            x, y, bw, bh = t.tlwh
            out.append(Track(track_id=t.track_id,
                             xyxy=np.array([x, y, x + bw, y + bh]),
                             score=float(t.score)))
        return out


register("botsort", "BoT-SORT-ReID (2022)", BoTSortReIDAdapter, needs_reid=True)
