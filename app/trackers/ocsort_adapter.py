from __future__ import annotations

import numpy as np

from .base import BaseTracker, Detection, Track, register
from .vendor.ocsort.ocsort import OCSort


class OCSortAdapter(BaseTracker):
    display_name = "OC-SORT"

    def __init__(self, det_thresh=0.5, max_age=30, min_hits=3, iou_threshold=0.3,
                 delta_t=3, asso_func="iou", inertia=0.2, use_byte=False, **_):
        self.tracker = OCSort(det_thresh=det_thresh, max_age=max_age, min_hits=min_hits,
                              iou_threshold=iou_threshold, delta_t=delta_t,
                              asso_func=asso_func, inertia=inertia, use_byte=use_byte)

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        dets = self._dets_to_array(detections)
        h, w = frame.shape[:2]
        online = self.tracker.update(dets, (h, w), (h, w))
        out = []
        for row in online:
            x1, y1, x2, y2, tid = row[:5]
            out.append(Track(track_id=int(tid), xyxy=np.array([x1, y1, x2, y2])))
        return out


register("ocsort", "OC-SORT (2023)", OCSortAdapter)
