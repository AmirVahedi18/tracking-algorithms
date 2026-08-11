from __future__ import annotations

import numpy as np

from .base import BaseTracker, Detection, Track, register, ns
from .vendor.hybridsort.hybrid_sort import Hybrid_Sort


class HybridSortAdapter(BaseTracker):
    display_name = "Hybrid-SORT"

    def __init__(self, det_thresh=0.6, max_age=30, min_hits=3, iou_threshold=0.3,
                 delta_t=3, asso_func="iou", inertia=0.2, use_byte=True,
                 track_thresh=0.6, tcm_first_weight=1.0, tcm_byte_weight=1.0, **_):
        args = ns(track_thresh=track_thresh, kalman_GPR=False,
                  TCM_first_step=True, TCM_byte_step=True,
                  TCM_first_step_weight=tcm_first_weight,
                  TCM_byte_step_weight=tcm_byte_weight)
        self.tracker = Hybrid_Sort(args, det_thresh=det_thresh, max_age=max_age,
                                   min_hits=min_hits, iou_threshold=iou_threshold,
                                   delta_t=delta_t, asso_func=asso_func,
                                   inertia=inertia, use_byte=use_byte)

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        dets = self._dets_to_array(detections)
        h, w = frame.shape[:2]
        online = self.tracker.update(dets, (h, w), (h, w))
        out = []
        for row in online:
            x1, y1, x2, y2, tid = row[:5]
            out.append(Track(track_id=int(tid), xyxy=np.array([x1, y1, x2, y2])))
        return out


register("hybridsort", "Hybrid-SORT (2024)", HybridSortAdapter)
