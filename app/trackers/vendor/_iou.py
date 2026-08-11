import numpy as np


def bbox_ious(boxes, query_boxes):
    boxes = np.ascontiguousarray(boxes, dtype=np.float64)
    query = np.ascontiguousarray(query_boxes, dtype=np.float64)
    N, K = boxes.shape[0], query.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)
    if N == 0 or K == 0:
        return overlaps

    q_area = (query[:, 2] - query[:, 0] + 1) * (query[:, 3] - query[:, 1] + 1)
    b_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    iw = np.minimum(boxes[:, None, 2], query[None, :, 2]) - np.maximum(boxes[:, None, 0], query[None, :, 0]) + 1
    ih = np.minimum(boxes[:, None, 3], query[None, :, 3]) - np.maximum(boxes[:, None, 1], query[None, :, 1]) + 1
    iw = np.clip(iw, 0, None)
    ih = np.clip(ih, 0, None)

    inter = iw * ih
    union = b_area[:, None] + q_area[None, :] - inter
    np.divide(inter, union, out=overlaps, where=union > 0)
    return overlaps
