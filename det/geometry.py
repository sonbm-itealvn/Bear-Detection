"""Letterbox geometry shared by labels, training, decoding and the C decoder.
Must match ResizeWithPadding in train_qat_novaic.py / score_darkfur.Letterbox."""
import numpy as np

IN_W, IN_H = 640, 360
GRID = {16: (23, 40), 8: (45, 80)}


def letterbox_params(w, h):
    s = min(IN_W / w, IN_H / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    return nw, nh, (IN_W - nw) // 2, (IN_H - nh) // 2


def orig_to_lb(boxes, w, h):
    nw, nh, px, py = letterbox_params(w, h)
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    sx, sy = nw / w, nh / h
    return b * np.array([sx, sy, sx, sy]) + np.array([px, py, px, py])


def lb_to_orig(boxes, w, h, clip=True):
    nw, nh, px, py = letterbox_params(w, h)
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    sx, sy = nw / w, nh / h
    o = (b - np.array([px, py, px, py])) / np.array([sx, sy, sx, sy])
    if clip:
        o[:, [0, 2]] = o[:, [0, 2]].clip(0, w)
        o[:, [1, 3]] = o[:, [1, 3]].clip(0, h)
    return o
