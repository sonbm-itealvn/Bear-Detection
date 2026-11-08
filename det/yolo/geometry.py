"""Letterbox geometry for the YOLO candidate: network input 640x384 (stride-32 multiple).
A 16:9 camera frame becomes 640x360 centred with 12-px black bars top and bottom."""
import numpy as np

IN_W, IN_H = 640, 384
STRIDES = (8, 16, 32)


def letterbox_params(w, h):
    s = min(IN_W / w, IN_H / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    return nw, nh, (IN_W - nw) // 2, (IN_H - nh) // 2


def orig_to_lb(boxes, w, h):
    nw, nh, px, py = letterbox_params(w, h)
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return b * np.array([nw / w, nh / h, nw / w, nh / h]) + np.array([px, py, px, py])


def lb_to_orig(boxes, w, h, clip=True):
    nw, nh, px, py = letterbox_params(w, h)
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    o = (b - np.array([px, py, px, py])) / np.array([nw / w, nh / h, nw / w, nh / h])
    if clip:
        o[:, [0, 2]] = o[:, [0, 2]].clip(0, w)
        o[:, [1, 3]] = o[:, [1, 3]].clip(0, h)
    return o
