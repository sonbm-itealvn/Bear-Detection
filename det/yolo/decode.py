"""Reference decoder for the raw YOLOv8 head exported by det/probe_yolo.RawHead:
6 outputs [box_p3, cls_p3, box_p4, cls_p4, box_p5, cls_p5], each (C,H,W) for one image.
box: 4 sides x 16 DFL bins (l,t,r,b), cls: 1 logit. Same maths as ultralytics
(make_anchors offset 0.5, DFL softmax expectation, dist2bbox * stride)."""
import numpy as np
from det.yolo.geometry import STRIDES, lb_to_orig

REG_MAX = 16
FALLBACK_TRIES = 64     # strongest anchors tried, in order, when nothing valid survives


def _softmax(a, axis):
    a = a - a.max(axis, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis, keepdims=True)


def decode_raw(outs):
    """All anchors -> boxes xyxy in NETWORK-input pixels [N,4] and sigmoid scores [N]."""
    boxes, scores = [], []
    for lvl, s in enumerate(STRIDES):
        box, cls = np.asarray(outs[2 * lvl], np.float64), np.asarray(outs[2 * lvl + 1], np.float64)
        _, h, w = cls.shape
        d = _softmax(box.reshape(4, REG_MAX, h, w), 1)
        d = (d * np.arange(REG_MAX).reshape(1, REG_MAX, 1, 1)).sum(1)            # 4,h,w in grid units
        gy, gx = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
        xyxy = np.stack([gx - d[0], gy - d[1], gx + d[2], gy + d[3]]) * s
        boxes.append(xyxy.reshape(4, -1).T)
        scores.append((1 / (1 + np.exp(-cls[0]))).reshape(-1))
    return np.concatenate(boxes), np.concatenate(scores)


def _nms_unused(b, s, iou_thr):
    order = np.argsort(-s, kind="stable")
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        r = order[1:]
        ix = np.clip(np.minimum(b[i, 2], b[r, 2]) - np.maximum(b[i, 0], b[r, 0]), 0, None)
        iy = np.clip(np.minimum(b[i, 3], b[r, 3]) - np.maximum(b[i, 1], b[r, 1]), 0, None)
        inter = ix * iy
        u = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1]) + (b[r, 2] - b[r, 0]) * (b[r, 3] - b[r, 1]) - inter
        order = r[inter / np.maximum(u, 1e-9) <= iou_thr]
    return keep


def decode(outs, W0, H0, thr=0.3, K=20, nms=0.5, pre_nms=256, fallback=False):
    """Boxes (x1,y1,x2,y2,score) in ORIGINAL pixels, score-descending, at most K.
    Candidates: anchors with score >= thr, best pre_nms of them (score desc, anchor index asc).
    fallback: if none reaches thr, the single best anchor is still returned (use when the V19
    classifier already said 'bear'). Mirrored exactly by det/c/yolo_det_decode.c."""
    b, s = decode_raw(outs)
    idx = np.where(s >= thr)[0]
    idx = idx[np.lexsort((idx, -s[idx]))][:pre_nms]
    keep = []
    for i in idx:
        ok = True
        for j in keep:
            ix = max(0.0, min(b[i, 2], b[j, 2]) - max(b[i, 0], b[j, 0]))
            iy = max(0.0, min(b[i, 3], b[j, 3]) - max(b[i, 1], b[j, 1]))
            inter = ix * iy
            u = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1]) + (b[j, 2] - b[j, 0]) * (b[j, 3] - b[j, 1]) - inter
            if u > 0 and inter / u > nms:
                ok = False
                break
        if ok:
            keep.append(i)
            if len(keep) == K:
                break
    o = lb_to_orig(b[keep], W0, H0, clip=True).reshape(-1, 4)
    out = np.concatenate([o, s[keep, None]], 1) if len(keep) else np.zeros((0, 5))
    out = out[(out[:, 2] - out[:, 0] >= 1) & (out[:, 3] - out[:, 1] >= 1)]
    if not len(out) and fallback:          # V19 said bear: always return one valid box
        allidx = np.arange(len(s))
        for i in allidx[np.lexsort((allidx, -s))][:FALLBACK_TRIES]:
            o = lb_to_orig(b[i:i + 1], W0, H0, clip=True)[0]
            if o[2] - o[0] >= 1 and o[3] - o[1] >= 1:
                out = np.array([[*o, s[i]]])
                break
    return out.astype(np.float32)
