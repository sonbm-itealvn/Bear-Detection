"""Reference decoder; det/c/bear_det_decode.c must reproduce it (<=1 px)."""
import math
import numpy as np
from det.geometry import letterbox_params

FALLBACK_TRIES = 64     # strongest peaks tried, in order, when nothing valid survives


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def decode(logits, heat, box, W0, H0, stride=16, thr=0.3, K=20, nms=0.5, gate=True):
    """Boxes (x1,y1,x2,y2,score) in ORIGINAL pixels, score-descending.
    gate: return nothing unless the classifier says bear (logit[1] > logit[0]).
    A 3x3 local maximum is a candidate; candidates >= thr are kept (top K, then NMS). If no
    valid box survives (e.g. every peak lies in a letterbox bar), the strongest peak that gives a
    valid clipped box is returned -- one box whenever the classifier said bear."""
    logits = np.asarray(logits, np.float32).reshape(-1)
    if gate and not (logits[1] > logits[0]):
        return np.zeros((0, 5), np.float32)
    z = np.asarray(heat, np.float32).reshape(heat.shape[-2], heat.shape[-1])
    bx = np.asarray(box, np.float32).reshape(4, *z.shape)
    Hg, Wg = z.shape
    zthr = math.log(thr / (1 - thr))
    cands = []
    for i in range(Hg):
        for j in range(Wg):
            v = z[i, j]
            if v >= z[max(0, i - 1):i + 2, max(0, j - 1):j + 2].max():
                cands.append((float(v), i, j))
    cands.sort(key=lambda c: (-c[0], c[1], c[2]))
    keep = [c for c in cands if c[0] >= zthr][:K]
    nw, nh, px, py = letterbox_params(W0, H0)
    sx, sy = nw / W0, nh / H0

    def to_box(v, i, j):
        cx, cy = (j + 0.5) * stride, (i + 0.5) * stride
        l, t, r, b = (math.exp(float(d)) * stride for d in bx[:, i, j])
        x1 = min(max((cx - l - px) / sx, 0.0), W0); y1 = min(max((cy - t - py) / sy, 0.0), H0)
        x2 = min(max((cx + r - px) / sx, 0.0), W0); y2 = min(max((cy + b - py) / sy, 0.0), H0)
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None                    # degenerate once clipped (e.g. inside a letterbox bar)
        return [x1, y1, x2, y2, 1.0 / (1.0 + math.exp(-v))]

    final = []
    for c in keep:                         # keep is score-sorted
        bb = to_box(*c)
        if bb is not None and all(_iou(bb, f) <= nms for f in final):
            final.append(bb)
    if not final:                          # the classifier said bear: always return one valid box
        for c in cands[:FALLBACK_TRIES]:
            bb = to_box(*c)
            if bb is not None:
                final = [bb]
                break
    return np.array(final, np.float32).reshape(-1, 5)
