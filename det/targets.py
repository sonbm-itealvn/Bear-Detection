"""TTFNet-style targets on the stride grid (spec §3). Boxes are in LETTERBOX pixels."""
import numpy as np
from det.geometry import GRID


def cell_centers(stride):
    H, W = GRID[stride]
    return (np.arange(W) + 0.5) * stride, (np.arange(H) + 0.5) * stride


def encode(boxes_lb, ignore_lb, stride, masked, alpha=0.54):
    """heat: elliptical Gaussian per box, exactly 1.0 at the cell containing the centre.
    box: log(distance / stride) from each cell centre to the l,t,r,b edges, written where
    the Gaussian is inside the box. box_w: Gaussian x log(area), normalised to sum 1 per
    image. heat_w: 0 inside ignore boxes (and everywhere when masked)."""
    H, W = GRID[stride]
    cx, cy = cell_centers(stride)
    gx, gy = np.meshgrid(cx, cy)
    heat = np.zeros((1, H, W), np.float32)
    heat_w = np.ones((1, H, W), np.float32)
    box = np.zeros((4, H, W), np.float32)
    box_w = np.zeros((1, H, W), np.float32)
    boxes = np.asarray(boxes_lb, float).reshape(-1, 4)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for k in np.argsort(-areas):                       # small boxes written last (win overlaps)
        x1, y1, x2, y2 = boxes[k]
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        bx, by = (x1 + x2) / 2, (y1 + y2) / 2
        sx, sy = max(alpha * w / 6, stride / 4), max(alpha * h / 6, stride / 4)
        g = np.exp(-((gx - bx) ** 2) / (2 * sx ** 2) - ((gy - by) ** 2) / (2 * sy ** 2))
        pi, pj = int(np.clip(by // stride, 0, H - 1)), int(np.clip(bx // stride, 0, W - 1))
        g = np.minimum(g / g[pi, pj], 1.0)             # the centre cell is the maximum
        heat[0] = np.maximum(heat[0], g)
        inside = (gx > x1) & (gx < x2) & (gy > y1) & (gy < y2)
        inside[pi, pj] = True                          # the peak always regresses its box
        reg = g * inside
        wgt = reg / reg.sum() * np.log(max(w * h, 2.0))
        upd = wgt > box_w[0]
        box_w[0][upd] = wgt[upd]
        d = np.stack([gx - x1, gy - y1, x2 - gx, y2 - gy])
        d = np.log(np.clip(d, 1e-3, None) / stride)
        box[:, upd] = d[:, upd]
    for x1, y1, x2, y2 in np.asarray(ignore_lb, float).reshape(-1, 4):
        heat_w[0][(gx >= x1) & (gx <= x2) & (gy >= y1) & (gy <= y2)] = 0
    if box_w.sum() > 0:
        box_w /= box_w.sum()
    if masked:
        heat_w[:] = 0
        box_w[:] = 0
    return {"heat": heat, "heat_w": heat_w, "box": box, "box_w": box_w, "n_pos": int(len(boxes))}
