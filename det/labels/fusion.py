"""Weighted box fusion + tiering of YOLO/OWLv2 proposals (spec §2)."""
import numpy as np


def iou(a, b):
    a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def wbf(boxes, scores, iou_thr=0.55):
    """Greedy weighted box fusion. Returns fused boxes, fused scores (sum / number of
    sources) and, per fused box, the set of source indices that contributed."""
    allb, alls, srcs = [], [], []
    for si, (b, s) in enumerate(zip(boxes, scores)):
        for bb, ss in zip(np.asarray(b, float).reshape(-1, 4), np.asarray(s, float).reshape(-1)):
            allb.append(bb); alls.append(ss); srcs.append(si)
    groups, fused = [], []
    for i in (np.argsort(-np.asarray(alls)) if alls else []):
        best, bi = 0.0, -1
        for g, fb in enumerate(fused):
            v = iou(allb[i][None], fb[None])[0, 0]
            if v > best:
                best, bi = v, g
        if bi >= 0 and best >= iou_thr:
            groups[bi].append(i)
            w = np.array([alls[j] for j in groups[bi]])
            fused[bi] = (np.array([allb[j] for j in groups[bi]]) * w[:, None]).sum(0) / w.sum()
        else:
            groups.append([i]); fused.append(allb[i].copy())
    nsrc = max(len(boxes), 1)
    fb = np.array(fused).reshape(-1, 4)
    fs = np.array([sum(alls[j] for j in g) / nsrc for g in groups])
    return fb, fs, [set(srcs[j] for j in g) for g in groups]


def drop_group_boxes(boxes, scores, inside=0.8):
    """Drop a box that contains >= 2 other boxes (each >= `inside` of its area inside it):
    a 'group of bears' box on top of the individual bears."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    keep = []
    for i, b in enumerate(boxes):
        n_in = 0
        for j, c in enumerate(boxes):
            if i == j:
                continue
            ix = max(0, min(b[2], c[2]) - max(b[0], c[0])) * max(0, min(b[3], c[3]) - max(b[1], c[1]))
            if ix / max((c[2] - c[0]) * (c[3] - c[1]), 1e-9) >= inside:
                n_in += 1
        if n_in < 2:
            keep.append(i)
    return np.array(keep, int)


def tier_boxes(yolo_bear, owl, other, iou_thr=0.55, b_conf=0.5, owl_b=None, other_owl=False):
    """A = YOLO bear and OWLv2 agree; B = YOLO bear alone at conf >= b_conf, or (optional)
    OWLv2 alone at score >= owl_b, or (optional) OWLv2 + a YOLO non-bear class agreeing
    (YOLO often calls a bear cub 'dog'); C = the rest."""
    yolo_bear = np.asarray(yolo_bear, float).reshape(-1, 5)
    owl = np.asarray(owl, float).reshape(-1, 5)
    other = np.asarray(other, float).reshape(-1, 5)
    fb, fs, src = wbf([yolo_bear[:, :4], owl[:, :4], other[:, :4]],
                      [yolo_bear[:, 4], owl[:, 4], other[:, 4]], iou_thr)
    if len(fb):
        k = drop_group_boxes(fb, fs)
        fb, fs, src = fb[k], fs[k], [src[i] for i in k]
    out = []
    for b, s, ss in zip(fb, fs, src):
        if 0 in ss and 1 in ss:
            tier = "A"
        elif ss == {0} and yolo_bear[iou(b[None], yolo_bear[:, :4])[0].argmax(), 4] >= b_conf:
            tier = "B"
        elif owl_b is not None and ss == {1} and owl[iou(b[None], owl[:, :4])[0].argmax(), 4] >= owl_b:
            tier = "B"
        elif other_owl and ss == {1, 2}:
            tier = "B"
        else:
            tier = "C"
        out.append({"xyxy": [float(v) for v in b], "score": float(s), "tier": tier,
                    "src": "+".join(("yolo", "owl", "other")[i] for i in sorted(ss))})
    return out
