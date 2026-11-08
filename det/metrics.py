"""Detection metrics on per-image prediction arrays [K,5] (x1,y1,x2,y2,score) and
ground truth [N,4], all in the same pixel space."""
import numpy as np
from det.labels.fusion import iou


def _match(preds, gts, iou_thr, gt_ignore=None, pred_ignore=None):
    """Greedy score-ordered matching per image -> list of (score, is_tp), n_gt.
    COCO-style ignores: a prediction matching an ignored ground truth, or an unmatched
    prediction flagged in pred_ignore, is neither TP nor FP; ignored gts are not counted."""
    recs, n_gt = [], 0
    for k_img, (p, g) in enumerate(zip(preds, gts)):
        p = np.asarray(p, float).reshape(-1, 5); g = np.asarray(g, float).reshape(-1, 4)
        gi = np.zeros(len(g), bool) if gt_ignore is None else np.asarray(gt_ignore[k_img], bool)
        pi = np.zeros(len(p), bool) if pred_ignore is None else np.asarray(pred_ignore[k_img], bool)
        n_gt += int((~gi).sum())
        used = np.zeros(len(g), bool)
        for k in np.argsort(-p[:, 4], kind="stable"):
            v = iou(p[k:k + 1, :4], g)[0] if len(g) else np.zeros(0)
            hit = None
            for want_ignored in (False, True):          # real gts first, then ignored ones
                cand = np.where((~used) & (gi == want_ignored) & (v >= iou_thr))[0] if len(g) else []
                if len(cand):
                    hit = cand[np.argmax(v[cand])]
                    break
            if hit is not None:
                used[hit] = True
                if gi[hit]:
                    continue                              # matched an ignored gt: not counted
                recs.append((p[k, 4], 1))
            elif not pi[k]:
                recs.append((p[k, 4], 0))
    return recs, n_gt


def ap_at(preds, gts, iou_thr, gt_ignore=None, pred_ignore=None):
    """All-point interpolated average precision (optional COCO-style ignore flags)."""
    recs, n_gt = _match(preds, gts, iou_thr, gt_ignore, pred_ignore)
    if not recs or n_gt == 0:
        return 0.0
    recs.sort(key=lambda r: -r[0])
    tps = np.cumsum([r[1] for r in recs]); fps = np.cumsum([1 - r[1] for r in recs])
    rec = tps / n_gt; prec = tps / np.maximum(tps + fps, 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def ap_range(preds, gts):
    """Mean AP over IoU 0.50:0.95 (step 0.05)."""
    return float(np.mean([ap_at(preds, gts, t) for t in np.arange(0.5, 0.951, 0.05)]))


def recall_at_fppi(preds_pos, gts_pos, preds_neg, fppi):
    """Box recall (IoU >= 0.5) at the score threshold that allows `fppi` false boxes per
    negative image on average."""
    neg = [np.asarray(p, float).reshape(-1, 5)[:, 4] for p in preds_neg]
    neg_scores = np.sort(np.concatenate(neg) if neg else np.zeros(0))[::-1]
    k = int(fppi * len(preds_neg))
    thr = neg_scores[k] if k < len(neg_scores) else -np.inf
    hit = 0
    for p, g in zip(preds_pos, gts_pos):
        p = np.asarray(p, float).reshape(-1, 5); g = np.asarray(g, float).reshape(-1, 4)
        p = p[p[:, 4] > thr]
        if len(p) and len(g):
            hit += int((iou(g, p[:, :4]).max(1) >= 0.5).sum())
    return hit / max(sum(len(np.asarray(g).reshape(-1, 4)) for g in gts_pos), 1)


def iou_stats(ious):
    a = np.asarray(ious, float)
    if not len(a):
        return {"median": 0.0, "p10": 0.0, "n": 0}
    return {"median": float(np.median(a)), "p10": float(np.percentile(a, 10)), "n": int(len(a))}
