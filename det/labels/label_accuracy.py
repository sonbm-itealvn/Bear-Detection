"""Accuracy of the automatic boxes against the human review (spec §2 acceptance rule:
per domain instance precision >= 95%, recall >= 90% at IoU 0.5, median IoU >= 0.80).
Auto boxes = the pre-review store rows; final boxes = what the reviewer accepted or drew."""
import json
from collections import defaultdict

import numpy as np

from det.labels.fusion import iou


def final_boxes(item, auto_row):
    if item["decision"] == "nobear":
        return np.zeros((0, 4))
    if item["decision"] == "ok":
        return np.array([b["xyxy"] for b in auto_row["boxes"]], float).reshape(-1, 4)
    return np.array([b[:4] for b in item["boxes"]], float).reshape(-1, 4)


def match(auto, final, thr=0.5):
    """Greedy one-to-one matching by IoU; returns matched IoUs."""
    if not len(auto) or not len(final):
        return []
    m = iou(auto, final)
    out = []
    while m.size and m.max() >= thr:
        i, j = np.unravel_index(m.argmax(), m.shape)
        out.append(float(m[i, j]))
        m[i, :] = -1; m[:, j] = -1
    return out


def accuracy(items, auto_rows, key=lambda r: r["domain"]):
    agg = defaultdict(lambda: {"n_img": 0, "auto": 0, "final": 0, "tp": 0, "ious": [], "masked": 0,
                               "tierA": [0, 0], "tierB": [0, 0], "dec": defaultdict(int)})
    for it in items:
        r = auto_rows.get(it["sha1"])
        if r is None or r["cls"] != "bear":
            continue
        for k in (key(r), "ALL"):
            a = agg[k]
            auto = np.array([b["xyxy"] for b in r["boxes"]], float).reshape(-1, 4)
            fin = final_boxes(it, r)
            ious = match(auto, fin)
            a["n_img"] += 1; a["auto"] += len(auto); a["final"] += len(fin); a["tp"] += len(ious)
            a["ious"] += ious; a["masked"] += r["status"] == "masked"; a["dec"][it["decision"]] += 1
            for b in r["boxes"]:                     # per-tier precision
                t = "tier" + b.get("tier", "A")
                if t in a:
                    a[t][0] += 1
                    a[t][1] += bool(match(np.array([b["xyxy"]], float), fin))
    rows = {}
    for k, a in agg.items():
        rows[k] = {"n_img": a["n_img"], "masked": a["masked"], "decisions": dict(a["dec"]),
                   "precision": a["tp"] / max(a["auto"], 1), "recall": a["tp"] / max(a["final"], 1),
                   "median_iou": float(np.median(a["ious"])) if a["ious"] else 0.0,
                   "auto_boxes": a["auto"], "final_boxes": a["final"],
                   "prec_tierA": a["tierA"][1] / max(a["tierA"][0], 1), "n_tierA": a["tierA"][0],
                   "prec_tierB": a["tierB"][1] / max(a["tierB"][0], 1), "n_tierB": a["tierB"][0]}
        r = rows[k]
        r["pass"] = r["precision"] >= 0.95 and r["recall"] >= 0.90 and r["median_iou"] >= 0.80
    return rows
