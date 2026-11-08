"""Proposals -> label-store rows (spec §2).
Tier A/B boxes become training boxes; tier C boxes and low-confidence YOLO bears become
ignore regions. A bear image with no A/B box is 'masked' (no heatmap loss at all).
non_bear images are pure negatives: never any box, never dropped because a detector
fired (spec: never exclude a negative only because YOLO saw a bear)."""
import argparse, json
from pathlib import Path
import numpy as np
from det.labels.fusion import tier_boxes
from det.labels.store import LabelStore, dhash

BEAR = 21          # COCO class id


def build_rows(prop, thr_yolo=0.25, thr_owl=0.15, b_conf=0.5, owl_b=None, other_owl=False):
    cls = Path(prop["path"]).parent.name
    w, h = prop["w"], prop["h"]
    row = {"sha1": prop["sha1"], "dhash": None, "w": w, "h": h, "path_hint": prop["path"],
           "cls": cls, "domain": prop["domain"], "status": "auto", "boxes": [],
           "ignore": [], "n_det": 0, "version": 1, "reviewer": None}
    if cls != "bear":
        return row
    y = np.array(prop["yolo"], float).reshape(-1, 6)
    o = np.array(prop["owl"], float).reshape(-1, 6)
    yb = y[(y[:, 5] == BEAR)][:, :5]
    other = y[(y[:, 5] != BEAR) & (y[:, 4] >= thr_yolo)][:, :5]
    ow = o[o[:, 4] >= thr_owl][:, :5]
    for t in tier_boxes(yb[yb[:, 4] >= thr_yolo], ow, other, b_conf=b_conf, owl_b=owl_b, other_owl=other_owl):
        x1, y1, x2, y2 = t["xyxy"]
        if t["tier"] in ("A", "B"):
            trunc = x1 <= 2 or y1 <= 2 or x2 >= w - 2 or y2 >= h - 2
            row["boxes"].append({**t, "truncated": bool(trunc), "crowd": False})
        else:
            row["ignore"].append(t["xyxy"])
    row["ignore"] += [list(map(float, b[:4])) for b in yb[(yb[:, 4] >= 0.05) & (yb[:, 4] < thr_yolo)]]
    row["n_det"] = len(row["boxes"])
    if not row["boxes"]:
        row["status"] = "masked"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", nargs="+", required=True)
    ap.add_argument("--store", required=True)
    a = ap.parse_args()
    st = LabelStore(a.store)
    n = kept = 0
    for f in a.proposals:
        for line in open(f):
            p = json.loads(line)
            old = st.get(p["sha1"])
            if old and old.get("status") in ("reviewed", "corrected", "excluded"):
                kept += 1
                continue                       # never overwrite human decisions
            r = build_rows(p)
            r["dhash"] = dhash(p["path"])
            st.put(r); n += 1
    st.save()
    print(f"wrote {n} rows, kept {kept} human-reviewed rows -> {a.store}")


if __name__ == "__main__":
    main()
