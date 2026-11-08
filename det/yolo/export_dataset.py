"""Label store -> ultralytics YOLO dataset (symlinked images + txt labels, class 0 = bear).
Rules (plan Task 13b, amended 16/10): masked/excluded/unlabelled bears are dropped; an ignore
region that is >= 50% covered by an accepted box is a redundant fragment and is dropped; any
OTHER ignore region is filled with YOLO's letterbox gray (114) outside the accepted boxes, in a
materialised copy of the image (YOLO cannot ignore; dropping those rows lost 45% of the bears). Negatives: the hard list
(V19's top-20k) + n_random_nb random train non_bear. Val = the eval-set val half.
python -m det.yolo.export_dataset --out det_runs/yolo_ds"""
import argparse, json, os, random
from pathlib import Path

import numpy as np

from det.dataset import _sha1_cached
from det.labels.store import LabelStore

ROOT = Path("/data/nas07_new/PersonalData/phuocthien")


def residual_ignore(row):
    """Ignore regions not explained by an accepted box (< 50% of the region inside one)."""
    b = np.array([x["xyxy"] for x in row["boxes"]], float).reshape(-1, 4)
    out = []
    for g in np.asarray(row["ignore"], float).reshape(-1, 4):
        a = max((g[2] - g[0]) * (g[3] - g[1]), 1e-9)
        cov = max([max(0, min(g[2], x[2]) - max(g[0], x[0])) * max(0, min(g[3], x[3]) - max(g[1], x[1])) / a
                   for x in b] or [0.0])
        if cov < 0.5:
            out.append(g)
    return out


def gray_fill(src, dst, regions, keep_boxes):
    from det.labels.store import load_rgb
    from PIL import Image
    a = np.array(load_rgb(src))
    h, w = a.shape[:2]
    mask = np.zeros((h, w), bool)
    for x1, y1, x2, y2 in regions:
        mask[max(0, int(y1)):min(h, int(np.ceil(y2))), max(0, int(x1)):min(w, int(np.ceil(x2)))] = True
    for x1, y1, x2, y2 in keep_boxes:
        mask[max(0, int(y1)):min(h, int(np.ceil(y2))), max(0, int(x1)):min(w, int(np.ceil(x2)))] = False
    a[mask] = 114
    Image.fromarray(a).save(dst, "JPEG", quality=95)


def _yolo_line(b, w, h):
    x1, y1, x2, y2 = b
    return f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def _put(out, split, src, lines, fill=None, keep=()):
    src = Path(src)
    name = f"{src.parts[-3]}__{src.parts[-2]}__{src.name}"
    img = out / "images" / split / name
    if fill:
        if img.is_symlink():
            img.unlink()
        gray_fill(src, img, fill, keep)
    elif not img.exists():
        os.symlink(src, img)
    (out / "labels" / split / (Path(name).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))


def export(store_path, root, out, eval_set_path, hard_nb_path, n_random_nb=20000, seed=0, cache=None):
    root, out = Path(root), Path(out)
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    st = LabelStore(store_path)
    es = json.load(open(eval_set_path))
    eval_sha = set(sum((es[s]["bear"] + es[s]["non_bear"] for s in es), []))
    n = {k: 0 for k in ("train_bear", "train_non_bear", "val_bear", "val_non_bear",
                        "gray_filled", "dropped_masked", "unlabelled")}
    bears = sorted((root / "train/bear").iterdir())
    sha = _sha1_cached(bears, cache)
    for p in bears:
        s = sha[str(p)]
        if s in eval_sha:
            continue
        r = st.get(s)
        if r is None or r.get("cls") != "bear":
            n["unlabelled"] += 1
            continue
        if r["status"] in ("masked", "excluded") or not r["boxes"]:
            n["dropped_masked"] += 1
            continue
        res = residual_ignore(r)
        boxes = [b["xyxy"] for b in r["boxes"]]
        _put(out, "train", p, [_yolo_line(b, r["w"], r["h"]) for b in boxes], fill=res, keep=boxes)
        n["gray_filled"] += bool(res)
        n["train_bear"] += 1
    hard = [Path(l.strip()) for l in open(hard_nb_path) if l.strip()] if hard_nb_path and Path(hard_nb_path).exists() else []
    hard = [p for p in hard if p.exists()]
    pool = sorted(set((root / "train/non_bear").iterdir()) - set(hard))
    rand = random.Random(seed).sample(pool, min(n_random_nb, len(pool)))
    for p in hard + rand:
        _put(out, "train", p, [])
        n["train_non_bear"] += 1
    for s in es["val"]["bear"]:
        r = st.get(s)
        if r and r["status"] not in ("masked", "excluded") and r["boxes"]:
            _put(out, "val", r["path_hint"], [_yolo_line(b["xyxy"], r["w"], r["h"]) for b in r["boxes"]])
            n["val_bear"] += 1
    for s in es["val"]["non_bear"]:
        r = st.get(s)
        if r:
            _put(out, "val", r["path_hint"], [])
            n["val_non_bear"] += 1
    (out / "data.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: bear\n")
    json.dump(n, open(out / "export_counts.json", "w"), indent=1)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "det_runs/yolo_ds"))
    ap.add_argument("--store", default=str(ROOT / "det_labels/labels_v1.jsonl"))
    ap.add_argument("--n-random-nb", type=int, default=20000)
    a = ap.parse_args()
    print(export(a.store, ROOT / "bear_dataset_v_final_final", a.out, ROOT / "det_labels/eval_set.json",
                 ROOT / "det_runs/hard_nb.txt", a.n_random_nb, cache=ROOT / "det_labels/sha1_cache.json"))


if __name__ == "__main__":
    main()
