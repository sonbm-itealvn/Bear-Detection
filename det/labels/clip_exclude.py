"""Spec §2 / plan Task 9 Step 3b: non_bear images whose CLIP ViT-L/14 5-fold out-of-fold
p(bear) >= 0.5 (audit_nonbear_1003/probe_oof.npz) are EXCLUDED from detector training
(heat = 0 everywhere would teach any bear in them as background). Files are never moved:
the store gets an 'excluded' row keyed by sha1.
python -m det.labels.clip_exclude --split train"""
import argparse, os
from pathlib import Path

import numpy as np

ROOT = Path("/data/nas07_new/PersonalData/phuocthien")


def clip_suspect_nonbear(oof_npz, root, split, thr=0.5):
    d = np.load(oof_npz, allow_pickle=True)
    files, y, oof = d["files"], d["y"], d["oof"]
    hot = {Path(str(f)).name for f, yy, p in zip(files, y, oof) if yy == 0 and p >= thr}
    nd = Path(root) / split / "non_bear"
    return sorted(nd / f for f in os.listdir(nd) if f in hot)


def main():
    from det.dataset import _sha1_cached
    from det.labels.store import LabelStore
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--store", default=str(ROOT / "det_labels/labels_v1.jsonl"))
    a = ap.parse_args()
    sus = clip_suspect_nonbear(ROOT / "audit_nonbear_1003/probe_oof.npz", ROOT / "bear_dataset_v_final_final", a.split, a.thr)
    sha = _sha1_cached(sus, ROOT / "det_labels/sha1_cache.json")
    st = LabelStore(a.store)
    n_new = 0
    for p in sus:
        s = sha[str(p)]
        r = st.get(s)
        if r is None:
            st.put({"sha1": s, "dhash": None, "w": None, "h": None, "path_hint": str(p), "cls": "non_bear",
                    "domain": "other", "status": "excluded", "boxes": [], "ignore": [], "n_det": 0,
                    "version": 1, "reviewer": None, "excluded_by": f"clip_oof>={a.thr}"})
            n_new += 1
        elif r["status"] != "reviewed":
            r["status"] = "excluded"; r["excluded_by"] = f"clip_oof>={a.thr}"; st.put(r); n_new += 1
    st.save()
    print(f"{a.split}: {len(sus)} non_bear with CLIP oof p(bear) >= {a.thr}; {n_new} rows marked excluded")


if __name__ == "__main__":
    main()
