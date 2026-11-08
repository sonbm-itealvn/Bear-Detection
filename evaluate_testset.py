"""Evaluate a bear/non-bear checkpoint on a dataset split (FP32, PyTorch).

Preprocessing is exactly what the camera/chip sees: letterbox to 640x360 (black bars),
BGR, (x - [104,116,124]) / 255. Decision = argmax (logit[1] > logit[0]); no threshold.

Reports recall / precision / F1 / FPR on the whole split and on the "novel" subset
(images whose nearest train image has CLIP ViT-L/14 cosine < 0.90, list from the
2026-10-09 scene re-split), and writes per-image margins to JSON.

    python evaluate_testset.py \
        --checkpoint 04_Model_Weights/v18_regnetx_ep58_fp32.pth \
        --data-dir bear_dataset_v_final_final --split test \
        --novel-list resplit_1009/test_novel.txt --device cuda:0 --out v18_test_eval.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from score_darkfur import Letterbox, load


class Split(Dataset):
    def __init__(self, files):
        self.files = files
        self.tf = transforms.Compose([Letterbox(), transforms.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            return self.tf(Image.open(self.files[i][0]).convert("RGB")), i
        except Exception:
            return torch.zeros(3, 360, 640), -1 - i      # unreadable file, reported separately


def metrics(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    r = tp / max(tp + fn, 1); pr = tp / max(tp + fp, 1)
    return {"n": len(y), "bears": int(y.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": round(100 * r, 2), "precision": round(100 * pr, 2),
            "f1": round(100 * 2 * r * pr / max(r + pr, 1e-9), 2), "fpr": round(100 * fp / max(fp + tn, 1), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default="bear_dataset_v_final_final")
    ap.add_argument("--split", default="test")
    ap.add_argument("--novel-list", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = Path(a.data_dir).resolve() / a.split
    files = [(str(p), c == "bear") for c in ("bear", "non_bear") for p in sorted((root / c).iterdir())
             if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    tail = lambda p: "/".join(Path(p).parts[-3:])          # <split>/<class>/<file>: independent of the folder prefix
    novel = set(tail(l.strip()) for l in open(a.novel_list) if l.strip()) if a.novel_list else set()
    net = load(a.checkpoint, a.device)
    mean = torch.tensor([104, 116, 124], device=a.device).view(1, 3, 1, 1) / 255
    margin = np.full(len(files), np.nan)
    bad = []
    with torch.no_grad():
        for x, idx in DataLoader(Split(files), batch_size=64, num_workers=a.workers):
            x = x.to(a.device)[:, [2, 1, 0]] - mean          # RGB tensor -> BGR, mean-subtracted
            o = net(x).float().cpu()
            for k, i in enumerate(idx.tolist()):
                if i < 0:
                    bad.append(files[-1 - i][0])
                else:
                    margin[i] = float(o[k, 1] - o[k, 0])
    ok = ~np.isnan(margin)
    y = np.array([b for _, b in files])
    res = {"checkpoint": a.checkpoint, "split": a.split, "unreadable": bad,
           "all": metrics(y[ok], margin[ok] > 0)}
    if novel:
        m = ok & np.array([tail(f) in novel for f, _ in files])
        if not m.any():
            raise SystemExit(f"--novel-list {a.novel_list} matches none of the {a.split} images")
        res["novel"] = metrics(y[m], margin[m] > 0)
    print(json.dumps({k: v for k, v in res.items() if k != "unreadable"}, indent=1), f"\nunreadable files: {len(bad)}")
    if a.out:
        res["margins"] = {f: (None if np.isnan(v) else round(float(v), 4)) for (f, _), v in zip(files, margin)}
        Path(a.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
