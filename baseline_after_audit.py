"""V16 / V17 FP32-equivalent weights on val and test, before and after the
03/10 non_bear audit quarantine -- the new baseline V18 is compared against.
'before' = current split + the images quarantined from it (both classes).

    python baseline_after_audit.py --gpu 4
"""
import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from score_darkfur import Letterbox, load

BASE = Path("/data/nas07_new/PersonalData/phuocthien")
DS = BASE / "bear_dataset_v_final_final"
Q = BASE / "quarantine_audit_1003"
MODELS = {"V16": BASE / "deliver_qat_pow2_20260917/checkpoint_fp32_equivalent.pth",
          "V17": BASE / "deliver_qat_pow2_20261002/checkpoint_fp32_equivalent.pth"}


def items(split):
    cur = [(str(DS / split / c / n), c == "bear") for c in ("bear", "non_bear")
           for n in os.listdir(DS / split / c)]
    q = [(str(Q / split / c / n), c == "bear") for c in ("bear", "non_bear")
         if (Q / split / c).is_dir() for n in os.listdir(Q / split / c)]
    return cur, q


def metrics(pred, y):
    tp = sum(p and t for p, t in zip(pred, y)); fp = sum(p and not t for p, t in zip(pred, y))
    fn = sum(t and not p for p, t in zip(pred, y))
    r, pr = tp / max(1, tp + fn), tp / max(1, tp + fp)
    return r, pr, 2 * r * pr / max(1e-9, r + pr), fp


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=0); a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    nets = {k: load(str(v), dev) for k, v in MODELS.items()}
    tf = transforms.Compose([Letterbox(), transforms.ToTensor()])
    mean = torch.tensor([104, 116, 124], device=dev).view(1, 3, 1, 1) / 255
    out = {}
    for split in ("val", "test"):
        cur, q = items(split)
        allit = cur + q

        class D(Dataset):
            def __len__(self): return len(allit)
            def __getitem__(self, i): return tf(Image.open(allit[i][0]).convert("RGB")), i
        preds = {k: [None] * len(allit) for k in nets}
        with torch.no_grad():
            for x, idx in DataLoader(D(), batch_size=64, num_workers=10):
                x = x.to(dev)[:, [2, 1, 0]] - mean
                for k, n in nets.items():
                    p = n(x).argmax(1).cpu().tolist()
                    for i, v in zip(idx.tolist(), p):
                        preds[k][i] = bool(v)
        y = [t for _, t in allit]
        nc = len(cur)
        for k in nets:
            b = metrics(preds[k], y); c = metrics(preds[k][:nc], y[:nc])
            out[f"{split}/{k}"] = {"before": b, "after": c}
            print(f"{split:<5}{k:<5} before: R {100*b[0]:.2f}  P {100*b[1]:.2f}  F1 {100*b[2]:.2f}  FP {b[3]:>4}"
                  f"   after: R {100*c[0]:.2f}  P {100*c[1]:.2f}  F1 {100*c[2]:.2f}  FP {c[3]:>4}", flush=True)
    json.dump(out, open(BASE / "audit_nonbear_1003/baseline_after_audit.json", "w"), indent=1)


if __name__ == "__main__":
    main()
