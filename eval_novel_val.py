"""Validation split by how novel each image is relative to train.

val/ and test/ were split by perceptual hash, not by source scene, so many val
images have a near-twin in train (CLIP ViT-L/14 cosine >= 0.95 for 43% of val
bears). On those, V17 scores 98-100% recall; on the 17% of val bears with no
train neighbour above 0.90 it scores 81%. Overall val F1 therefore mostly
measures memory of near-duplicates. This reports every model per similarity
band, and "novel" (< 0.90) is the number to select on.

Similarity comes from the audit's CLIP embeddings (audit_nonbear_1003/clip_*.npz);
quarantined images are dropped.

    python eval_novel_val.py --gpu 2 --model V17=<ckpt> --model V18=<ckpt> ...
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from score_darkfur import Letterbox, load

BASE = Path("/data/nas07_new/PersonalData/phuocthien")
BANDS = ((0.0, 0.90, "novel (<0.90)"), (0.90, 0.95, "similar (0.90-0.95)"),
         (0.95, 0.98, "very similar (0.95-0.98)"), (0.98, 1.01, "near-dup (>=0.98)"))


def similarity(split, dev):
    cache = BASE / f"audit_nonbear_1003/maxcos_{split}.npz"
    if cache.exists():
        d = np.load(cache)
        # Images quarantined after the cache was built (audit round 2, 07/10)
        # are dropped; listing the dirs is far faster than stat-ing each file.
        present = set()
        for cls in ("bear", "non_bear"):
            dd = BASE / "bear_dataset_v_final_final" / split / cls
            present.update(str(dd / n) for n in os.listdir(dd))
        keep = np.array([f in present for f in d["files"]])
        return d["files"][keep], d["y"][keep], d["maxcos"][keep]
    parts = [np.load(f) for f in sorted(glob.glob(str(BASE / "audit_nonbear_1003/clip_*of3.npz")))]
    E = np.concatenate([p["E"] for p in parts])
    files = np.concatenate([p["files"] for p in parts])
    lab = np.concatenate([p["labels"] for p in parts])
    keep = np.array([os.path.exists(f) for f in files])
    E, files, lab = E[keep], files[keep], lab[keep]
    sp = np.array([f.split("/")[-3] for f in files])
    T = torch.tensor(E[sp == "train"], dtype=torch.float16, device=dev)
    idx = np.where(sp == split)[0]
    Q = torch.tensor(E[idx], dtype=torch.float16, device=dev)
    m = torch.cat([(Q[i:i + 1024] @ T.T).max(1).values.float().cpu()
                   for i in range(0, len(Q), 1024)]).numpy()
    files, y = files[idx], (lab[idx] == "bear").astype(int)
    np.savez(cache, files=files, y=y, maxcos=m)
    return files, y, m


def build(ckpt, dev):
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    bb = (raw.get("config") or {}).get("backbone", "torchvision_mobilenet_v2")
    if bb == "torchvision_mobilenet_v2":
        return load(ckpt, dev)
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    import timm

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = timm.create_model(bb, pretrained=False, num_classes=0)
            self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.m.num_features, 2))

        def forward(self, x):
            return self.classifier(torch.flatten(F.adaptive_avg_pool2d(self.m.forward_features(x), 1), 1))
    net = Net()
    sd = {k.replace("module.", "").replace("features.m.", "m.", 1): v
          for k, v in raw["model_state_dict"].items() if not k.startswith("projection.")}
    net.load_state_dict(sd, strict=True)
    return net.eval().to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--model", action="append", required=True, help="NAME=checkpoint")
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    files, y, m = similarity(a.split, dev)
    torch.cuda.empty_cache()
    tf = transforms.Compose([Letterbox(), transforms.ToTensor()])

    class D(Dataset):
        def __len__(self):
            return len(files)

        def __getitem__(self, i):
            return tf(Image.open(files[i]).convert("RGB")), i

    mean = torch.tensor([104, 116, 124], device=dev).view(1, 3, 1, 1) / 255
    print(f"{a.split}: {len(files)} images; per band (bears / non_bears): " + ", ".join(
        f"{n} {((m >= lo) & (m < hi) & (y == 1)).sum()}/{((m >= lo) & (m < hi) & (y == 0)).sum()}"
        for lo, hi, n in BANDS))
    hdr = f"{'model':<12}{'all R':>8}{'all P':>8}{'all F1':>8}" + "".join(f"{'R ' + n.split()[0]:>14}" for *_, n in BANDS) + f"{'novel FP%':>11}{'novel F1':>10}"
    print(hdr)
    for spec in a.model:
        name, ck = spec.split("=", 1)
        net = build(ck, dev)
        pred = np.zeros(len(files), int)
        with torch.no_grad():
            for x, i in DataLoader(D(), batch_size=16, num_workers=8):
                pred[i.numpy()] = net(x.to(dev)[:, [2, 1, 0]] - mean).argmax(1).cpu().numpy()
        del net
        torch.cuda.empty_cache()

        def prf(mask):
            tp = ((pred == 1) & (y == 1) & mask).sum(); fp = ((pred == 1) & (y == 0) & mask).sum()
            fn = ((pred == 0) & (y == 1) & mask).sum()
            r, p = tp / max(1, tp + fn), tp / max(1, tp + fp)
            return r, p, 2 * r * p / max(1e-9, r + p), fp / max(1, ((y == 0) & mask).sum())
        allm = np.ones(len(files), bool)
        R, P, F1, _ = prf(allm)
        row = f"{name:<12}{100*R:>8.2f}{100*P:>8.2f}{100*F1:>8.2f}"
        for lo, hi, _n in BANDS:
            row += f"{100*prf((m >= lo) & (m < hi))[0]:>14.1f}"
        nr, np_, nf, nfp = prf(m < 0.90)
        row += f"{100*nfp:>11.2f}{100*nf:>10.2f}"
        print(row, flush=True)
        np.save(BASE / f"audit_nonbear_1003/pred_{a.split}_{name}.npy", pred)


if __name__ == "__main__":
    main()
