"""Camera-capture package for V19 (06/10): dark-fur negatives AND bears.

The 26/09 package's train_neg half was never filmed. This replaces it:
  * hard negatives are re-ranked with the current model (V18 best) and drawn
    only from dark-fur images still in train (the 03/10 audit quarantined some);
  * it carries BEARS too. Filming negatives alone would make "looks filmed off
    a screen" a non-bear cue -- the reason ScreenCaptureAugmentation has to be
    symmetric -- and cost recall in the screen-filmed acceptance test;
  * images already in the camera domain (acceptance frames, screencaps, MFDC
    clips) are excluded: filming them again adds nothing.

Groups (defaults ~4,000 images, the user found 4,500 too many on 28/09):
  non_bear  darkfur_hard    highest V18 p(bear) among dark-fur train images
            darkfur_random  random dark-fur train images, for variety
            lila            LILA camera-trap confusables, species-balanced, IR first
  bear      bear_hard       lowest V18 p(bear) among train bears
            bear_ir         random night/IR train bears (low saturation)
Everything is letterboxed to 1920x1080 and shown in one shuffled order (a
session cut short stays balanced); labels.csv carries the truth for
acceptance_manual_test.py. Frames go to train ONLY after matching
(match_acceptance_frames.py) and merging (merge_acceptance_frames_to_train.py).

    python export_capture_package.py --gpu 2
"""
import argparse
import collections
import csv
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from score_darkfur import Letterbox
from eval_novel_val import build

BASE = Path("/data/nas07_new/PersonalData/phuocthien")
DS = BASE / "bear_dataset_v_final_final"
OUT = BASE / "capture_package_1006"
MODEL = BASE / "BearDetection_mobilenetv2_relu_novaic_640x360_0310/models/best_model.pth"
DARKFUR_MANIFESTS = ("merge_manifest_20260924_174034.json", "merge_manifest_20260928_211611.json")
CAMERA_PREFIXES = ("acc1109", "acc1809", "acc1002", "screencap", "bear_MFDC", "non_bear_MFDC",
                   "MFDC", "capture", "bear_full_capture", "non_bear_full_capture")
LILA_PREFIXES = ("swg_", "caltech_", "ena24_")


def letterbox(src, dst, W=1920, H=1080):
    im = Image.open(src).convert("RGB")
    k = min(W / im.width, H / im.height)
    im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
    c = Image.new("RGB", (W, H), (0, 0, 0))
    c.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
    c.save(dst, quality=95)


def saturation(p):
    im = Image.open(p).convert("HSV").resize((64, 36))
    return float(np.asarray(im)[..., 1].mean() / 255)


def score(paths, gpu):
    dev = f"cuda:{gpu}"
    net = build(str(MODEL), dev)
    tf = transforms.Compose([Letterbox(), transforms.ToTensor()])
    mean = torch.tensor([104, 116, 124], device=dev).view(1, 3, 1, 1) / 255

    class D(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            return tf(Image.open(paths[i]).convert("RGB")), i
    out = np.zeros(len(paths))
    with torch.no_grad():
        for x, i in DataLoader(D(), batch_size=32, num_workers=8):
            out[i.numpy()] = net(x.to(dev)[:, [2, 1, 0]] - mean).softmax(1)[:, 1].cpu().numpy()
    return out


def yolo_bear(paths, gpu, batch=32):
    """Max COCO `bear` (class 21) confidence per image. On bear photos the class
    is ~98.7% precise (memory yolo-bear-class-is-a-good-screen). Run with
    CUDA_VISIBLE_DEVICES set: ultralytics otherwise touches GPU0."""
    from ultralytics import YOLO
    model = YOLO(str(BASE / "yolov8m.pt"))
    out = np.zeros(len(paths))
    for i in range(0, len(paths), batch):
        res = model.predict([str(p) for p in paths[i:i + batch]], conf=0.05, device=gpu,
                            classes=[21], verbose=False, imgsz=640)
        for j, r in enumerate(res):
            if len(r.boxes):
                out[i + j] = float(r.boxes.conf.max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-darkfur-hard", type=int, default=1200)
    ap.add_argument("--n-darkfur-random", type=int, default=300)
    ap.add_argument("--n-lila", type=int, default=900)
    ap.add_argument("--n-bear-hard", type=int, default=800)
    ap.add_argument("--n-bear-ir", type=int, default=800)
    ap.add_argument("--ir-sat", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    # ---- dark-fur negatives still in train
    darkfur = []
    for m in DARKFUR_MANIFESTS:
        for p in json.load(open(BASE / m))["added_files"]:
            p = Path(p) if Path(p).is_absolute() else BASE / p
            if "/train/non_bear/" in str(p) and p.exists() and not p.name.startswith(LILA_PREFIXES):
                darkfur.append(p)
    darkfur = sorted(set(darkfur))
    # ---- bears in train, not already camera-domain
    bears = sorted(p for p in (DS / "train/bear").iterdir()
                   if not p.name.startswith(CAMERA_PREFIXES) and not p.is_symlink())
    print(f"dark-fur train negatives: {len(darkfur)}   train bears (non-camera): {len(bears)}", flush=True)

    s_df = score(darkfur, a.gpu)
    order = np.argsort(-s_df)
    # The hardest negatives must not be bears the audit missed: skip any that
    # YOLO calls a bear with high confidence (only kept out of the package,
    # never removed from the dataset -- YOLO also fires on dark mammals).
    cand = [darkfur[i] for i in order[:a.n_darkfur_hard + 300]]
    yb = yolo_bear(cand, a.gpu)
    hard = [p for p, c in zip(cand, yb) if c < 0.6][:a.n_darkfur_hard]
    print(f"  dark-fur hard: {int((yb >= 0.6).sum())} of the top {len(cand)} skipped as YOLO bear >= 0.6", flush=True)
    hs = set(hard)
    rest = [darkfur[i] for i in order if darkfur[i] not in hs]
    rng.shuffle(rest)
    print(f"  dark-fur hard: p(bear) {s_df[order[0]]:.3f} .. {s_df[order[a.n_darkfur_hard - 1]]:.3f}", flush=True)

    s_b = score(bears, a.gpu)
    border = np.argsort(s_b)
    # The lowest-scored "bears" are largely bear SIGN -- tracks, scat, skulls,
    # pelts (checked by eye 06/10) -- which must not be filmed as bears. Keep
    # only those where YOLO finds a bear at >= 0.5 (at 0.25 a moose and a porcupine
    # close-up still got through, 06/10).
    cand = [bears[i] for i in border[:5 * a.n_bear_hard]]
    yb = yolo_bear(cand, a.gpu)
    bear_hard = [p for p, c in zip(cand, yb) if c >= 0.5][:a.n_bear_hard]
    nosign = [(str(p.relative_to(BASE)), float(s_b[bears.index(p)])) for p, c in zip(cand, yb) if c < 0.05]
    json.dump(nosign, open(OUT.parent / "bear_class_no_yolo_bear_1006.json", "w"), indent=0)
    print(f"  bear hard: {len(bear_hard)} kept with YOLO bear >= 0.5 out of the {len(cand)} lowest-scored;"
          f" {len(nosign)} with no YOLO bear at all listed for the label audit", flush=True)
    chosen = set(bear_hard)
    pool = [b for b in bears if b not in chosen]
    rng.shuffle(pool)
    bear_ir = []
    for b in pool:
        if len(bear_ir) >= a.n_bear_ir:
            break
        if saturation(b) < a.ir_sat:
            bear_ir.append(b)

    # ---- LILA confusables (same pool and rule as the 26/09 package)
    lila = BASE / "lila_confusables_2609"
    man = [json.loads(l) for l in open(lila / "manifest.jsonl")]
    by_sp = collections.defaultdict(lambda: {"ir": [], "day": []})
    for m in man:
        by_sp[(m["dataset"], m["species"])]["ir" if m["sat"] < a.ir_sat else "day"].append(m)
    for v in by_sp.values():
        rng.shuffle(v["ir"]); rng.shuffle(v["day"])
    picked = []
    while len(picked) < a.n_lila and any(v["ir"] or v["day"] for v in by_sp.values()):
        for v in by_sp.values():
            for k in ("ir", "day"):
                if v[k] and len(picked) < a.n_lila:
                    picked.append(v[k].pop())

    rows = ([(p, "non_bear", "darkfur_hard") for p in hard]
            + [(p, "non_bear", "darkfur_random") for p in rest[:a.n_darkfur_random]]
            + [(lila / m["path"], "non_bear", "lila_ir" if m["sat"] < a.ir_sat else "lila_day") for m in picked]
            + [(p, "bear", "bear_hard") for p in bear_hard]
            + [(p, "bear", "bear_ir") for p in bear_ir])
    rng.shuffle(rows)

    d = OUT / "train"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "labels.csv", "w", newline="") as fl, open(OUT / "manifest.csv", "w", newline="") as fm:
        wl, wm = csv.writer(fl), csv.writer(fm)
        wl.writerow(["file", "truth", "group", "source"])
        wm.writerow(["name", "set", "truth", "group", "source"])
        for i, (p, truth, grp) in enumerate(rows, 1):
            name = f"train_{i:05d}.jpg"
            if not (d / name).exists():
                letterbox(p, d / name)
            src = str(p.relative_to(BASE))
            wl.writerow([name, truth, grp, src])
            wm.writerow([name, "train", truth, grp, src])
    g = collections.Counter((t, grp) for _, t, grp in rows)
    for (t, grp), n in sorted(g.items()):
        print(f"  {t:9s} {grp:16s} {n:6d}")
    print(f"  total {len(rows)}  (bear {sum(1 for r in rows if r[1] == 'bear')})")
    z = BASE / "capture_package_1006.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
        for f in sorted(d.glob("*.jpg")):
            zf.write(f, f"train/{f.name}")
        zf.write(d / "labels.csv", "train/labels.csv")
    print(f"  -> {z}  ({z.stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
