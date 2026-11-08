"""V18 step 1: find images labelled non_bear that actually show a bear.

Why: nearest-neighbour attribution of V17's missed bears (nn_attrib_v17_recall.py)
found bear-containing images inside non_bear among their closest training
neighbours -- a Ring-doorbell bear (NOT_BEAR_animated_cgi_text...), a brown bear on
a lawn (QWENVL_NO_in_urban...), video_in_urban_videos_bear_opening_door_...,
bear_MFDC0100_76, non_bear_pasteaug_neg_00524. Each one teaches the model that
a bear is not a bear.

Three independent signals over every non_bear image in train/val/test:
  v16, v17  p(bear) from the two delivered builds' FP32-equivalent weights. Both
            were trained on most of these labels, so they UNDER-score a memorised
            mislabel -- a high score here survives despite training against it.
  owl       OWLv2 open-vocabulary detector, "a bear" -- never trained on our
            labels, the only signal that caught the dumpster-cam frames in the
            29/09 trace (calibrated there: >= 0.15 flags 3.3% of LILA empties).
No judge removes anything: candidates go to the user as a review zip, and only
what the user confirms is quarantined, with a manifest (memory: no LLM or
detector may remove negatives unmeasured).

    python audit_nonbear.py --stage cls  --gpu 2
    python audit_nonbear.py --stage owl  --gpu 4 --shard 0/4     # x4 GPUs
    python audit_nonbear.py --stage report
"""
import argparse
import json
import os
from pathlib import Path

BASE = Path("/data/nas07_new/PersonalData/phuocthien")
DS = BASE / "bear_dataset_v_final_final"
OUT = BASE / "audit_nonbear_1003"
MODELS = {
    "v16": BASE / "deliver_qat_pow2_20260917/checkpoint_fp32_equivalent.pth",
    "v17": BASE / "deliver_qat_pow2_20261002/checkpoint_fp32_equivalent.pth",
}


def all_nonbear():
    files = []
    for split in ("train", "val", "test"):
        d = DS / split / "non_bear"
        files += sorted(str(d / n) for n in os.listdir(d))
    return files


# ------------------------------------------------------------- classifiers
def stage_cls(gpu, workers):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from score_darkfur import Letterbox, load

    files = all_nonbear()
    out_path = OUT / "cls_scores.json"
    print(f"non_bear images: {len(files)}", flush=True)
    dev = f"cuda:{gpu}"
    nets = {k: load(str(v), dev) for k, v in MODELS.items()}
    tf = transforms.Compose([Letterbox(), transforms.ToTensor()])

    class DS_(Dataset):
        def __len__(self):
            return len(files)

        def __getitem__(self, i):
            try:
                return tf(Image.open(files[i]).convert("RGB")), i
            except Exception:
                return torch.zeros(3, 360, 640), -1 - i

    dl = DataLoader(DS_(), batch_size=64, num_workers=workers)
    mean = torch.tensor([104, 116, 124], device=dev).view(1, 3, 1, 1) / 255
    res, bad = {}, []
    with torch.no_grad():
        for b, (x, idx) in enumerate(dl):
            x = x.to(dev)[:, [2, 1, 0]] - mean
            p = {k: n(x).softmax(1)[:, 1].cpu().tolist() for k, n in nets.items()}
            for j, i in enumerate(idx.tolist()):
                if i < 0:
                    bad.append(files[-1 - i])
                    continue
                res[files[i]] = {k: round(p[k][j], 4) for k in nets}
            if b % 200 == 0:
                print(f"  {len(res)}/{len(files)}", flush=True)
    OUT.mkdir(exist_ok=True)
    json.dump(res, open(out_path, "w"))
    print(f"wrote {out_path} ({len(res)} scored, {len(bad)} unreadable)", flush=True)
    if bad:
        json.dump(bad, open(OUT / "unreadable.json", "w"), indent=1)


# ------------------------------------------------------------------- OWLv2
def stage_owl(gpu, shard, workers, batch):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    k, n = (int(x) for x in shard.split("/"))
    files = all_nonbear()[k::n]
    out_path = OUT / f"owl_scores_{k}of{n}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["f"] for l in open(out_path)}
    todo = [f for f in files if f not in done]
    print(f"shard {shard}: {len(files)} images, {len(todo)} to do", flush=True)

    name = "google/owlv2-base-patch16-ensemble"
    proc = Owlv2Processor.from_pretrained(name)
    dev = f"cuda:{gpu}"
    model = Owlv2ForObjectDetection.from_pretrained(name, torch_dtype=torch.float16).to(dev).eval()
    # Same prompts as trace_video_bears.py, so its 0.15 calibration carries over.
    txt = proc(text=[["a photo of a bear", "a bear"]], return_tensors="pt")

    class DS_(Dataset):
        def __len__(self):
            return len(todo)

        def __getitem__(self, i):
            try:
                im = Image.open(todo[i]).convert("RGB")
            except Exception:
                return torch.zeros(3, 960, 960), -1
            return proc(images=im, return_tensors="pt")["pixel_values"][0], i

    dl = DataLoader(DS_(), batch_size=batch, num_workers=workers)
    OUT.mkdir(exist_ok=True)
    fh = open(out_path, "a")
    n_done = 0
    with torch.no_grad():
        for x, idx in dl:
            B = x.shape[0]
            o = model(input_ids=txt["input_ids"].repeat(B, 1).to(dev),
                      attention_mask=txt["attention_mask"].repeat(B, 1).to(dev),
                      pixel_values=x.to(dev, torch.float16))
            s = o.logits.float().max(-1).values.sigmoid().max(-1).values.cpu().tolist()
            for i, v in zip(idx.tolist(), s):
                if i >= 0:
                    fh.write(json.dumps({"f": todo[i], "owl": round(v, 4)}) + "\n")
            fh.flush()
            n_done += B
            if (n_done // B) % 250 == 0:
                print(f"  {n_done}/{len(todo)}", flush=True)
    fh.close()
    print(f"shard {shard} done", flush=True)


# ------------------------------------------------- OWLv2, competing labels
# "a bear" alone fires on almost any animal (97% of gorillas, 95% of NTLNP boar,
# 70% of raccoon dogs on the first 132k), because OWL scores each query on its
# own. With the confusable species as competing queries, a box only counts as a
# bear when "bear" beats every other label ON THAT BOX.
QUERIES = ["a bear", "a wild boar", "a pig", "a dog", "a badger", "a raccoon",
           "a raccoon dog", "a monkey", "a gorilla", "a deer", "a cat", "a fox",
           "an otter", "a marten", "a squirrel", "a rabbit", "a goat", "a sheep",
           "a cow", "a horse", "an elephant", "a person", "a bird", "a civet"]


def stage_owlq(gpu, shard, workers, batch, min_owl):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    owl = {}
    for f in OUT.glob("owl_scores_*of4.jsonl"):
        for l in open(f):
            r = json.loads(l)
            owl[r["f"]] = r["owl"]
    k, n = (int(x) for x in shard.split("/"))
    files = sorted(f for f, v in owl.items() if v >= min_owl)[k::n]
    out_path = OUT / f"owlq_{k}of{n}.jsonl"
    done = {json.loads(l)["f"] for l in open(out_path)} if out_path.exists() else set()
    todo = [f for f in files if f not in done]
    print(f"owlq shard {shard}: {len(files)} with owl >= {min_owl}, {len(todo)} to do", flush=True)

    name = "google/owlv2-base-patch16-ensemble"
    proc = Owlv2Processor.from_pretrained(name)
    dev = f"cuda:{gpu}"
    model = Owlv2ForObjectDetection.from_pretrained(name, torch_dtype=torch.float16).to(dev).eval()
    txt = proc(text=[QUERIES], return_tensors="pt")

    class DS_(Dataset):
        def __len__(self):
            return len(todo)

        def __getitem__(self, i):
            try:
                im = Image.open(todo[i]).convert("RGB")
            except Exception:
                return torch.zeros(3, 960, 960), -1
            return proc(images=im, return_tensors="pt")["pixel_values"][0], i

    dl = DataLoader(DS_(), batch_size=batch, num_workers=workers)
    fh = open(out_path, "a")
    n_done = 0
    with torch.no_grad():
        for x, idx in dl:
            B = x.shape[0]
            o = model(input_ids=txt["input_ids"].repeat(B, 1).to(dev),
                      attention_mask=txt["attention_mask"].repeat(B, 1).to(dev),
                      pixel_values=x.to(dev, torch.float16))
            lg = o.logits.float()                          # [B, boxes, Q]
            for b, i in enumerate(idx.tolist()):
                if i < 0:
                    continue
                L = lg[b]
                bear = L[:, 0]
                other, oi = L[:, 1:].max(-1)
                j = int(bear.argmax())                     # the most bear-like box
                top_box = int(L.max(-1).values.argmax())   # the most confident box, any label
                fh.write(json.dumps({
                    "f": todo[i],
                    "bear": round(float(bear[j].sigmoid()), 4),
                    "margin": round(float(bear[j] - other[j]), 3),
                    "rival": QUERIES[1 + int(oi[j])],
                    "wins": round(float(torch.where(bear > other, bear, torch.full_like(bear, -99)).max().sigmoid()), 4),
                    "top": QUERIES[int(L[top_box].argmax())],
                }) + "\n")
            fh.flush()
            n_done += B
            if (n_done // B) % 250 == 0:
                print(f"  {n_done}/{len(todo)}", flush=True)
    fh.close()
    print(f"owlq shard {shard} done", flush=True)


# ------------------------------------------------ CLIP embeddings + probe
# OWLv2 could not separate a bear from a gorilla or a boar, even with those as
# competing queries, and it called the grizzly in QWENVL_NO_..._00106 "a dog".
# Confident learning instead: a linear probe on frozen CLIP features, trained
# K-fold on OUR labels, scores every image with a probe that never saw it. It
# cannot memorise one image, so a bear filed under non_bear keeps a high
# out-of-fold p(bear), while a gorilla is pulled down by all the other gorillas.
CLIP_NAME = "openai/clip-vit-large-patch14"


def all_labelled():
    out = []
    for split in ("train", "val", "test"):
        for cls in ("bear", "non_bear"):
            d = DS / split / cls
            out += [(str(d / n), cls) for n in sorted(os.listdir(d))]
    return out


def stage_embed(gpu, shard, workers, batch):
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from transformers import CLIPModel

    k, n = (int(x) for x in shard.split("/"))
    items = all_labelled()[k::n]
    print(f"embed shard {shard}: {len(items)} images", flush=True)
    model = CLIPModel.from_pretrained(CLIP_NAME, torch_dtype=torch.float16).to(f"cuda:{gpu}").eval()
    # Pad to square instead of CLIP's centre crop: a 16:9 frame loses its
    # sides to a crop, and a bear at the edge (the Ring doorbell frame) with it.
    norm = transforms.Normalize((0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758))

    def square(im):
        w, h = im.size
        s_ = max(w, h)
        c = Image.new("RGB", (s_, s_), (0, 0, 0))
        c.paste(im, ((s_ - w) // 2, (s_ - h) // 2))
        return c.resize((224, 224), Image.BICUBIC)

    tf = transforms.Compose([transforms.Lambda(square), transforms.ToTensor(), norm])

    class DS_(Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, i):
            try:
                return tf(Image.open(items[i][0]).convert("RGB")), i
            except Exception:
                return torch.zeros(3, 224, 224), -1

    dl = DataLoader(DS_(), batch_size=batch, num_workers=workers)
    E = np.zeros((len(items), 768), np.float16)
    ok = np.zeros(len(items), bool)
    with torch.no_grad():
        for b, (x, idx) in enumerate(dl):
            e = model.get_image_features(pixel_values=x.to(f"cuda:{gpu}", torch.float16))
            e = torch.nn.functional.normalize(e.float(), dim=1).cpu().numpy()
            for j, i in enumerate(idx.tolist()):
                if i >= 0:
                    E[i] = e[j]
                    ok[i] = True
            if b % 200 == 0:
                print(f"  {b * batch}/{len(items)}", flush=True)
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / f"clip_{k}of{n}.npz", E=E, ok=ok,
             files=np.array([f for f, _ in items]), labels=np.array([c for _, c in items]))
    print(f"embed shard {shard} done ({ok.sum()} ok)", flush=True)


def stage_probe(folds=5):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    parts = [np.load(f) for f in sorted(OUT.glob("clip_*of*.npz"))]
    E = np.concatenate([p["E"] for p in parts]).astype(np.float32)
    ok = np.concatenate([p["ok"] for p in parts])
    files = np.concatenate([p["files"] for p in parts])
    y = (np.concatenate([p["labels"] for p in parts]) == "bear").astype(int)
    E, files, y = E[ok], files[ok], y[ok]
    print(f"probe: {len(y)} images, {y.sum()} bear / {(1 - y).sum()} non_bear", flush=True)
    oof = np.zeros(len(y))
    for i, (tr, te) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=0).split(E, y)):
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(E[tr], y[tr])
        oof[te] = clf.predict_proba(E[te])[:, 1]
        acc = ((oof[te] >= 0.5) == y[te]).mean()
        print(f"  fold {i}: held-out accuracy {100 * acc:.2f}%", flush=True)
    np.savez(OUT / "probe_oof.npz", files=files, y=y, oof=oof)
    nb = y == 0
    for t in (0.5, 0.8, 0.9, 0.95):
        print(f"  non_bear with oof p(bear) >= {t}: {(oof[nb] >= t).sum():>6}   "
              f"bear with oof p(bear) <= {1 - t:.2f}: {(oof[~nb] <= 1 - t).sum():>6}")


# ------------------------------------------------------------------ review
def stage_review(t_all=0.9, t_suspect=0.5):
    """Export what the user reviews by eye. Ranked by the probe's out-of-fold
    p(bear): every non_bear >= t_all, plus the SUSPECT_PREFIXES families down to
    t_suspect (their names say bear and they came through unreliable judges).
    Measured on a random 40 at >= 0.9: ~30% showed a real bear."""
    import csv
    import shutil
    import zipfile
    import numpy as np
    d = np.load(OUT / "probe_oof.npz")
    files, y, oof = d["files"], d["y"], d["oof"]
    sel = [i for i in np.where(y == 0)[0]
           if oof[i] >= t_all or (oof[i] >= t_suspect and Path(files[i]).name.startswith(SUSPECT_PREFIXES))]
    sel.sort(key=lambda i: -oof[i])
    rv = OUT / "review_probe"
    shutil.rmtree(rv, ignore_errors=True)
    rv.mkdir(parents=True)
    rows = []
    for r, i in enumerate(sel):
        src = Path(files[i])
        split = src.parent.parent.name
        name = f"{r:04d}_p{oof[i]:.3f}__{split}__{src.name}"
        shutil.copy2(src, rv / name)
        rows.append({"rank": r, "p_bear_oof": round(float(oof[i]), 4), "split": split,
                     "family": family(str(src)), "path": str(src), "review_name": name})
    with open(OUT / "review_probe.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    zp = BASE / "audit_nonbear_review_1003.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for r in rows:
            z.write(rv / r["review_name"], f"review/{r['review_name']}")
        z.write(OUT / "review_probe.csv", "review_probe.csv")
    import collections
    print(f"review set: {len(rows)} images -> {zp}")
    print("  by split:", dict(collections.Counter(r["split"] for r in rows)))
    for k, v in collections.Counter(r["family"] for r in rows).most_common(15):
        print(f"    {k:<24}{v:>5}")


def stage_review_bear(t=0.1):
    """The other direction: bear-class images the probe calls non-bear (out-of-fold
    p(bear) <= t), lowest first. Under the 03/10 policy taxidermy, drawings and toys
    are non_bear, and those are expected here alongside frames with no bear."""
    import csv
    import shutil
    import zipfile
    import numpy as np
    d = np.load(OUT / "probe_oof.npz")
    files, y, oof = d["files"], d["y"], d["oof"]
    sel = sorted((i for i in np.where(y == 1)[0] if oof[i] <= t), key=lambda i: oof[i])
    rv = OUT / "review_bear"
    shutil.rmtree(rv, ignore_errors=True)
    rv.mkdir(parents=True)
    rows = []
    for r, i in enumerate(sel):
        src = Path(files[i])
        split = src.parent.parent.name
        name = f"{r:03d}_p{oof[i]:.3f}__{split}__{src.name}"
        shutil.copy2(src, rv / name)
        rows.append({"rank": r, "p_bear_oof": round(float(oof[i]), 4), "split": split,
                     "family": family(str(src)), "path": str(src), "review_name": name})
    with open(OUT / "review_bear.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    zp = BASE / "audit_bear_review_1003.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for r in rows:
            z.write(rv / r["review_name"], f"review_bear/{r['review_name']}")
        z.write(OUT / "review_bear.csv", "review_bear.csv")
    import collections
    print(f"bear review set: {len(rows)} images -> {zp}")
    print("  by split:", dict(collections.Counter(r["split"] for r in rows)))
    for k, v in collections.Counter(r["family"] for r in rows).most_common(15):
        print(f"    {k:<24}{v:>5}")


def stage_review_bands():
    """Second audit round (06/10): the bands the first one left.
      non_bear with out-of-fold p(bear) in [0.5, 0.9)  -> may still hold bears
      bear     with out-of-fold p(bear) in (0.1, 0.5]  -> may not be bears
      bear images YOLO finds no bear in among V18's lowest-scored 4,000
               (bear_class_no_yolo_bear_1006.json): tracks, scat, skulls, pelts,
               which CLIP rates as "bear" and the first round missed.
    Only images still in the dataset (listed per directory -- 253k os.path.exists
    calls on the NAS took over 30 minutes)."""
    import csv
    import shutil
    import zipfile
    import numpy as np
    present = set()
    for split in ("train", "val", "test"):
        for cls in ("bear", "non_bear"):
            d = DS / split / cls
            present.update(str(d / n) for n in os.listdir(d))
    d = np.load(OUT / "probe_oof.npz")
    files, y, oof = d["files"], d["y"], d["oof"]
    oofd = dict(zip(files, oof))
    nb = [i for i in range(len(files)) if y[i] == 0 and files[i] in present and 0.5 <= oof[i] < 0.9]
    nb.sort(key=lambda i: -oof[i])
    bb = {files[i] for i in range(len(files)) if y[i] == 1 and files[i] in present and 0.1 < oof[i] <= 0.5}
    noyolo = [str(BASE / p) for p, _ in json.load(open(BASE / "bear_class_no_yolo_bear_1006.json"))]
    bb |= {p for p in noyolo if p in present}
    bb = sorted(bb, key=lambda p: oofd.get(p, 0.0))
    yset = set(noyolo)
    for tag, items, getp in (("nonbear", [files[i] for i in nb], None), ("bear", bb, None)):
        rv = OUT / f"review2_{tag}"
        shutil.rmtree(rv, ignore_errors=True)
        rv.mkdir(parents=True)
        rows = []
        for r, f in enumerate(items):
            src = Path(f)
            split = src.parent.parent.name
            flag = "_noyolo" if f in yset else ""
            name = f"{r:04d}_p{oofd.get(f, -1):.3f}{flag}__{split}__{src.name}"
            shutil.copy2(src, rv / name)
            rows.append({"rank": r, "p_bear_oof": round(float(oofd.get(f, -1)), 4),
                         "no_yolo_bear": bool(flag), "split": split, "path": f, "review_name": name})
        with open(OUT / f"review2_{tag}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        zp = BASE / f"audit_review2_{tag}_1006.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
            for r in rows:
                z.write(rv / r["review_name"], f"review2_{tag}/{r['review_name']}")
            z.write(OUT / f"review2_{tag}.csv", f"review2_{tag}.csv")
        print(f"{tag}: {len(rows)} images -> {zp} ({zp.stat().st_size / 2**20:.0f} MB)", flush=True)


# -------------------------------------------------------------- quarantine
QDIR = BASE / "quarantine_audit_1003"


def stage_quarantine(reverse=None, which="non_bear"):
    """Move the reviewed set out of the dataset (user approved ALL 1,286 on
    03/10: the head of the list is nearly all bears, the tail still has some).
    Not relabelled to bear -- only removed. Symlinks (the IR boar/deer/monkey
    hard negatives linked from non_bear) cannot simply be renamed: their target
    is relative ('../boar/x.jpg'), so the link is re-created in quarantine with
    an absolute target and the original text is kept for --reverse. The 5-class
    files they point to are not touched."""
    import csv
    from datetime import datetime
    if reverse:
        man = json.load(open(reverse))
        n = 0
        for m in man["moved"]:
            src, dst = Path(m["from"]), Path(m["to"])
            if m.get("link"):
                if dst.is_symlink():
                    dst.unlink()
                os.symlink(m["link"], src)
            else:
                os.rename(dst, src)
            n += 1
        print(f"restored {n}/{len(man['moved'])}")
        return
    csv_name = {"non_bear": "review_probe.csv", "bear": "review_bear.csv"}[which]
    rows = list(csv.DictReader(open(OUT / csv_name)))
    moved = []
    for r in rows:
        src = Path(r["path"])
        split = src.parent.parent.name
        dst = QDIR / split / which / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        m = {"from": str(src), "to": str(dst), "p_bear_oof": float(r["p_bear_oof"]),
             "rank": int(r["rank"])}
        if src.is_symlink():
            link = os.readlink(src)
            m["link"] = link
            os.symlink(os.path.realpath(src), dst)
            src.unlink()
        else:
            os.rename(src, dst)
        moved.append(m)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mp = BASE / f"audit_quarantine_{which}_manifest_{ts}.json"
    reason = {
        "non_bear": "V18 label audit 03/10: non_bear images with CLIP 5-fold probe "
                    "out-of-fold p(bear) >= 0.9, plus bear-named suspect families "
                    ">= 0.5; all 1,286 quarantined on user approval (~30% real "
                    "bears by sample, densest at the top of the ranking)",
        "bear": "V18 label audit 03/10: bear images with CLIP 5-fold probe out-of-fold "
                "p(bear) <= 0.1; user reviewed the zip (~99% not a bear: tracks, scat, "
                "other animals, title cards, failed paste-aug positives) and chose to "
                "quarantine all 443 rather than relabel",
    }[which]
    json.dump({"reason": reason, "moved": moved}, open(mp, "w"), indent=1)
    import collections
    print(f"quarantined {len(moved)} -> {QDIR}")
    print("  by split:", dict(collections.Counter(Path(m['from']).parent.parent.name for m in moved)))
    print(f"  symlinks re-created: {sum(1 for m in moved if m.get('link'))}")
    print(f"manifest: {mp}")
    print(f"undo: python audit_nonbear.py --stage quarantine --reverse {mp}")


def stage_quarantine_round2():
    """Round-2 decisions by the user (07/10), applied to review2_*.csv:
      bear:     ranks 0000-1393
      non_bear: ranks 0000-1000, every row whose name contains train_bear1
                (the user: "all of them have bears"), and
                1559_p0.644__train__non_bear_full_capture_seg0227_sub00
    Same mechanics as stage_quarantine (symlinks re-created, manifest kept)."""
    import csv
    from datetime import datetime
    pick = {
        "bear": [r for r in csv.DictReader(open(OUT / "review2_bear.csv")) if int(r["rank"]) <= 1393],
        "non_bear": [r for r in csv.DictReader(open(OUT / "review2_nonbear.csv"))
                     if int(r["rank"]) <= 1000 or "train_bear1" in r["review_name"]
                     or r["review_name"].startswith("1559_p0.644__train__non_bear_full_capture_seg0227_sub00")],
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for which, rows in pick.items():
        moved = []
        for r in rows:
            src = Path(r["path"])
            if not os.path.lexists(src):
                continue
            dst = QDIR / src.parent.parent.name / which / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            m = {"from": str(src), "to": str(dst), "p_bear_oof": float(r["p_bear_oof"]),
                 "rank": int(r["rank"])}
            if src.is_symlink():
                m["link"] = os.readlink(src)
                os.symlink(os.path.realpath(src), dst)
                src.unlink()
            else:
                os.rename(src, dst)
            moved.append(m)
        mp = BASE / f"audit_quarantine_round2_{which}_manifest_{ts}.json"
        json.dump({"reason": f"V18 label audit round 2 (07/10), {which}: user-reviewed "
                             "review2 zip; see stage_quarantine_round2 for the selection",
                   "moved": moved}, open(mp, "w"), indent=1)
        import collections
        print(f"{which}: quarantined {len(moved)} "
              f"{dict(collections.Counter(Path(m['from']).parent.parent.name for m in moved))}"
              f", symlinks {sum(1 for m in moved if m.get('link'))} -> {mp.name}")


# ------------------------------------------------------------------ report
def family(path):
    """Coarse source family from the filename, for the per-source table."""
    import re
    n = Path(path).name
    for pre in ("video_", "NOT_BEAR_", "QWENVL_", "animal_frames", "non_bear_pasteaug",
                "bear_MFDC", "trail_camera", "acc1109", "acc1809", "acc1002",
                "screencap", "empty_", "lens_", "train_non_bear", "val_non_bear",
                "test_non_bear", "swg_", "caltech_", "ena24_", "coco_", "openimages_"):
        if n.startswith(pre):
            return pre.rstrip("_")
    m = re.match(r"([a-z]+_[a-z]+)", n)
    return m.group(1) if m else "other"


def load_scores():
    cls = json.load(open(OUT / "cls_scores.json"))
    owl = {}
    for f in OUT.glob("owl_scores_*of4.jsonl"):
        for l in open(f):
            r = json.loads(l)
            owl[r["f"]] = r["owl"]
    return cls, owl


def tiers(c, o):
    """Why an image is a candidate. OWL is the primary signal (independent of our
    labels); the classifiers can only add, never veto, because they were trained
    to say non_bear on exactly these images."""
    t = []
    if o is not None and o >= OWL_STRONG:
        t.append("owl")
    if o is not None and o >= OWL_WEAK and max(c["v16"], c["v17"]) >= 0.5:
        t.append("owl+cls")
    if min(c["v16"], c["v17"]) >= CLS_BOTH:
        t.append("cls_both")
    return t


# Filename families inside non_bear whose NAME says bear: images an earlier
# LLM pass (Qwen-VL, "NOT_BEAR") or a frame harvester moved out of the bear
# class. Those judges were measured unreliable, and two of the five mislabels
# the NN attribution found came from here, so they are reviewed in full,
# not by score.
SUSPECT_PREFIXES = ("NOT_BEAR", "QWENVL_NO", "bear_frames", "bear_MFDC", "low_light",
                    "bear_pasteaug", "bear_full")

OWL_STRONG = 0.30   # trace 29/09: about half of the top-48 >= 0.15 showed a bear
OWL_WEAK = 0.15     # 29/09 calibration: flags 3.3% of LILA empties on its own
CLS_BOTH = 0.90


def stage_report(export):
    import collections
    import csv
    import shutil
    cls, owl = load_scores()
    files = [f for f in cls if f in owl]
    print(f"scored by both: {len(files)}  (cls {len(cls)}, owl {len(owl)})")
    for t in (0.15, 0.2, 0.3, 0.5):
        print(f"  owl >= {t:.2f}: {sum(owl[f] >= t for f in files):>6}")
    for t in (0.5, 0.9):
        print(f"  v16&v17 >= {t}: {sum(min(cls[f]['v16'], cls[f]['v17']) >= t for f in files):>6}")

    suspect = [f for f in files if Path(f).name.startswith(SUSPECT_PREFIXES)]
    cand = []
    for f in files:
        if Path(f).name.startswith(SUSPECT_PREFIXES):
            continue
        t = tiers(cls[f], owl[f])
        if t:
            cand.append({"f": f, "owl": owl[f], "v16": cls[f]["v16"], "v17": cls[f]["v17"],
                         "why": "+".join(t), "family": family(f),
                         "split": Path(f).parent.parent.name})
    # strongest evidence first: OWL, then the classifiers' agreement
    cand.sort(key=lambda r: (-r["owl"], -min(r["v16"], r["v17"])))
    print(f"\ncandidates: {len(cand)}")
    print("  by reason:", dict(collections.Counter(r["why"] for r in cand)))
    print("  by split :", dict(collections.Counter(r["split"] for r in cand)))
    fam_all = collections.Counter(family(f) for f in files)
    fam_c = collections.Counter(r["family"] for r in cand)
    print(f"\n  {'family':<24}{'flagged':>8}{'of':>8}{'rate':>8}")
    for k, v in fam_c.most_common(30):
        print(f"  {k:<24}{v:>8}{fam_all[k]:>8}{100*v/fam_all[k]:>7.1f}%")

    with open(OUT / "candidates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rank"] + list(cand[0].keys()))
        w.writeheader()
        for i, r in enumerate(cand):
            w.writerow({"rank": i, **r})
    print(f"\nwrote {OUT/'candidates.csv'}")

    if export:
        d = OUT / "review"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        for i, r in enumerate(cand[:export]):
            src = Path(r["f"])
            dst = d / (f"{i:04d}_owl{r['owl']:.2f}_v16-{r['v16']:.2f}_v17-{r['v17']:.2f}"
                       f"__{r['split']}__{src.name}")
            shutil.copy2(src, dst)
        print(f"exported top {min(export, len(cand))} -> {d}")
        fam = OUT / "review_families"
        shutil.rmtree(fam, ignore_errors=True)
        for pre in SUSPECT_PREFIXES:
            fs = sorted((f for f in suspect if Path(f).name.startswith(pre)),
                        key=lambda f: -owl[f])
            if not fs:
                continue
            (fam / pre).mkdir(parents=True)
            for i, f in enumerate(fs):
                shutil.copy2(f, fam / pre / (f"{i:03d}_owl{owl[f]:.2f}_v16-{cls[f]['v16']:.2f}"
                                             f"_v17-{cls[f]['v17']:.2f}__{Path(f).parent.parent.name}"
                                             f"__{Path(f).name}"))
        print(f"suspect families, all {len(suspect)} images -> {fam}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["cls", "owl", "owlq", "embed", "probe", "review", "review_bear", "review_bands", "quarantine", "quarantine_round2", "report"])
    ap.add_argument("--reverse")
    ap.add_argument("--which", choices=["non_bear", "bear"], default="non_bear")
    ap.add_argument("--min-owl", type=float, default=0.15)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--export", type=int, default=0, help="report: copy top-N for review")
    a = ap.parse_args()
    if a.stage == "cls":
        stage_cls(a.gpu, a.workers)
    elif a.stage == "owl":
        stage_owl(a.gpu, a.shard, a.workers, a.batch)
    elif a.stage == "embed":
        stage_embed(a.gpu, a.shard, a.workers, a.batch)
    elif a.stage == "probe":
        stage_probe()
    elif a.stage == "review":
        stage_review()
    elif a.stage == "quarantine_round2":
        stage_quarantine_round2()
    elif a.stage == "review_bands":
        stage_review_bands()
    elif a.stage == "review_bear":
        stage_review_bear()
    elif a.stage == "quarantine":
        stage_quarantine(a.reverse, a.which)
    elif a.stage == "owlq":
        stage_owlq(a.gpu, a.shard, a.workers, a.batch, a.min_owl)
    else:
        stage_report(a.export)
