"""Re-split train/val/test by SCENE CLUSTER instead of perceptual hash (V19, 09/10).

The phash split put frames of one video / one scene on both sides: 43% of val
bears had a CLIP cosine >= 0.95 twin in train, and V17 scored 98-100% recall on
those but 81% on val bears with no twin >= 0.90 (eval_novel_val.py). Val F1 was
mostly measuring memory.

Here every image (CLIP ViT-L/14 embedding from the 03/10 audit) is linked to
its near neighbours at cosine >= --thr, connected components are scenes, and
whole scenes go to one split (70/15/15 per class). Rules:
  * camera-domain frames (acceptance acc*, dark-fur capture cap1009_*) stay in
    train -- val/test stay independent of the acceptance benchmark;
  * images without an embedding (added after 03/10) stay where they are;
  * components larger than --giant (share of a class) are forced to train, so
    one huge scene cannot dominate val;
  * --cap N keeps at most N images of any one scene in TRAIN; the rest move to
    capped_scene_1009/ (reversible), so long videos stop drowning the rest.
Symlinks (IR boar/deer/monkey hard negatives) are re-created with an absolute
target when they change split. Everything is recorded in a manifest.

    python resplit_by_scene.py --stage plan  --gpu 0      # dry run, prints the plan
    python resplit_by_scene.py --stage apply              # moves files, writes manifest
    python resplit_by_scene.py --stage reverse --manifest resplit_manifest_*.json
"""
import argparse
import collections
import glob
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np

BASE = Path("/data/nas07_new/PersonalData/phuocthien")
DS = BASE / "bear_dataset_v_final_final"
OUT = BASE / "resplit_1009"
CAPPED = BASE / "capped_scene_1009"
CAMERA = ("acc1109", "acc1809", "acc1002", "cap1009")
SPLITS = ("train", "val", "test")
TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}


def present():
    cur = {}
    for sp in SPLITS:
        for c in ("bear", "non_bear"):
            d = DS / sp / c
            for n in os.listdir(d):
                cur[str(d / n)] = (sp, c)
    return cur


def knn(gpu, k=32):
    """Top-k CLIP neighbours of every non-camera image still in the dataset,
    cached: union-find at several thresholds is then cheap."""
    cache = OUT / "knn.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return d["files"], d["v"], d["j"]
    import torch
    parts = [np.load(f) for f in sorted(glob.glob(str(BASE / "audit_nonbear_1003/clip_*of3.npz")))]
    E = np.concatenate([p["E"] for p in parts])
    files = np.concatenate([p["files"] for p in parts])
    cur = present()
    keep = np.array([f in cur and not Path(f).name.startswith(CAMERA) for f in files])
    E, files = E[keep], files[keep]
    dev = f"cuda:{gpu}"
    T = torch.tensor(E, dtype=torch.float16, device=dev)
    V, J = [], []
    for i in range(0, len(files), 2048):
        v, j = (T[i:i + 2048] @ T.T).topk(k, dim=1)
        V.append(v.float().cpu().numpy().astype(np.float16)); J.append(j.cpu().numpy().astype(np.int32))
    OUT.mkdir(exist_ok=True)
    np.savez(cache, files=files, v=np.concatenate(V), j=np.concatenate(J))
    return files, np.concatenate(V), np.concatenate(J)


def union(v, j, thr):
    n = len(v)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    rows, cols = np.nonzero(v >= thr)
    for a, c in zip(rows, cols):
        b = int(j[a, c])
        if b != a:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    return np.array([find(a) for a in range(n)])


def components(gpu, thr, k=32):
    import torch
    parts = [np.load(f) for f in sorted(glob.glob(str(BASE / "audit_nonbear_1003/clip_*of3.npz")))]
    E = np.concatenate([p["E"] for p in parts])
    files = np.concatenate([p["files"] for p in parts])
    cur = present()
    keep = np.array([f in cur and not Path(f).name.startswith(CAMERA) for f in files])
    E, files = E[keep], files[keep]
    n = len(files)
    print(f"embedded images still in the dataset (non-camera): {n}", flush=True)
    dev = f"cuda:{gpu}"
    T = torch.tensor(E, dtype=torch.float16, device=dev)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    n_edges = 0
    for i in range(0, n, 2048):
        s = T[i:i + 2048] @ T.T
        v, j = s.topk(k, dim=1)
        v, j = v.float().cpu().numpy(), j.cpu().numpy()
        for r in range(v.shape[0]):
            a = i + r
            for val, b in zip(v[r], j[r]):
                if val < thr:
                    break
                if b != a:
                    ra, rb = find(a), find(int(b))
                    if ra != rb:
                        parent[ra] = rb
                    n_edges += 1
        if (i // 2048) % 20 == 0:
            print(f"  {i}/{n}", flush=True)
    comp = np.array([find(a) for a in range(n)])
    print(f"  edges >= {thr}: {n_edges}", flush=True)
    return files, comp, cur


def sweep(a):
    files, v, j = knn(a.gpu)
    for thr in (0.92, 0.94, 0.95, 0.96, 0.97, 0.98):
        comp = union(v, j, thr)
        sizes = np.array(list(collections.Counter(comp).values()))
        big = np.sort(sizes)[::-1][:3]
        print(f"thr {thr:.2f}: components {len(sizes):>6}  largest {list(big)}  "
              f"images in comps >500: {int(sizes[sizes > 500].sum()):>6}  >50: {int(sizes[sizes > 50].sum()):>6}", flush=True)


def plan(a):
    files, v, j = knn(a.gpu)
    cur = present()
    comp = union(v, j, a.thr)
    capcomp = union(v, j, a.cap_thr)
    label = np.array([cur[f][1] for f in files])
    oldsp = np.array([cur[f][0] for f in files])
    members = collections.defaultdict(list)
    for i, c in enumerate(comp):
        members[c].append(i)
    sizes = np.array([len(v) for v in members.values()])
    print(f"components: {len(members)}  sizes: median {np.median(sizes):.0f}, "
          f"p99 {np.percentile(sizes, 99):.0f}, max {sizes.max()}; singletons {int((sizes == 1).sum())}")
    tot = collections.Counter(label)
    rng = random.Random(a.seed)
    comps = list(members.values())
    rng.shuffle(comps)
    comps.sort(key=len, reverse=True)          # big ones first, ties random
    have = {sp: collections.Counter() for sp in SPLITS}
    assign = {}
    forced = 0
    for mem in comps:
        cnt = collections.Counter(label[mem])
        major = cnt.most_common(1)[0][0]
        if len(mem) > a.giant * tot[major]:
            sp = "train"
            forced += 1
        else:
            # the split furthest below its target share for this component's majority class
            need = {s: TARGET[s] * tot[major] - have[s][major] for s in SPLITS}
            sp = max(need, key=need.get)
        for i in mem:
            assign[i] = sp
        have[sp].update(cnt)
    # cap images per scene in train
    capped = set()
    capmembers = collections.defaultdict(list)
    for i, c in enumerate(capcomp):
        capmembers[c].append(i)
    for mem in capmembers.values():
        tr = [i for i in mem if assign[i] == "train"]
        if len(tr) > a.cap:
            rng.shuffle(tr)
            capped.update(tr[a.cap:])
    print(f"forced to train (giant components > {a.giant:.0%} of a class): {forced}")
    for sp in SPLITS:
        c = have[sp]
        print(f"  {sp:<5} bear {c['bear']:>6} ({100 * c['bear'] / tot['bear']:.1f}%)  "
              f"non_bear {c['non_bear']:>7} ({100 * c['non_bear'] / tot['non_bear']:.1f}%)")
    moves = sum(1 for i in range(len(files)) if assign[i] != oldsp[i])
    print(f"images changing split: {moves} of {len(files)}")
    print(f"capped out of train (> {a.cap} near-duplicates at cos >= {a.cap_thr}): {len(capped)} "
          f"{dict(collections.Counter(label[list(capped)]))}")
    OUT.mkdir(exist_ok=True)
    json.dump({"thr": a.thr, "cap_thr": a.cap_thr, "cap": a.cap, "giant": a.giant, "seed": a.seed,
               "assign": {files[i]: assign[i] for i in range(len(files))},
               "capped": [files[i] for i in capped]},
              open(OUT / "plan.json", "w"))
    print(f"wrote {OUT / 'plan.json'}")


def move(src, dst, moved, link_note=True):
    dst.parent.mkdir(parents=True, exist_ok=True)
    m = {"from": str(src), "to": str(dst)}
    if src.is_symlink():
        m["link"] = os.readlink(src)
        os.symlink(os.path.realpath(src), dst)
        src.unlink()
    else:
        os.rename(src, dst)
    moved.append(m)


def apply(a):
    p = json.load(open(OUT / "plan.json"))
    capped = set(p["capped"])
    moved = []
    for f, sp in p["assign"].items():
        src = Path(f)
        if not os.path.lexists(src):
            continue
        cls = src.parent.name
        if f in capped:
            move(src, CAPPED / cls / src.name, moved)
        elif src.parent.parent.name != sp:
            move(src, DS / sp / cls / src.name, moved)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mp = BASE / f"resplit_manifest_{ts}.json"
    json.dump({"reason": "V19 scene-cluster re-split (09/10); see resplit_by_scene.py",
               "params": {k: p[k] for k in ("thr", "cap_thr", "cap", "giant", "seed")},
               "moved": moved}, open(mp, "w"), indent=0)
    print(f"moved {len(moved)} files -> {mp}")
    for sp in SPLITS:
        print(f"  {sp:<5} bear {len(os.listdir(DS / sp / 'bear')):>6}  non_bear {len(os.listdir(DS / sp / 'non_bear')):>7}")
    print(f"undo: python resplit_by_scene.py --stage reverse --manifest {mp}")


def reverse(a):
    man = json.load(open(a.manifest))
    for m in reversed(man["moved"]):
        src, dst = Path(m["from"]), Path(m["to"])
        if m.get("link"):
            if dst.is_symlink():
                dst.unlink()
            os.symlink(m["link"], src)
        else:
            os.rename(dst, src)
    print(f"restored {len(man['moved'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["sweep", "plan", "apply", "reverse"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.92)
    ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--cap-thr", type=float, default=0.97)
    ap.add_argument("--giant", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest")
    a = ap.parse_args()
    {"sweep": sweep, "plan": plan, "apply": apply, "reverse": reverse}[a.stage](a)
