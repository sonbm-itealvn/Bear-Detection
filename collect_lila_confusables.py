"""Collect camera-trap frames of dark-furred confusable mammals from LILA BC.

The 26/09 dark-fur false positives that survived the daylight-photo merge fell
into two groups this script targets:
  small  the animal is small/distant in the frame (a badger on a far road, an
         otter far out in water) -- DistanceSimulation made "small subject"
         a bear-only feature
  ir     close-up night IR shots of pale mustelids -- NightVisionSimulation
         made "IR look" a bear-only feature
Camera traps are full of both, and every frame here has a HUMAN species label
and a bounding box, so no LLM judge gets to veto anything (on 24/09 all three
judges proved unreliable on confusable negatives; see CLAUDE.md).

Sources (all with boxes):
  swg      SWG Camera Traps, Laos/Vietnam -- ferret badger, yellow-throated
           marten, civets, linsang, mongoose, squirrels: the Asian species
  caltech  Caltech Camera Traps bbox subset, SW USA -- raccoon, cat, fox, ...
  ena24    ENA24, eastern N. America -- foxes, cats, squirrels, skunk, ...
Images that carry ANY bear annotation are excluded outright (SWG has Asiatic
black and sun bears, ENA24 has American black bears).

Sampling: one frame per seq_id (bursts are one image, not three), round-robin
over camera locations, capped per species; within a species half the quota
goes to the smallest boxes and half to the largest (close-ups). IR is decided
after download from the pixels (HSV saturation), then images are downscaled
to 1280 px -- the trainer's input is 640x360.

    python collect_lila_confusables.py --per-species 600 --dry-run
    python collect_lila_confusables.py --per-species 600
"""
import argparse
import collections
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from PIL import Image

BASE = Path("/data/nas07_new/PersonalData/phuocthien")
META = BASE / "lila_meta"
OUT = BASE / "lila_confusables_2609"
MANIFEST = OUT / "manifest.jsonl"

DATASETS = {
    "swg": {
        "meta": "swg_camera_traps.bounding_boxes.with_species.json",
        "url": "https://lilawildlife.blob.core.windows.net/lila-wildlife/swg-camera-traps/",
        "species": {
            "ferret_badger": "ferret_badger", "hog_badger": "hog_badger",
            "yellow_throated_marten": "yellow_throated_marten",
            "masked_palm_civet": "masked_palm_civet", "common_palm_civet": "common_palm_civet",
            "large_indian_civet": "large_indian_civet", "owstons_civet": "owstons_civet",
            "unidentified_palm_civet": "palm_civet", "spotted_linsang": "spotted_linsang",
            "crab_eating_mongoose": "crab_eating_mongoose",
            "red_cheeked_squirrel": "red_cheeked_squirrel", "pallass_squirrel": "pallass_squirrel",
            "black_giant_squirrel": "black_giant_squirrel",
            "unidentified_weasel": "weasel", "back_striped_weasel": "weasel",
            "unidentified_otter": "otter", "leopard_cat": "leopard_cat",
            "annamite_striped_rabbit": "striped_rabbit",
            "asiatic_brush_tailed_porcupine": "porcupine", "malayan_porcupine": "porcupine",
        },
        "bear": {"asiatic_black_bear", "sun_bear", "unidentified_bear"},
    },
    "caltech": {
        "meta": "caltech_bboxes_20200316.json",
        "url": "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images/",
        "species": {"raccoon": "raccoon", "cat": "cat", "fox": "fox", "skunk": "skunk",
                    "squirrel": "squirrel", "badger": "badger", "bobcat": "bobcat",
                    "rabbit": "rabbit", "opossum": "opossum"},
        "bear": set(),
    },
    "ena24": {
        "meta": "ena24.json",
        "url": "https://lilawildlife.blob.core.windows.net/lila-wildlife/ena24/images/",
        "species": {"Grey Fox": "grey_fox", "Red Fox": "red_fox", "Domestic Cat": "cat",
                    "Eastern Fox Squirrel": "squirrel", "Eastern Gray Squirrel": "squirrel",
                    "Striped Skunk": "skunk", "Northern Raccoon": "raccoon",
                    "Bobcat": "bobcat", "Virginia Opossum": "opossum"},
        "bear": {"American Black Bear"},
    },
}


def box_frac(bbox, w, h):
    x, y, bw, bh = bbox
    if max(bbox) <= 1.0:            # normalized boxes
        return bw * bh
    return (bw * bh) / float(w * h)


def plan(per_species, seed):
    rng = random.Random(seed)
    picks = []
    for key, cfg in DATASETS.items():
        m = json.loads((META / cfg["meta"]).read_text())
        cats = {c["id"]: c["name"] for c in m["categories"]}
        imgs = {im["id"]: im for im in m["images"]}
        by_img = collections.defaultdict(list)
        for a in m["annotations"]:
            by_img[a["image_id"]].append(a)
        # species -> location -> {seq -> best candidate}
        pool = collections.defaultdict(lambda: collections.defaultdict(dict))
        for iid, anns in by_img.items():
            names = {cats[a["category_id"]] for a in anns}
            if names & cfg["bear"]:
                continue
            sp = [cfg["species"][n] for n in names if n in cfg["species"]]
            if len(set(sp)) != 1:          # no target, or two different animals
                continue
            im = imgs[iid]
            fr = max((box_frac(a["bbox"], im.get("width", 1), im.get("height", 1))
                      for a in anns if "bbox" in a), default=None)
            if fr is None:
                continue
            seq = im.get("seq_id") or iid
            loc = str(im.get("location", "na"))
            pool[sp[0]][loc].setdefault(seq, (fr, im["file_name"]))
        for sp, locs in pool.items():
            cands = [(fr, loc, fn) for loc, seqs in locs.items() for fr, fn in seqs.values()]
            cands.sort()
            half = per_species // 2
            small = round_robin([c for c in cands if c[0] < 0.03], half, rng)
            # True close-ups (>25% of the frame) are rare in camera traps --
            # only raccoon and bobcat have many -- so "close" means this
            # species' largest quarter of boxes, not a fixed threshold.
            q75 = cands[int(len(cands) * 0.75)][0] if cands else 1.0
            large = round_robin([c for c in cands if c[0] >= max(q75, 0.03)], half, rng)
            chosen = small + large
            if len(chosen) < per_species:       # top up from the middle
                got = {c[2] for c in chosen}
                rest = [c for c in cands if c[2] not in got]
                chosen += round_robin(rest, per_species - len(chosen), rng)
            for fr, loc, fn in chosen:
                picks.append({"dataset": key, "species": sp, "location": loc,
                              "box_frac": round(fr, 5), "file_name": fn,
                              "url": cfg["url"] + fn})
        del m
    return picks


def round_robin(cands, n, rng):
    by_loc = collections.defaultdict(list)
    for c in cands:
        by_loc[c[1]].append(c)
    for v in by_loc.values():
        rng.shuffle(v)
    out, locs = [], list(by_loc)
    rng.shuffle(locs)
    while len(out) < n and any(by_loc.values()):
        for loc in locs:
            if by_loc[loc] and len(out) < n:
                out.append(by_loc[loc].pop())
    return out


def fetch(p):
    dst = OUT / f"{p['dataset']}_{p['species']}" / (
        p["file_name"].replace("/", "__").rsplit(".", 1)[0] + ".jpg")
    if dst.exists():
        return p, dst, "exists"
    for _ in range(3):
        try:
            r = requests.get(p["url"], timeout=60)
            if r.status_code == 200:
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_suffix(".part")
                tmp.write_bytes(r.content)
                im = Image.open(tmp).convert("RGB")
                im.thumbnail((1280, 1280))
                im.save(dst, quality=90)
                tmp.unlink()
                return p, dst, "ok"
        except Exception:
            pass
    return p, dst, "fail"


def saturation(path):
    im = Image.open(path)
    im.draft("RGB", (160, 160))
    hsv = np.asarray(im.convert("RGB").resize((96, 54)).convert("HSV"), dtype=np.float32)
    return float(hsv[..., 1].mean() / 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-species", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    picks = plan(a.per_species, a.seed)
    c = collections.Counter((p["dataset"], p["species"]) for p in picks)
    for (d, s), n in sorted(c.items()):
        fr = [p["box_frac"] for p in picks if p["dataset"] == d and p["species"] == s]
        print(f"  {d:8s} {s:24s} {n:5d}   small(<3%) {sum(f < .03 for f in fr):4d}"
              f"   >=10% {sum(f >= .10 for f in fr):4d}   median {np.median(fr)*100:5.1f}%")
    print(f"  TOTAL {len(picks)}")
    if a.dry_run:
        return

    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if MANIFEST.exists():
        done = {json.loads(l)["url"] for l in open(MANIFEST)}
    todo = [p for p in picks if p["url"] not in done]
    print(f"  downloading {len(todo)} (already have {len(done)})", flush=True)
    stats = collections.Counter()
    with open(MANIFEST, "a") as fh, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fetch, p) for p in todo]
        for i, f in enumerate(as_completed(futs), 1):
            p, dst, st = f.result()
            stats[st] += 1
            if st in ("ok", "exists"):
                p = dict(p, path=str(dst.relative_to(OUT)), sat=round(saturation(dst), 4))
                fh.write(json.dumps(p) + "\n")
            if i % 500 == 0:
                fh.flush()
                print(f"    {i}/{len(todo)} {dict(stats)}", flush=True)
    print(f"  done {dict(stats)}")


if __name__ == "__main__":
    main()
