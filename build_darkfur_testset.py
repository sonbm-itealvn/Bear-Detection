"""A held-out daylight test set of dark-furred non-bear animals.

The customer tested V16 themselves and found that photos and videos of
dark-furred animals -- foxes, rabbits -- pulled from the web get called bears.
Before collecting training data for that, measure it: how often does V16 (and
the model currently training) fire on such images, per species?

The training set's coverage of these animals is lopsided in exactly the way
that matters: 2,831 fox images but ~2,600 of them are night-vision trail-cam
frames from a handful of YouTube videos, only 221 daylight; 222 rabbit/hare
images and every one of them a night trail-cam frame. The customer's test is
daylight photos from the web.

Held-out by construction:
  * newest research-grade observations first -- the training collector pages
    `order_by=id, order=asc`, so it took the OLDEST ones;
  * any observation whose id appears anywhere in the dataset's filenames is
    skipped, so nothing here can have been trained on.

    python build_darkfur_testset.py --per-species 120
"""
import argparse
import os
import re
import time
from pathlib import Path

import requests

BASE = Path("/data/nas07_new/PersonalData/phuocthien")
DS = BASE / "bear_dataset_v_final_final"
OUT = BASE / "darkfur_test_2409"

# Verified against the iNaturalist API on 2026-09-24 (name match, rank=species).
TAXA = {
    "red_fox": 42069, "gray_fox": 42076,
    "japanese_hare": 43139, "european_rabbit": 43151, "eastern_cottontail": 43111,
    "japanese_marten": 41794, "japanese_badger": 74557, "masked_palm_civet": 41625,
    "domestic_cat": 118552, "raccoon_dog": 855311,
    "japanese_squirrel": 46000, "eurasian_red_squirrel": 46001,
    "american_mink": 1264432, "eurasian_otter": 41850,
}


def training_obs_ids():
    ids = set()
    for split in ("train", "val", "test"):
        for cls in ("bear", "non_bear", "boar", "deer", "monkey"):
            d = DS / split / cls
            if not d.exists():
                continue
            for n in os.listdir(d):
                ids.update(re.findall(r"(?<!\d)(\d{6,10})(?!\d)", n))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-species", type=int, default=120)
    a = ap.parse_args()

    seen = training_obs_ids()
    print(f"  {len(seen)} id quan sát đã có trong dataset (sẽ bị loại)", flush=True)
    s = requests.Session()
    for sp, tid in TAXA.items():
        d = OUT / sp
        d.mkdir(parents=True, exist_ok=True)
        have = len(list(d.glob("*.jpg")))
        page, skipped = 1, 0
        while have < a.per_species and page <= 15:
            try:
                r = s.get("https://api.inaturalist.org/v1/observations", timeout=40, params={
                    "taxon_id": tid, "quality_grade": "research", "photos": "true",
                    "order_by": "id", "order": "desc", "per_page": 200, "page": page}).json()
            except Exception as e:
                print(f"    {sp}: API lỗi trang {page}: {e}", flush=True)
                time.sleep(5)
                continue
            res = r.get("results", [])
            if not res:
                break
            for o in res:
                if have >= a.per_species:
                    break
                oid = str(o["id"])
                if oid in seen:
                    skipped += 1
                    continue
                photos = o.get("photos") or []
                if not photos:
                    continue
                url = photos[0]["url"].replace("square", "large")
                dst = d / f"{sp}_{oid}.jpg"
                if dst.exists():
                    continue
                try:
                    img = s.get(url, timeout=40)
                    if img.status_code == 200 and len(img.content) > 5000:
                        dst.write_bytes(img.content)
                        have += 1
                except Exception:
                    pass
            page += 1
            time.sleep(1.1)
        print(f"  {sp:<24} {have:>4} ảnh  (bỏ {skipped} quan sát đã có trong train)", flush=True)


if __name__ == "__main__":
    main()
