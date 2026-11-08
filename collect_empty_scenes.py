"""Collect trail-camera frames with NO animal in them.

Bedrock classified all 39 still-image false positives of the 18/09 acceptance
run: 10 of them (26%, the largest category after wild boar) contain no animal
at all -- dark forest, boulders, log piles, a stream, sensor noise. The model
fires on empty scenes. No iNaturalist taxon covers "an empty forest", so the
two sources here are the ones that actually hold such frames:

  lila   LILA BC camera traps, where a human already labelled 1.2M frames
         `empty`. Idaho additionally labels `snow on lens` and `foggy lens`,
         which are the same failure mode seen as "foggy forest" in the FPs,
         so they are collected as their own subtype.
  video  The 4,045 bear-source videos already downloaded under
         collected_bears*/. A bear video is mostly empty scene: the frames
         before the bear walks in are the same camera, same lighting, same
         compression as the positives -- the closest negatives obtainable.

Two sampling rules keep this from becoming thousands of copies of one scene:
  * one frame per `seq_id`. Camera traps fire a 3-frame burst; the frames of
    a burst differ by milliseconds and count as one image, not three.
  * round-robin over `location`. Caltech's busiest camera alone holds 37,766
    frames -- taking them in file order would collect one hillside.

The video path screens with YOLO at a deliberately LOW confidence and rejects
a frame if ANYTHING is detected. A missed animal here is worse than a
discarded frame: it becomes a bear labelled non_bear.

    python collect_empty_scenes.py --source lila  --n 12000
    python collect_empty_scenes.py --source video --n 6000 --gpus 0 1
"""
import argparse
import collections
import itertools
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE = Path("/data/nas07_new/PersonalData/phuocthien")
OUT = BASE / "empty_scenes_1809"

LILA = {
    "caltech": {
        "meta": BASE / "caltech_metadata.json",
        "url": "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images",
        # Caltech labels nothing but `empty`; the lens-artefact classes are Idaho's.
        "labels": {"empty": "empty"},
    },
    "idaho": {
        "meta": BASE / "idaho_metadata.json",
        "url": "https://lilawildlife.blob.core.windows.net/lila-wildlife/idaho-camera-traps/public",
        "labels": {"empty": "empty", "snow on lens": "lens_snow", "foggy lens": "lens_fog"},
    },
}

VIDEO_ROOTS = ["collected_bears", "collected_bears_28_5", "collected_bears_gaps_10_8",
               "collected_bears_fn_gaps_18_8", "collected_bears_night_v2_13_7"]

# COCO ids for every class that is an animal or a person. A frame showing any
# of these is not an empty scene. `bird` is included: one of the false
# positives was a black bird, so birds are a confusable, not background.
COCO_LIVING = set(range(14, 24)) | {0}


# ---------------------------------------------------------------- LILA

def plan_lila(n, seed, only=None):
    """Pick which LILA files to fetch, spread over cameras and subtypes."""
    rng = random.Random(seed)
    per_label = collections.defaultdict(list)   # subtype -> [(loc, dataset, fname)]

    for key, cfg in LILA.items():
        if only and key not in only:
            continue
        print(f"  reading {cfg['meta'].name} ...", flush=True)
        m = json.loads(cfg["meta"].read_text())
        cats = {c["id"]: c["name"] for c in m["categories"]}
        want = {i for i, nm in cats.items() if nm in cfg["labels"]}
        label_of = {}
        for a in m["annotations"]:
            if a["category_id"] in want:
                label_of[a["image_id"]] = cfg["labels"][cats[a["category_id"]]]

        seen_seq = set()
        for im in m["images"]:
            lab = label_of.get(im["id"])
            if lab is None:
                continue
            seq = im.get("seq_id") or im["id"]
            if seq in seen_seq:      # one frame per burst
                continue
            seen_seq.add(seq)
            # Namespace the camera id by dataset: both datasets number their
            # locations from zero, so a bare "26" would merge a Caltech desert
            # camera with an Idaho forest one into a single round-robin bucket
            # and let the 9x larger Idaho pool crowd Caltech out of it.
            per_label[lab].append((f"{key}:{im.get('location', '?')}", key, im["file_name"]))
        print(f"    {key}: " + ", ".join(
            f"{lab} {sum(1 for x in v if x[1]==key)}" for lab, v in per_label.items()), flush=True)

    # Budget: the lens artefacts are a small, distinct failure mode -- give them
    # a fixed slice rather than letting `empty` (1.2M frames) crowd them out.
    quota = {"empty": int(n * 0.80), "lens_snow": int(n * 0.10), "lens_fog": n - int(n*0.80) - int(n*0.10)}
    picked = []
    for lab, items in per_label.items():
        want = quota.get(lab, 0)
        by_loc = collections.defaultdict(list)
        for loc, ds, fn in items:
            by_loc[loc].append((ds, fn))
        for v in by_loc.values():
            rng.shuffle(v)
        # round-robin the cameras so no single one dominates
        got = []
        for row in itertools.zip_longest(*by_loc.values()):
            for cell in row:
                if cell is not None:
                    got.append(cell)
                    if len(got) >= want:
                        break
            if len(got) >= want:
                break
        print(f"  {lab}: {len(got)} chosen from {len(by_loc)} cameras (pool {len(items)})")
        picked += [(lab, ds, fn) for ds, fn in got]
    return picked


def fetch_lila(picked, workers):
    dst_root = OUT / "lila"
    session = requests.Session()
    ok = fail = skip = 0

    def one(item):
        lab, ds, fn = item
        d = dst_root / lab
        d.mkdir(parents=True, exist_ok=True)
        dst = d / f"{ds}_{fn.replace('/', '_')}"
        if dst.exists() and dst.stat().st_size > 1024:
            return "skip"
        url = f"{LILA[ds]['url']}/{fn}"
        for attempt in range(3):
            try:
                r = session.get(url, timeout=30)
                r.raise_for_status()
                dst.write_bytes(r.content)
                return "ok"
            except Exception:
                time.sleep(attempt + 1)
        return "fail"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, it) for it in picked]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            ok += r == "ok"; fail += r == "fail"; skip += r == "skip"
            if i % 500 == 0:
                print(f"    {i}/{len(picked)}  ok {ok}  skip {skip}  fail {fail}", flush=True)
    print(f"  LILA done: {ok} downloaded, {skip} already present, {fail} failed")


# ---------------------------------------------------------------- video

def video_worker(gpu, videos, n_target, every_sec, per_video, conf):
    import cv2
    import torch
    from ultralytics import YOLO

    model = YOLO(str(BASE / "yolov8m.pt"))
    dev = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    dst = OUT / "video"
    dst.mkdir(parents=True, exist_ok=True)
    kept = 0

    for vi, vp in enumerate(videos):
        if kept >= n_target:
            break
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(fps * every_sec))
        idx = 0
        from_this = 0
        while from_this < per_video and kept < n_target:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            okf, frame = cap.read()
            if not okf:
                break
            idx += step
            # Low confidence on purpose: this is a REJECTION test, so a weak,
            # uncertain detection must still disqualify the frame.
            res = model.predict(frame, conf=conf, device=dev, verbose=False)[0]
            if any(int(c) in COCO_LIVING for c in res.boxes.cls.tolist()):
                continue
            name = f"{vp.parent.parent.name}_{vp.parent.name}_{vp.stem[:40]}_f{idx:06d}.jpg"
            cv2.imwrite(str(dst / name), frame)
            kept += 1
            from_this += 1
        cap.release()
        if vi % 25 == 0:
            print(f"  [gpu{gpu}] video {vi}/{len(videos)}  kept {kept}", flush=True)
    print(f"  [gpu{gpu}] finished: {kept} empty frames", flush=True)


def run_video(n, gpus, every_sec, per_video, conf, seed):
    vids = []
    for r in VIDEO_ROOTS:
        for ext in ("*.mp4", "*.mkv", "*.webm"):
            vids += list((BASE / r).rglob(ext))
    random.Random(seed).shuffle(vids)
    print(f"  {len(vids)} videos available")
    if not vids:
        # The collected_bears* source videos were deleted in the 2026-11-04 cleanup.
        sys.exit(f"no source videos under {VIDEO_ROOTS}: collect new ones first or edit VIDEO_ROOTS")

    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    chunks = [vids[i::len(gpus)] for i in range(len(gpus))]
    procs = [ctx.Process(target=video_worker,
                         args=(g, chunks[i], n // len(gpus), every_sec, per_video, conf))
             for i, g in enumerate(gpus)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["lila", "video"])
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16, help="lila: parallel downloads")
    ap.add_argument("--gpus", nargs="+", type=int, default=[0])
    ap.add_argument("--every-sec", type=float, default=4.0, help="video: sampling interval")
    ap.add_argument("--per-video", type=int, default=6, help="video: cap frames per file")
    ap.add_argument("--conf", type=float, default=0.10, help="video: YOLO rejection threshold")
    ap.add_argument("--datasets", nargs="+", choices=list(LILA), help="lila: restrict to these")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if a.source == "lila":
        picked = plan_lila(a.n, a.seed, a.datasets)
        print(f"\n  downloading {len(picked)} images with {a.workers} workers\n", flush=True)
        fetch_lila(picked, a.workers)
    else:
        run_video(a.n, a.gpus, a.every_sec, a.per_video, a.conf, a.seed)


if __name__ == "__main__":
    main()
