"""Map every frame of an acceptance recording back to the benchmark image it
was displaying.

Written for the 11/09 session and parameterised on 19/09 for the 18/09 one,
which reused the same two reference sets in the same order; only the clips,
the session.json files and the output directory differ.

The recording is a trail camera filming a screen: 364 clips of 15s at
1920x1080/30fps, one benchmark image shown roughly every 1.75s, the video
session (2,091 items) first and the image session (1,065 items) after --
3,156 displays in total, which is exactly what 364x15s at 1.75s/item holds.

Why ORB and not a perceptual hash: the screen is reflective and the room
(shelves, a lit PC, ceiling lights, people) is superimposed on every frame.
That reflection is strong and low-frequency, which is the exact band a DCT
hash reads, so hashing scored a 0-2 bit margin between the best and second
best reference out of 2,091 -- indistinguishable from noise, and the same
reference came back as "best" for unrelated segments. ORB keypoints plus a
RANSAC homography survive it, because a reflection adds its own keypoints but
does not displace the displayed image's: measured margins were 35-126 inliers
on the same segments, and the top match was confirmed by eye.

Why the session order matters: brute force is 3,276 segments x 3,156
references = 10.3M ORB matches, about 86 hours. session.json records the exact
display order, so each segment only has to be searched in a window around the
expected position -- ~66k matches, well under an hour. The order constraint
also resolves ties that per-frame evidence alone cannot.

The recording is NOT continuous (motion-triggered clips, plus session breaks
including a 10.5h overnight gap), so the aligner does not assume one anchor
holds throughout: it tracks consecutive weak matches and re-anchors with a
global search when it loses the sequence.

    python match_acceptance_frames.py --stage segments
    python match_acceptance_frames.py --stage match
    python match_acceptance_frames.py --stage export
"""
import argparse
import json
import os
import pickle
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

BASE = Path("/data/nas07_new/PersonalData/phuocthien")
CLIPS = BASE / "acceptance_1109_clips"
OUT = BASE / "acceptance_1109_match"
REF_VIDEO = Path("/tmp/refs/video_categorized_grouped_1in6")
REF_IMAGES = BASE / "acceptance_merged"
SESS_VIDEO = Path("/tmp/s1109/_acceptance_session/session.json")
SESS_IMAGES = Path("/tmp/si1109/_acceptance_session/session.json")

WORK = 480, 270          # working resolution for both refs and frames
N_FEATURES = 800
RATIO = 0.75             # Lowe ratio
MIN_INLIERS = 18         # below this a match is not trusted
MIN_MARGIN = 8           # inlier gap to the runner-up
WINDOW = 25              # +/- positions searched around the expected index
LOST_AFTER = 6           # consecutive weak matches before a global re-anchor
SEG_MIN_FRAMES = 8       # ignore sub-segments shorter than this (transitions)
TIE_RATIO = 0.8          # a runner-up within this fraction of the top is a tie


def orb():
    return cv2.ORB_create(nfeatures=N_FEATURES)


# ----------------------------------------------------------------- references
def build_reference_list():
    """One ordered list of the 3,156 expected displays: video session then
    image session, matching how the recording was made."""
    items = []
    for tag, sess, root in (("video", SESS_VIDEO, REF_VIDEO),
                            ("images", SESS_IMAGES, REF_IMAGES)):
        if sess is None:      # --sess-images none: a video-only recording
            continue
        d = json.load(open(sess))
        for it in d["items"]:
            items.append({"session": tag, "file": it["file"].replace("\\", "/"),
                          "truth": it["truth"], "group": it.get("video"),
                          "path": str(root / it["file"].replace("\\", "/"))})
    return items


def _describe(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    kp, des = orb().detectAndCompute(cv2.resize(g, WORK), None)
    if des is None or len(kp) < 10:
        return None
    return (np.float32([p.pt for p in kp]), des)


def stage_references():
    items = build_reference_list()
    print(f"references: {len(items)}", flush=True)
    with Pool(16) as pool:
        desc = pool.map(_describe, [it["path"] for it in items], chunksize=16)
    ok = sum(1 for d in desc if d is not None)
    print(f"  described: {ok}/{len(items)}", flush=True)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "refs.pkl", "wb") as f:
        pickle.dump({"items": items, "desc": desc}, f)
    print(f"  wrote {OUT/'refs.pkl'}", flush=True)


# ------------------------------------------------------------------- segments
def _segment_clip(clip_path):
    """Split one clip into stable display segments and describe the middle
    frame of each. Returns (clip_name, [segment dicts])."""
    cap = cv2.VideoCapture(clip_path)
    frames, small = [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        # Keep only the two downscaled copies the code below reads; holding
        # full 1080p frames cost ~0.9 GB per clip per worker.
        frames.append(cv2.resize(g, WORK))
        small.append(cv2.resize(g, (160, 90)).astype(np.float32))
    cap.release()
    if len(frames) < SEG_MIN_FRAMES:
        return Path(clip_path).stem, []

    d = np.array([np.abs(small[i] - small[i - 1]).mean() for i in range(1, len(small))])
    thr = max(d.mean() + 3 * d.std(), 3.0)
    cuts = [0] + [i + 1 for i in np.where(d > thr)[0]] + [len(frames)]

    segs = []
    o = orb()
    for a, b in zip(cuts, cuts[1:]):
        if b - a < SEG_MIN_FRAMES:
            continue
        mid = (a + b) // 2
        kp, des = o.detectAndCompute(frames[mid], None)
        if des is None or len(kp) < 10:
            continue
        segs.append({"start": int(a), "end": int(b), "mid": int(mid),
                     "pts": np.float32([p.pt for p in kp]), "des": des})
    return Path(clip_path).stem, segs


def stage_segments():
    clips = sorted(CLIPS.glob("*.MP4"), key=lambda p: int(p.stem.replace("MFDC", "")))
    # MFDC0496 is stored first in the archive but its number puts it last;
    # numeric order is the recording order, so sorting by number is correct.
    print(f"clips: {len(clips)}", flush=True)
    OUT.mkdir(exist_ok=True)
    results = []
    with Pool(24) as pool:
        for i, (name, segs) in enumerate(
                pool.imap(_segment_clip, [str(c) for c in clips], chunksize=2), 1):
            results.append((name, segs))
            if i % 40 == 0:
                print(f"  {i}/{len(clips)} clips segmented", flush=True)
    results.sort(key=lambda r: int(r[0].replace("MFDC", "")))
    total = sum(len(s) for _, s in results)
    print(f"  segments: {total} (mean {total/max(len(results),1):.1f}/clip)", flush=True)
    with open(OUT / "segments.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"  wrote {OUT/'segments.pkl'}", flush=True)


# -------------------------------------------------------------------- matching
_REF = None


def _init_ref(refs):
    global _REF
    _REF = refs


def _score(args):
    """Inlier count of one (segment, reference) pair."""
    idx, q_pts, q_des = args
    r = _REF["desc"][idx]
    if r is None:
        return idx, 0
    r_pts, r_des = r
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    m = bf.knnMatch(q_des, r_des, k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < RATIO * b.distance]
    if len(good) < 8:
        return idx, 0
    src = q_pts[[a.queryIdx for a in good]].reshape(-1, 1, 2)
    dst = r_pts[[a.trainIdx for a in good]].reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return idx, int(mask.sum()) if mask is not None else 0


def _source_video(item):
    """Benchmark frames are named <youtube id>_<frame>.jpg; frames of one
    source video share the id."""
    return item["group"], Path(item["file"]).stem.rsplit("_", 1)[0]


def _best(pool, seg, candidates, expected):
    """Best reference for a segment: (idx, inliers, runner-up, resolved_by_order).

    Several benchmark videos are near-static (a den camera, a fixed trail cam),
    so two of their frames can match a display almost equally well: on 18/09,
    37 segments with 61-152 inliers were rejected on margins of 1-8. When every
    reference within MIN_MARGIN of the top one comes from the same source
    video, the label is not in doubt; pick the one nearest the expected
    position instead of throwing the segment away.

    The tie test is relative (TIE_RATIO), not just MIN_MARGIN: on 02/10 a
    den-camera frame matched 178 vs 164 -- a "clear" 14-inlier win for a
    reference 20 positions ahead of the one on screen. The label survived (same
    video) but the expected position jumped, and the next correct match then
    looked out of order."""
    args = [(i, seg["pts"], seg["des"]) for i in candidates]
    scores = pool.map(_score, args, chunksize=4)
    scores.sort(key=lambda x: -x[1])
    top_idx, top = scores[0]
    second = scores[1][1] if len(scores) > 1 else top
    if top >= MIN_INLIERS and expected is not None and (
            top - second < MIN_MARGIN or second >= TIE_RATIO * top):
        close = [i for i, n in scores if n > top - MIN_MARGIN or n >= TIE_RATIO * top]
        if len({_source_video(_REF["items"][i]) for i in close}) == 1:
            pick = min(close, key=lambda i: (abs(i - expected), i))
            rest = max((n for i, n in scores if i not in close), default=0)
            return pick, top, rest, True
    return top_idx, top, second, False


def stage_match():
    refs = pickle.load(open(OUT / "refs.pkl", "rb"))
    segments = pickle.load(open(OUT / "segments.pkl", "rb"))
    _init_ref(refs)       # _best() reads item metadata in this process too
    n_ref = len(refs["items"])
    all_idx = list(range(n_ref))

    out_path = OUT / "matches.jsonl"
    done = set()
    if out_path.exists():
        for line in open(out_path):
            r = json.loads(line)
            done.add((r["clip"], r["seg"]))
        print(f"resuming, {len(done)} segments already matched", flush=True)

    fh = open(out_path, "a")
    expected = None
    last_acc = None
    lost = 0
    n_done = n_ok = 0

    with Pool(36, initializer=_init_ref, initargs=(refs,)) as pool:
        for clip, segs in segments:
            for si, seg in enumerate(segs):
                if (clip, si) in done:
                    continue
                if expected is None or lost >= LOST_AFTER:
                    cands = all_idx
                    mode = "global"
                else:
                    lo, hi = max(0, expected - WINDOW), min(n_ref, expected + WINDOW + 1)
                    cands = list(range(lo, hi))
                    mode = "window"

                idx, inl, second, by_order = _best(pool, seg, cands, expected)
                strong = inl >= MIN_INLIERS and (inl - second) >= MIN_MARGIN
                # Going backwards is recorded, not rejected: on 02/10 every one
                # checked by eye was a correct match -- either the operator
                # pressed UNDO (the previous image is shown again) or the
                # PREVIOUS match had picked a near-identical frame further ahead.
                step = None if last_acc is None else idx - last_acc
                out_of_order = strong and step is not None and step < 0 and mode == "window"

                if strong:
                    last_acc = idx
                    expected = idx + 1
                    lost = 0
                    n_ok += 1
                else:
                    lost += 1
                    if expected is not None:
                        expected += 1

                it = refs["items"][idx]
                fh.write(json.dumps({
                    "clip": clip, "seg": si, "start": seg["start"], "end": seg["end"],
                    "mid": seg["mid"], "mode": mode, "ref_idx": idx,
                    "inliers": inl, "second": second, "margin": inl - second,
                    "accepted": bool(strong), "by_order": by_order,
                    "out_of_order": bool(out_of_order), "session": it["session"],
                    "ref_file": it["file"], "truth": it["truth"],
                }) + "\n")
                fh.flush()
                n_done += 1
                if n_done % 50 == 0:
                    print(f"  {n_done} segments  accepted={n_ok} "
                          f"({100*n_ok/max(n_done,1):.0f}%)  expected_idx={expected}",
                          flush=True)
    fh.close()
    print(f"\ndone: {n_done} segments, {n_ok} accepted", flush=True)


# ---------------------------------------------------------------------- export
def stage_export():
    import csv
    rows = [json.loads(l) for l in open(OUT / "matches.jsonl")]
    rows.sort(key=lambda r: (int(r["clip"].replace("MFDC", "")), r["seg"]))
    frames_dir = OUT / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    by_clip = {}
    for r in rows:
        if r["accepted"]:
            by_clip.setdefault(r["clip"], []).append(r)

    n = 0
    for clip, rs in sorted(by_clip.items()):
        cap = cv2.VideoCapture(str(CLIPS / f"{clip}.MP4"))
        for r in rs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, r["mid"])
            ok, f = cap.read()
            if not ok:
                continue
            stem = Path(r["ref_file"]).stem
            cv2.imwrite(str(frames_dir / f"{r['session']}__{r['truth']}__{stem}__{clip}_{r['mid']:04d}.jpg"), f)
            n += 1
        cap.release()

    with open(OUT / "matches.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    acc = sum(1 for r in rows if r["accepted"])
    print(f"segments: {len(rows)}, accepted: {acc} ({100*acc/len(rows):.1f}%)")
    print(f"  of which resolved by order: {sum(1 for r in rows if r['accepted'] and r.get('by_order'))}")
    print(f"  out of order (exported, flagged): {sum(1 for r in rows if r.get('out_of_order'))}")
    print(f"distinct displays captured: {len({r['ref_idx'] for r in rows if r['accepted']})}")
    print(f"frames exported: {n} -> {frames_dir}")
    print(f"table: {OUT/'matches.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["references", "segments", "match", "export"])
    # Paths default to the 11/09 run so the original invocations still work.
    ap.add_argument("--clips", help="directory of MFDC*.MP4 clips")
    ap.add_argument("--out", help="output directory for refs/segments/matches")
    ap.add_argument("--ref-video", help="root of the video-session reference images")
    ap.add_argument("--ref-images", help="root of the image-session reference images")
    ap.add_argument("--sess-video", help="video session.json")
    ap.add_argument("--sess-images", help="image session.json")
    a = ap.parse_args()
    for arg, name in (("clips", "CLIPS"), ("out", "OUT"),
                      ("ref_video", "REF_VIDEO"), ("ref_images", "REF_IMAGES"),
                      ("sess_video", "SESS_VIDEO"), ("sess_images", "SESS_IMAGES")):
        v = getattr(a, arg)
        if v:
            globals()[name] = None if v.lower() == "none" else Path(v)
    {"references": stage_references, "segments": stage_segments,
     "match": stage_match, "export": stage_export}[a.stage]()
