"""Sort the camera-filmed frames of the 07-08/10 capture of capture_package_1006.

match_acceptance_frames.py mapped each frame of the 372 clips (100MFCAM.zip) to
the package image on screen (train_NNNNN.jpg). This joins each frame to:
  * its truth from the package's labels.csv (bear / non_bear),
  * the package group (bear_hard, bear_ir, darkfur_hard, darkfur_random, lila_*),
  * the source image in the dataset (capture_package_1006/manifest.csv),
  * what the camera answered (session responses.jsonl, undo-aware),
and DROPS frames whose source image was quarantined after the package was built
(audit round 2, 07/10: 141 package images) -- their labels are no longer trusted.

Writes frames_by_outcome/<truth>/... plus frames.csv in the layout
merge_acceptance_frames_to_train.py --frames-csv reads.

    python split_capture_frames_1009.py
"""
import collections
import csv
import glob
import json
import shutil
from pathlib import Path

BASE = Path("/data/nas07_new/PersonalData/phuocthien")
MATCH = BASE / "acceptance_1009_match"
PKG = BASE / "capture_package_1006"
SESSION = BASE / "capture_1009/_acceptance_session"


def replay(path):
    ans = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        f = r.get("file", "").replace("\\", "/")
        if r.get("event") == "undo":
            ans.pop(f, None)
        elif "answer" in r and f:
            ans[f] = r["answer"]
    return ans


def main():
    man = {r["name"]: r for r in csv.DictReader(open(PKG / "manifest.csv"))}
    answer = replay(SESSION / "responses.jsonl")
    quarantined = set()
    for m in glob.glob(str(BASE / "audit_quarantine_round2*_manifest_*.json")):
        for x in json.load(open(m))["moved"]:
            quarantined.add(x["from"].replace(str(BASE) + "/", ""))

    rows = [r for r in csv.DictReader(open(MATCH / "matches.csv")) if r["accepted"] == "True"]
    out = MATCH / "frames_by_outcome"
    shutil.rmtree(out, ignore_errors=True)
    kept, stats = [], collections.Counter()
    for r in rows:
        m = man.get(r["ref_file"])
        if m is None:
            stats["not in package manifest"] += 1
            continue
        if m["source"] in quarantined:
            stats["source quarantined (dropped)"] += 1
            continue
        t, v = m["truth"], answer.get(r["ref_file"], "none")
        src = MATCH / "frames" / f"{r['session']}__{r['truth']}__{Path(r['ref_file']).stem}__{r['clip']}_{int(r['mid']):04d}.jpg"
        if not src.exists():
            stats["frame missing"] += 1
            continue
        d = out / t
        d.mkdir(parents=True, exist_ok=True)
        name = f"{t}__{m['group']}__{Path(r['ref_file']).stem}__{r['clip']}_{int(r['mid']):04d}.jpg"
        shutil.copy2(src, d / name)
        stats[t] += 1
        kept.append({"frame": f"{t}/{name}", "ref_file": r["ref_file"], "true_class": t,
                     "outcome": m["group"], "device_answer": v, "source": m["source"],
                     "clip": r["clip"], "mid": r["mid"], "inliers": r["inliers"]})
    with open(out / "frames.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(kept[0].keys()))
        w.writeheader()
        w.writerows(kept)
    refs = {k["ref_file"] for k in kept}
    print(f"{len(rows)} matched frames -> {len(kept)} kept, {len(refs)} distinct package images")
    for k, n in stats.most_common():
        print(f"   {k:<32}{n:>6}")
    print("  by group:", dict(collections.Counter(k["outcome"] for k in kept)))
    # Camera answers per group, one per distinct image (the session's own report
    # was written after day 1 only).
    per = collections.defaultdict(collections.Counter)
    for name, a in answer.items():
        m = man.get(name)
        if m:
            per[m["group"]][a] += 1
    print("  camera answers per group (all answered images):")
    for g, c in sorted(per.items()):
        n = sum(c.values())
        print(f"    {g:<16} n={n:>5}  said bear {c['bear']:>5} ({100 * c['bear'] / n:5.1f}%)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
