"""
verify_screencap.py

Verify screen-captured frames with Claude Bedrock before merging into dataset.

For bear frames:     KEEP=YES, REJECT=NO, REVIEW=UNSURE/ERROR/TIMEOUT
For non_bear frames: KEEP=NO,  REJECT=YES, REVIEW=UNSURE/ERROR/TIMEOUT

UNSURE/ERROR/TIMEOUT images are moved to _review/ folder, not auto-merged.
Rejected images are moved to _rejected/ folder, not deleted.

Results saved to screencap_verify_cache.pkl — reusable in merge_new_animals.py

Usage:
    # Dry run (default) — report only, no file moves
    python verify_screencap.py \\
        --bear-dir frames/bear \\
        --nonbear-dir frames/non_bear

    # Apply — move rejected/review images
    python verify_screencap.py \\
        --bear-dir frames/bear \\
        --nonbear-dir frames/non_bear \\
        --apply
"""

import os
import sys
import io
import time
import random
import pickle
import shutil
import argparse
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import boto3
except ImportError:
    raise ImportError("Please install boto3 first: pip install boto3")

try:
    from PIL import Image as PILImage
except ImportError:
    raise ImportError("Please install Pillow first: pip install Pillow")

EXTS = {'.jpg', '.jpeg', '.png'}

PROMPT = """You are an expert wildlife biologist reviewing trail camera images.
This image was captured by a camera recording a screen displaying trail camera footage.
Is there a real, live bear visible in the image (even through screen artifacts, moire, glare)?
Consider: night vision, IR, blurry, distant, partial view, screen reflections.
Reply ONE WORD: YES / NO / UNSURE"""

from PIL import ImageStat

try:
    import imagehash
except ImportError:
    imagehash = None  # dedup auto-disabled in main(); dedup_and_cap raises loudly if reached


def _frame_features(fp):
    """Return (mean, std, phash) from a 512px grayscale thumbnail, or None if unreadable.

    Only genuine image-read failures return None. A missing `imagehash` dependency is NOT
    masked here as an unreadable frame — dedup_and_cap raises a clear error instead.
    """
    try:
        img = PILImage.open(fp)
        img.thumbnail((512, 512))
        gray = img.convert("L")
        stat = ImageStat.Stat(gray)
        return float(stat.mean[0]), float(stat.stddev[0]), imagehash.phash(gray)
    except Exception:
        return None


BLACK_MEAN = 25.0
BLACK_STD = 12.0


def dedup_and_cap(img_paths, expected_label,
                  keep_per_cluster=3, hamming_thr=6, max_black=30,
                  black_mean=BLACK_MEAN, black_std=BLACK_STD, seed=42):
    """Cluster near-duplicate frames by phash (keep K per cluster) and cap dark frames.

    Near-duplicate dedup applies to all labels. The dark-frame cap (mean<black_mean AND
    std<black_std, keep <= max_black) applies ONLY when expected_label == 'non_bear'.
    Unreadable images are always kept. Pure compute: performs no file moves.
    Returns (survivors: list[Path], dropped: list[Path], stats: dict).

    stats['n_black'] counts dark frames surviving dedup BEFORE the cap is applied — it is
    a diagnostic of how dark the post-dedup set is, NOT the count of dark frames finally
    kept (the cap may drop some afterwards).
    """
    if imagehash is None:
        raise RuntimeError("dedup_and_cap requires the 'imagehash' package "
                           "(pip install imagehash)")
    paths = list(img_paths)
    feats, unreadable = {}, []
    for i, p in enumerate(paths):
        f = _frame_features(p)
        if f is None:
            unreadable.append(p)
        else:
            feats[i] = f

    idxs = sorted(feats.keys())
    parent = {i: i for i in idxs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # O(n^2) pairwise Hamming — fine for per-directory frame counts (hundreds / low
    # thousands). Do NOT call this on the full ~133k dataset at once.
    for ai in range(len(idxs)):
        for bi in range(ai + 1, len(idxs)):
            i, j = idxs[ai], idxs[bi]
            if (feats[i][2] - feats[j][2]) <= hamming_thr:
                union(i, j)

    clusters = {}
    for i in idxs:
        clusters.setdefault(find(i), []).append(i)

    survivors, dropped = [], []
    for members in clusters.values():
        ordered = sorted(members, key=lambda i: (-feats[i][1], paths[i].name))
        survivors.extend(ordered[:keep_per_cluster])
        dropped.extend(ordered[keep_per_cluster:])

    def is_black(i):
        return feats[i][0] < black_mean and feats[i][1] < black_std

    n_black = sum(1 for i in survivors if is_black(i))
    if expected_label == "non_bear":
        black = [i for i in survivors if is_black(i)]
        if len(black) > max_black:
            rng = random.Random(seed)
            keep_black = set(rng.sample(sorted(black, key=lambda i: paths[i].name), max_black))
            black_set = set(black)
            kept = []
            for i in survivors:
                if i in black_set and i not in keep_black:
                    dropped.append(i)
                else:
                    kept.append(i)
            survivors = kept

    survivor_paths = [paths[i] for i in survivors] + unreadable
    dropped_paths = [paths[i] for i in dropped]
    stats = {
        "total": len(paths),
        "readable": len(feats),
        "n_unreadable": len(unreadable),
        "n_clusters": len(clusters),
        "n_black": n_black,
        "n_dropped": len(dropped_paths),
        "n_kept": len(survivor_paths),
    }
    return survivor_paths, dropped_paths, stats


def parse_answer(raw_text):
    """Parse Claude response — use first word for safety."""
    ans = raw_text.strip().upper()
    first = ans.split()[0].strip(".,:;!?") if ans.split() else "UNSURE"
    if first == "YES":    return "YES"
    elif first == "NO":   return "NO"
    elif first == "UNSURE": return "UNSURE"
    else:                 return "UNSURE"


def get_client():
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    )


def check_one(fp, client, model_id):
    retries, backoff = 6, 3
    while retries > 0:
        try:
            img = PILImage.open(fp).convert("RGB")
            img.thumbnail((512, 512))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            resp = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [
                    {"image": {"format": "jpeg", "source": {"bytes": buf.getvalue()}}},
                    {"text": PROMPT}
                ]}],
                inferenceConfig={"maxTokens": 10, "temperature": 0.0}
            )
            raw = resp['output']['message']['content'][0]['text']
            return parse_answer(raw)
        except Exception as e:
            err = str(e).lower()
            if "throttling" in err or "429" in err or "503" in err:
                time.sleep(backoff + random.uniform(1, 2))
                retries -= 1; backoff *= 2
            else:
                return "ERROR"
    return "TIMEOUT"


def verify_dir(src_dir, expected_label, cache, cache_path,
               client, model_id, max_workers=5, apply=False,
               dedup=True, keep_per_cluster=3, hamming_thr=6, max_black=30,
               claude_verify=True):
    src_dir = Path(src_dir).resolve()
    if not src_dir.exists():
        print(f"  [SKIP] {src_dir} not found")
        return

    all_imgs = [p for p in src_dir.rglob('*') if p.suffix.lower() in EXTS]

    # Dedup/cap pre-pass: collapse near-duplicate frames and cap dark non_bear frames
    # BEFORE Claude verification, so dropped frames don't burn API calls.
    if dedup and all_imgs:
        survivors, dropped, stats = dedup_and_cap(
            all_imgs, expected_label,
            keep_per_cluster=keep_per_cluster,
            hamming_thr=hamming_thr, max_black=max_black,
        )
        extra = f", unreadable={stats['n_unreadable']}" if stats['n_unreadable'] else ""
        print(f"  [dedup] {expected_label}: total={stats['total']}, "
              f"clusters={stats['n_clusters']}, black={stats['n_black']}, "
              f"dropped={stats['n_dropped']}, kept={stats['n_kept']}{extra}")
        if apply and dropped:
            dedup_dir = src_dir.parent / f"{src_dir.name}_dedup"
            for fp in dropped:
                # Preserve subdirectory structure so basename collisions across
                # subdirs (rglob can surface them) don't silently overwrite.
                dest = dedup_dir / fp.relative_to(src_dir)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(fp), str(dest))
            print(f"  [dedup] moved {len(dropped)} frames -> {dedup_dir}")
        elif dropped:
            print(f"  [dedup][DRY RUN] would move {len(dropped)} frames -> {src_dir.name}_dedup/")
        all_imgs = survivors

    # No-Claude-verify mode: keep everything by folder label (only dedup ran). No AWS needed.
    if not claude_verify:
        print(f"\n  {expected_label}: {len(all_imgs)} images "
              f"(Claude verify SKIPPED — kept by folder label; dedup {'on' if dedup else 'off'})")
        return len(all_imgs)

    # Re-verify TIMEOUT/ERROR entries — they weren't actually checked
    RECHECK = {"TIMEOUT", "ERROR"}
    uncached = []
    for p in all_imgs:
        key = str(p.resolve())
        cached = cache.get(key)
        if cached is None or cached.get('claude_answer') in RECHECK:
            uncached.append(p)

    n_recheck = sum(1 for p in all_imgs
                    if cache.get(str(p.resolve()), {}).get('claude_answer') in RECHECK)
    print(f"\n  {expected_label}: {len(all_imgs)} images "
          f"({len(all_imgs)-len(uncached)} cached, {len(uncached)} to verify "
          f"[{n_recheck} recheck TIMEOUT/ERROR])")

    if uncached:
        processed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(check_one, fp, client, model_id): fp
                       for fp in uncached}
            for fut in as_completed(futures):
                fp = futures[fut]
                ans = fut.result()
                fp_key = str(fp.resolve())
                cache[fp_key] = {
                    "has_bear":      ans == "YES",
                    "claude_answer": ans,
                    "species":       expected_label,
                    "dataset":       "screencap",
                    "api_success":   ans not in ("ERROR", "TIMEOUT"),
                    "verified":      ans in ("YES", "NO"),
                    "needs_review":  ans in ("UNSURE", "ERROR", "TIMEOUT"),
                    "gemini":        ans,
                    "folder_label":  1 if expected_label == "bear" else 0,
                }
                processed += 1
                if processed % 50 == 0:
                    # Save cache periodically
                    with open(cache_path, 'wb') as f:
                        pickle.dump(cache, f)
                    print(f"    {processed}/{len(uncached)} verified, cache saved...")

    # Tally decisions
    keep = reject = review = 0
    rejected_dir = src_dir.parent / f"{src_dir.name}_rejected"
    review_dir   = src_dir.parent / f"{src_dir.name}_review"

    for fp in all_imgs:
        fp_key = str(fp.resolve())
        info   = cache.get(fp_key, {})
        ans    = info.get('claude_answer', 'UNSURE')

        if expected_label == 'bear':
            if ans == "YES":
                keep += 1
            elif ans == "NO":
                reject += 1
                if apply:
                    rejected_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(fp), str(rejected_dir / fp.name))
            elif ans == "UNSURE":
                review += 1
                if apply:
                    review_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(fp), str(review_dir / fp.name))
            else:  # ERROR/TIMEOUT — keep in place for recheck next run
                review += 1
        else:
            if ans == "NO":
                keep += 1
            elif ans == "YES":
                reject += 1
                if apply:
                    rejected_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(fp), str(rejected_dir / fp.name))
            elif ans == "UNSURE":
                review += 1
                if apply:
                    review_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(fp), str(review_dir / fp.name))
            else:  # ERROR/TIMEOUT — keep in place for recheck next run
                review += 1

    ans_counts = Counter(cache.get(str(fp.resolve()), {}).get('claude_answer', '?')
                         for fp in all_imgs)
    print(f"  Claude answers: {dict(ans_counts)}")
    print(f"  Keep={keep}, Reject={reject}, Review={review}")
    if not apply:
        print(f"  [DRY RUN] Re-run with --apply to move files")

    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bear-dir",    default=None)
    parser.add_argument("--nonbear-dir", default=None)
    parser.add_argument("--cache",       default="screencap_verify_cache.pkl")
    parser.add_argument("--workers",     type=int, default=5)
    parser.add_argument("--apply",       action="store_true",
                        help="Actually move rejected/review images (default: dry-run)")
    parser.add_argument("--dedup", dest="dedup", action="store_true", default=True,
                        help="Enable phash dedup + dark-frame cap pre-pass (default on)")
    parser.add_argument("--no-dedup", dest="dedup", action="store_false",
                        help="Disable the dedup pre-pass")
    parser.add_argument("--keep-per-cluster", type=int, default=3)
    parser.add_argument("--hamming-thr", type=int, default=6)
    parser.add_argument("--max-black", type=int, default=30)
    parser.add_argument("--no-verify", dest="claude_verify", action="store_false", default=True,
                        help="Skip Claude verification entirely (dedup-only; keep images by "
                             "folder label, no AWS credentials needed)")
    args = parser.parse_args()

    if args.dedup and imagehash is None:
        print("  [dedup] imagehash not installed — dedup disabled. "
              "Install: pip install imagehash")
        args.dedup = False

    model_id = os.environ.get(
        "BEDROCK_MODEL_ID",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )

    print(f"\n{'='*60}")
    print(f"SCREEN-CAPTURE FRAME VERIFICATION")
    print(f"  Mode:        {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"  Bear dir:    {args.bear_dir}")
    print(f"  Non-bear:    {args.nonbear_dir}")
    print(f"  Cache:       {args.cache}")
    print(f"{'='*60}")

    cache_path = Path(args.cache)
    cache = {}
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        print(f"\nLoaded cache: {len(cache)} entries")

    # Only create the Bedrock client (and require AWS creds) when actually verifying.
    client = get_client() if args.claude_verify else None
    if not args.claude_verify:
        print("  [verify] Claude verification DISABLED (--no-verify): dedup-only, "
              "images kept by folder label.")

    if args.bear_dir:
        verify_dir(args.bear_dir, 'bear', cache, cache_path,
                   client, model_id, args.workers, args.apply,
                   dedup=args.dedup, keep_per_cluster=args.keep_per_cluster,
                   hamming_thr=args.hamming_thr, max_black=args.max_black,
                   claude_verify=args.claude_verify)

    if args.nonbear_dir:
        verify_dir(args.nonbear_dir, 'non_bear', cache, cache_path,
                   client, model_id, args.workers, args.apply,
                   dedup=args.dedup, keep_per_cluster=args.keep_per_cluster,
                   hamming_thr=args.hamming_thr, max_black=args.max_black,
                   claude_verify=args.claude_verify)

    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    print(f"\nCache saved: {len(cache)} entries -> {cache_path}")

    if not args.apply:
        print(f"\nRe-run with --apply to move rejected/review images")
    else:
        print(f"\nNext:")
        print(f"  python merge_new_animals.py \\")
        if args.bear_dir:
            print(f"      --bear-dir {args.bear_dir} \\")
        if args.nonbear_dir:
            print(f"      --nonbear-dir {args.nonbear_dir} \\")
        print(f"      --cache {args.cache} lila_verify_cache.pkl inat_verify_cache.pkl")


if __name__ == "__main__":
    main()
