"""
merge_new_animals.py

Merge newly collected confusable animals (elk, moose, raccoon, wild_boar, coyote)
and urban bear into the existing dataset.

Features:
- Perceptual hash dedup vs existing dataset
- Dedup within new data
- Stratified split by species (80/10/10)
- Reads verify cache for species labels and Claude answers
- Leakage check after split

Usage:
    # Dry run first
    python merge_new_animals.py \\
        --dataset bear_dataset_v_final_final \\
        --nonbear-dir non_bear_confusable \\
        --bear-dir bear_urban_collected \\
        --cache lila_verify_cache.pkl inat_verify_cache.pkl \\
        --dry-run

    # Apply
    python merge_new_animals.py \\
        --dataset bear_dataset_v_final_final \\
        --nonbear-dir non_bear_confusable \\
        --bear-dir bear_urban_collected \\
        --cache lila_verify_cache.pkl inat_verify_cache.pkl
"""

import os
import sys
import pickle
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from PIL import Image

PHASH_THRESHOLD = 8   # hamming distance for near-duplicate detection
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
EXTS = {'.jpg', '.jpeg', '.png'}


# ============================================================
# PERCEPTUAL HASH
# ============================================================

def phash_img(img, hash_size=16):
    img = img.convert('L').resize((hash_size, hash_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    mean = arr.mean()
    return ''.join('1' if v >= mean else '0' for v in arr.flatten())


def phash_file(fp):
    try:
        return phash_img(Image.open(fp))
    except Exception:
        return None


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


# ============================================================
# UNION-FIND for clustering
# ============================================================

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def get_clusters(self):
        clusters = defaultdict(list)
        for i in range(len(self.parent)):
            clusters[self.find(i)].append(i)
        return list(clusters.values())


# ============================================================
# LOAD EXISTING DATASET HASHES
# ============================================================

def load_existing_hashes(dataset_root, label, cache_file=None, old_dataset=None):
    """Load phashes for existing train/val/test images of given label.

    Priority:
    1. Per-folder hash_cache.pkl (from merge_and_resplit.py old format)
    2. Combined cache file (phash_cache_<dataset>_<label>.pkl)
    3. Compute from scratch and save to combined cache
    """
    dataset_root = Path(dataset_root)

    # Default combined cache path
    if cache_file is None:
        cache_file = Path(f"phash_cache_{dataset_root.name}_{label}.pkl")

    # Load combined cache if exists
    cached = {}
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        print(f"  Loaded combined phash cache: {len(cached)} entries from {cache_file.name}")

    # Load per-folder hash_cache.pkl from old dataset (e.g. bear_dataset/)
    # Old format: {filename: np.array} — keys are filenames, not full paths
    # We store by filename only so they match images in new dataset too
    old_fname_cache = {}  # {filename: hash_str}
    if old_dataset:
        old_root = Path(old_dataset)
        for split in ['train', 'val', 'test']:
            old_cls_dir = old_root / split / label
            hc_file = old_cls_dir / 'hash_cache.pkl'
            if hc_file.exists():
                with open(hc_file, 'rb') as f:
                    raw = pickle.load(f)
                for fname, h in raw.items():
                    if h is not None:
                        h_str = ''.join(str(int(b)) for b in h.flatten()) \
                            if hasattr(h, 'tolist') else h
                        old_fname_cache[fname] = h_str
        if old_fname_cache:
            print(f"  Loaded {len(old_fname_cache)} hashes from old dataset {old_dataset}")

    hashes = {}
    new_entries = 0

    for split in ['train', 'val', 'test']:
        cls_dir = dataset_root / split / label
        if not cls_dir.exists():
            continue

        fps = [p for p in cls_dir.rglob('*') if p.suffix.lower() in EXTS]

        # Load per-folder hash_cache.pkl from new dataset folder (if exists)
        folder_cache = {}
        hc_file = cls_dir / 'hash_cache.pkl'
        if hc_file.exists():
            with open(hc_file, 'rb') as f:
                raw = pickle.load(f)
            for fname, h in raw.items():
                full_path = str(cls_dir / fname)
                if h is not None:
                    h_str = ''.join(str(int(b)) for b in h.flatten()) \
                        if hasattr(h, 'tolist') else h
                    folder_cache[full_path] = h_str

        n_folder = sum(1 for p in fps if str(p) in folder_cache)
        n_combined = sum(1 for p in fps if str(p) in cached)
        n_old = sum(1 for p in fps if p.name in old_fname_cache)
        print(f"  {split}/{label}: {len(fps)} images "
              f"(combined={n_combined}, folder={n_folder}, old_dataset={n_old})")

        for fp in fps:
            fp_str = str(fp)
            if fp_str in cached:
                hashes[fp_str] = cached[fp_str]
            elif fp_str in folder_cache:
                h = folder_cache[fp_str]
                hashes[fp_str] = h
                cached[fp_str] = h
                new_entries += 1
            elif fp.name in old_fname_cache:
                h = old_fname_cache[fp.name]
                hashes[fp_str] = h
                cached[fp_str] = h
                new_entries += 1
            else:
                h = phash_file(fp)
                if h:
                    hashes[fp_str] = h
                    cached[fp_str] = h
                    new_entries += 1

    # Save updated combined cache
    if new_entries > 0:
        with open(cache_file, 'wb') as f:
            pickle.dump(cached, f)
        print(f"  Saved combined phash cache: {len(cached)} entries "
              f"({new_entries} new) -> {cache_file.name}")

    print(f"  Total existing {label} hashes: {len(hashes)}")
    return hashes


# ============================================================
# SCAN NEW IMAGES
# ============================================================

def scan_new_images(src_dir, label, verify_caches, exclude_species=None):
    """
    Scan source directory for images.
    Returns list of dicts: {fp, species, label, dataset, verified}
    """
    src_dir = Path(src_dir)
    if not src_dir.exists():
        print(f"  [SKIP] {src_dir} not found")
        return []

    exclude_species = set(s.lower() for s in (exclude_species or []))

    # Merge all verify caches
    cache = {}
    for c in verify_caches:
        cache.update(c)

    images = []
    for fp in src_dir.rglob('*'):
        if fp.suffix.lower() not in EXTS:
            continue

        fp_str = str(fp)
        info = cache.get(fp_str, {})

        # For non_bear: skip if Claude says has_bear
        if label == 'non_bear' and info.get('has_bear', False):
            continue
        # For bear: skip if Claude says no bear
        if label == 'bear' and info.get('has_bear') is False and info.get('verified'):
            continue

        species = info.get('species') or fp.parent.name

        # Skip excluded species
        if species.lower() in exclude_species:
            continue

        dataset = info.get('dataset', 'unknown')

        images.append({
            'fp': fp,
            'fp_str': fp_str,
            'species': species,
            'label': label,
            'dataset': dataset,
            'verified': info.get('verified', False),
            'phash': None,
        })

    print(f"  Scanned {src_dir}: {len(images)} {label} images")
    species_count = Counter(img['species'] for img in images)
    for sp, cnt in sorted(species_count.items()):
        print(f"    {sp:<22}: {cnt}")
    return images


# ============================================================
# COMPUTE HASHES FOR NEW IMAGES
# ============================================================

def compute_hashes(images, cache_file=None):
    """Compute phashes for new images, with cache."""
    if cache_file is None:
        cache_file = Path("phash_cache_new_images.pkl")

    cached = {}
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)

    uncached = [img for img in images if img['fp_str'] not in cached]
    print(f"  Computing phashes: {len(uncached)} new, {len(images)-len(uncached)} cached")

    new_entries = 0
    failed = 0
    for i, img in enumerate(uncached):
        h = phash_file(img['fp'])
        if h:
            cached[img['fp_str']] = h
            new_entries += 1
        else:
            failed += 1
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(uncached)}")
            with open(cache_file, 'wb') as f:
                pickle.dump(cached, f)

    if new_entries > 0:
        with open(cache_file, 'wb') as f:
            pickle.dump(cached, f)

    if failed:
        print(f"  Warning: {failed} images failed to hash (excluded)")

    for img in images:
        img['phash'] = cached.get(img['fp_str'])
    images[:] = [img for img in images if img['phash']]
    return images


# ============================================================
# DEDUP NEW VS EXISTING
# ============================================================


# ============================================================
# DCT CONFIRMATION (see memory: coarse-phash-unreliable)
# ============================================================
# The 16x16 mean-threshold hash above is a CANDIDATE GENERATOR, not a verdict.
# Measured on 300 cross-dataset pairs it called similar, 31.7% were not similar
# at all and 3 of 31 "perfect" distance-0 matches were visually unrelated -- a
# cougar on a trail matched a brown bear with a cub. Two mechanisms: letterboxed
# footage is mostly black bars, and nearly every outdoor photo is bright on top
# and dark below, so a 16x16 thumbnail encodes layout rather than content.
#
# It is fine within one homogeneous batch (2.5% error on the 26/08 screen-capture
# dedup) and collapses across diverse pools -- which is exactly what a merge is.
# So every coarse hit is confirmed the way clean_leakage_v2.py confirms it:
# 64-bit DCT phash <= 10 AND 64x64 grayscale RMSE <= 25.

DCT_THR = 10
RMSE_THR = 25.0
# Two caches, same format ((hex DCT phash, 64x64 uint8 thumbnail), keyed by
# path relative to the data root). `phash_dct_cache.pkl` is clean_leakage_v2's
# and already holds the dataset side; it is read but never written, so a merge
# cannot corrupt what the leakage tooling depends on. New work accumulates in
# the merge's own file.
_DATA_ROOT = Path("/data/nas07_new/PersonalData/phuocthien")
_dct_cache_path = Path("phash_dct_cache_merge.pkl")
_dct_shared_path = Path("phash_dct_cache.pkl")
_dct_cache = {}
_dct_shared_keys = set()
_dct_loaded = False


def _dct_key(path):
    p = Path(path)
    try:
        return str(p.relative_to(_DATA_ROOT))
    except ValueError:
        return str(p)


def _dct_load():
    global _dct_cache, _dct_loaded
    if _dct_loaded:
        return
    _dct_loaded = True
    for src in (_dct_shared_path, _dct_cache_path):
        if src.exists():
            try:
                with open(src, "rb") as f:
                    d = pickle.load(f)
                _dct_cache.update(d)
                if src is _dct_shared_path:
                    _dct_shared_keys.update(d)
            except Exception:
                pass
    print(f"  DCT cache: {len(_dct_cache)} entries loaded")


def _dct_save():
    """Persist only what this run computed; the shared cache stays as it was."""
    try:
        own = {k: v for k, v in _dct_cache.items() if k not in _dct_shared_keys}
        with open(_dct_cache_path, "wb") as f:
            pickle.dump(own, f)
    except Exception:
        pass


def _dct_entry(path):
    """(hex DCT phash, 64x64 grayscale array) for one image, cached on disk."""
    key = _dct_key(path)
    if key in _dct_cache:
        return _dct_cache[key]
    try:
        import imagehash
        with Image.open(path) as im:
            im = im.convert("L")
            ph = str(imagehash.phash(im))
            th = np.asarray(im.resize((64, 64), Image.LANCZOS), dtype=np.uint8)
        _dct_cache[key] = (ph, th)
    except Exception:
        _dct_cache[key] = None
    return _dct_cache[key]


def confirm_duplicate(a_path, b_path):
    """True only if the two really are near-duplicates."""
    import imagehash
    ea, eb = _dct_entry(a_path), _dct_entry(b_path)
    if ea is None or eb is None:
        return False          # unreadable: keep the image rather than guess
    d = imagehash.hex_to_hash(ea[0]) - imagehash.hex_to_hash(eb[0])
    if d > DCT_THR:
        return False
    rmse = float(np.sqrt(((ea[1].astype(np.float32) - eb[1].astype(np.float32)) ** 2).mean()))
    return rmse <= RMSE_THR


def dedup_vs_existing(new_images, existing_hashes, threshold=PHASH_THRESHOLD):
    """Remove new images that are too similar to existing dataset images.

    LSH-banded: build a band index over the existing hashes, then only
    hamming-compare each new image against existing hashes that share at least
    one band. For 256-bit hashes with N_BANDS=16 (16-bit bands) this is
    EXHAUSTIVE at threshold<=15 by pigeonhole (<=T differing bits touch <=T of
    the 16 bands, so >=16-T bands match exactly) -> identical results to the old
    O(n*m) nested loop but ~100x faster. Falls back to a naive scan if the
    threshold is too large for banding to stay provably exhaustive.
    """
    existing_hash_list = list(existing_hashes.values())
    removed = []
    kept = []

    print(f"  Dedup new vs existing ({len(new_images)} new, {len(existing_hashes)} existing)...")

    if not existing_hash_list:
        print(f"  Removed 0 duplicates vs existing, kept {len(new_images)}")
        return list(new_images)

    N_BANDS = 16
    hash_len = len(existing_hash_list[0])
    exhaustive = threshold <= (N_BANDS - 1) and hash_len % N_BANDS == 0

    if not exhaustive:
        for img in new_images:
            hit = next((ep for ep, eh in existing_hashes.items()
                        if hamming(img['phash'], eh) <= threshold), None)
            if hit and confirm_duplicate(img['fp'], hit):
                removed.append(img)
            else:
                kept.append(img)
        print(f"  Removed {len(removed)} duplicates vs existing, kept {len(kept)} (naive)")
        return kept

    # Bands are contiguous slices of the phash string, but trail-cam photos
    # correlate strongly by position (bright sky rows vs dark ground rows
    # repeat across most images), so a contiguous 16-bit slice is nowhere
    # near uniformly distributed -- one band value alone was shared by
    # 42613/127478 existing hashes, collapsing the "banding" back into a
    # near-naive O(n*m) scan (observed: 76+ min and still running on a
    # 7111-vs-127478 dedup that should take seconds). A fixed random
    # permutation of the bit positions breaks that spatial correlation before
    # slicing -- it doesn't change any Hamming distance (same multiset of
    # bits, just reordered identically for every hash), so the threshold<=15
    # exhaustiveness guarantee is untouched, but bucket sizes go from
    # tens-of-thousands back down to near hash_len/N_BANDS on average.
    band_len = hash_len // N_BANDS
    rng = random.Random(42)
    perm = list(range(hash_len))
    rng.shuffle(perm)

    def permuted(h):
        return ''.join(h[i] for i in perm)

    from collections import defaultdict
    buckets = defaultdict(list)
    for eh in existing_hash_list:
        peh = permuted(eh)
        for b in range(N_BANDS):
            buckets[(b, peh[b * band_len:(b + 1) * band_len])].append(eh)

    hash_to_path = {}
    for ep, eh in existing_hashes.items():
        hash_to_path.setdefault(eh, ep)

    _dct_load()
    coarse_hits = 0
    for img in new_images:
        h = img['phash']
        ph = permuted(h)
        seen = set()
        is_dup = False
        for b in range(N_BANDS):
            for eh in buckets.get((b, ph[b * band_len:(b + 1) * band_len]), ()):
                if eh in seen:
                    continue
                seen.add(eh)
                if hamming(h, eh) <= threshold:
                    coarse_hits += 1
                    if confirm_duplicate(img['fp'], hash_to_path[eh]):
                        is_dup = True
                        break
            if is_dup:
                break
        (removed if is_dup else kept).append(img)
    _dct_save()

    print(f"  Removed {len(removed)} duplicates vs existing, kept {len(kept)}"
          f"   [{coarse_hits} coarse hits, {len(removed)} confirmed by DCT+RMSE]")
    return kept


# ============================================================
# DEDUP WITHIN NEW
# ============================================================

def dedup_within_new(new_images, threshold=PHASH_THRESHOLD):
    """Remove near-duplicates within new images, keep one per cluster."""
    if not new_images:
        return new_images

    n = len(new_images)
    uf = UnionFind(n)

    print(f"  Dedup within new ({n} images)...")
    hashes = [img['phash'] for img in new_images]

    # Sliding window comparison (sorted by hash for efficiency)
    order = sorted(range(n), key=lambda i: hashes[i])
    window = 500

    # Same rule as the cross-dataset pass: the coarse hash proposes, DCT+RMSE
    # disposes. Cheaper here than it looks, because a union already made makes
    # the pair irrelevant and the DCT entries are cached across both passes.
    _dct_load()
    for i in range(len(order)):
        for j in range(i + 1, min(i + window, len(order))):
            a, b = order[i], order[j]
            if uf.find(a) == uf.find(b):
                continue
            if hamming(hashes[a], hashes[b]) <= threshold and \
                    confirm_duplicate(new_images[a]['fp'], new_images[b]['fp']):
                uf.union(a, b)
    _dct_save()

    clusters = uf.get_clusters()
    kept_indices = set()
    for cluster in clusters:
        # Keep one representative per cluster (first alphabetically)
        best = min(cluster, key=lambda i: str(new_images[i]['fp']))
        kept_indices.add(best)

    kept = [img for i, img in enumerate(new_images) if i in kept_indices]
    print(f"  Reduced {n} -> {len(kept)} after within-new dedup "
          f"({n - len(kept)} duplicates removed)")
    return kept


# ============================================================
# STRATIFIED SPLIT BY SPECIES
# ============================================================

def stratified_split(images, train_r=TRAIN_RATIO, val_r=VAL_RATIO, test_r=TEST_RATIO, seed=42):
    """Split images stratified by species."""
    random.seed(seed)

    # Group by species
    by_species = defaultdict(list)
    for img in images:
        by_species[img['species']].append(img)

    train, val, test = [], [], []

    print(f"\n  Stratified split by species:")
    for species, imgs in sorted(by_species.items()):
        random.shuffle(imgs)
        n = len(imgs)
        n_val  = max(1, int(n * val_r))
        n_test = max(1, int(n * test_r))
        n_train = n - n_val - n_test

        train.extend(imgs[:n_train])
        val.extend(imgs[n_train:n_train + n_val])
        test.extend(imgs[n_train + n_val:])

        print(f"    {species:<22}: total={n:>4} | "
              f"train={n_train:>4}, val={n_val:>3}, test={n_test:>3}")

    return train, val, test


# ============================================================
# LEAKAGE CHECK
# ============================================================

def check_leakage(train, val, test, threshold=PHASH_THRESHOLD):
    """Move genuine cross-split near-duplicates into train.

    Two changes from the original, both forced by measurement:

    * **A coarse hit is not a leak.** On this merge the 16x16 hash flagged
      2,931 cross-dataset pairs of which DCT+RMSE confirmed 354 -- an 88%
      false-positive rate, far worse than the 31.7% measured on an earlier,
      less uniform pool. Moving on the coarse verdict alone would shovel most
      of val and test into train and leave nothing to evaluate on. Note that
      `dedup_within_new` has already removed every confirmed duplicate among
      these images, so a genuine leak here should be rare by construction.

    * **Speed.** The original compared every val image against every train
      hash with a per-character Python loop: 1,980 x 7,921 pairs of 256-char
      strings, which had not finished after two hours. Hashes become ints
      once and the distance is a XOR plus `bit_count`.
    """
    train_ints = [int(img['phash'], 2) for img in train]
    leaks_val = leaks_test = 0
    coarse = 0
    _dct_load()

    def leak_of(img):
        nonlocal coarse
        h = int(img['phash'], 2)
        for ti, timg in zip(train_ints, train):
            if (h ^ ti).bit_count() <= threshold:
                coarse += 1
                if confirm_duplicate(img['fp'], timg['fp']):
                    return True
        return False

    clean_val = []
    for img in val:
        if leak_of(img):
            train.append(img)
            train_ints.append(int(img['phash'], 2))
            leaks_val += 1
        else:
            clean_val.append(img)

    clean_test = []
    for img in test:
        if leak_of(img):
            train.append(img)
            train_ints.append(int(img['phash'], 2))
            leaks_test += 1
        else:
            clean_test.append(img)
    _dct_save()

    if leaks_val or leaks_test:
        print(f"  Leakage: moved {leaks_val} val + {leaks_test} test -> train"
              f"   [{coarse} coarse hits]")
    else:
        print(f"  No leakage detected   [{coarse} coarse hits, none confirmed]")

    return train, clean_val, clean_test


# ============================================================
# COPY FILES TO DATASET
# ============================================================

def copy_to_dataset(images, split, label, dataset_root, dry_run=False, manifest=None):
    """Copy images to dataset split/label folder. Logs added paths to manifest."""
    dst_dir = Path(dataset_root) / split / label
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    copied = skipped = 0
    for img in images:
        species = img['species']
        src = img['fp']
        dst_name = f"{species}_{src.name}"
        dst = dst_dir / dst_name

        # Handle name collision
        if dst.exists():
            dst = dst_dir / f"{species}_{src.stem}_dup{src.suffix}"

        if dry_run:
            pass
        else:
            shutil.copy2(str(src), str(dst))
            if manifest is not None:
                manifest.append(str(dst))
        copied += 1

    return copied


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None,
                        help="Path to existing dataset root")
    parser.add_argument("--nonbear-dir", nargs="+", default=[],
                        help="Directories with new non_bear images (supports multiple)")
    parser.add_argument("--bear-dir", nargs="+", default=[],
                        help="Directories with new bear images (supports multiple)")
    parser.add_argument("--collected-bears", nargs="+", default=[],
                        help="collected_bears root dirs — auto-scans */bear_frames/ subdirs")
    parser.add_argument("--exclude-species", nargs="+", default=[],
                        help="Species to exclude (e.g. asian_elephant)")
    parser.add_argument("--cache", nargs="+", default=[],
                        help="Verify cache pkl files")
    parser.add_argument("--phash-threshold", type=int, default=PHASH_THRESHOLD)
    parser.add_argument("--old-dataset", default=None,
                        help="Original dataset with hash_cache.pkl files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without copying files")
    parser.add_argument("--reverse", default=None,
                        help="Reverse a previous merge using its manifest JSON file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.reverse and not args.dataset:
        parser.error("--dataset is required unless --reverse is given")
    # ── Reverse mode ────────────────────────────────────────────
    if args.reverse:
        import json
        manifest_path = Path(args.reverse)
        if not manifest_path.exists():
            print(f"Manifest not found: {manifest_path}")
            return
        with open(manifest_path) as f:
            data = json.load(f)
        files = data.get("added_files", [])
        print(f"\nReversing merge from {manifest_path}")
        print(f"  Timestamp: {data.get('timestamp')}")
        print(f"  Files to remove: {len(files)}")

        removed = missing = 0
        for fp in files:
            p = Path(fp)
            if p.exists():
                p.unlink()
                removed += 1
            else:
                missing += 1
        print(f"  Removed: {removed}, Already missing: {missing}")
        print(f"  Dataset restored to pre-merge state.")
        print(f"  Note: re-run check_data_leakage.py to verify.")
        return

    print(f"\n{'='*60}")
    print(f"MERGE NEW ANIMALS INTO DATASET")
    print(f"  Dataset:          {args.dataset}")
    print(f"  Non-bear dirs:    {args.nonbear_dir}")
    print(f"  Bear dirs:        {args.bear_dir}")
    print(f"  Collected bears:  {args.collected_bears}")
    print(f"  Exclude species:  {args.exclude_species}")
    print(f"  Caches:           {args.cache}")
    print(f"  Phash thr:        {args.phash_threshold}")
    print(f"  Dry run:          {args.dry_run}")
    print(f"{'='*60}\n")

    # Load verify caches
    # Auto-discover gemini_cache.pkl from collected_bears subdirs
    cache_files = list(args.cache)
    for cb_root in args.collected_bears:
        for gc in Path(cb_root).glob("*/bear_frames/gemini_cache.pkl"):
            cache_files.append(str(gc))

    verify_caches = []
    for cache_file in cache_files:
        cp = Path(cache_file)
        if cp.exists():
            with open(cp, 'rb') as f:
                verify_caches.append(pickle.load(f))
            print(f"Loaded cache: {cache_file} ({len(verify_caches[-1])} entries)")
        else:
            print(f"Warning: cache not found: {cache_file}")

    exclude = args.exclude_species
    all_new_nonbear = []
    all_new_bear = []

    # ── Non-bear ────────────────────────────────────────────────
    nonbear_dirs = list(args.nonbear_dir)
    if nonbear_dirs:
        print(f"\n--- Scanning non_bear ---")
        existing_nb_hashes = load_existing_hashes(
            args.dataset, 'non_bear', old_dataset=args.old_dataset)

        for src in nonbear_dirs:
            print(f"  Source: {src}")
            new_nb = scan_new_images(src, 'non_bear', verify_caches,
                                     exclude_species=exclude)
            new_nb = compute_hashes(new_nb)
            new_nb = dedup_vs_existing(new_nb, existing_nb_hashes, args.phash_threshold)
            new_nb = dedup_within_new(new_nb, args.phash_threshold)
            all_new_nonbear.extend(new_nb)
            # Add new hashes to existing so subsequent dirs dedup against them too
            for img in new_nb:
                if img['phash']:
                    existing_nb_hashes[img['fp_str']] = img['phash']

    # ── Bear ────────────────────────────────────────────────────
    bear_dirs = list(args.bear_dir)

    # Auto-expand collected_bears dirs → scenario/bear_frames/
    for cb_root in args.collected_bears:
        cb_root = Path(cb_root)
        for bear_frames in sorted(cb_root.glob("*/bear_frames")):
            if bear_frames.is_dir():
                bear_dirs.append(str(bear_frames))
                print(f"  Found bear_frames: {bear_frames}")

    if bear_dirs:
        print(f"\n--- Scanning bear ---")
        existing_bear_hashes = load_existing_hashes(
            args.dataset, 'bear', old_dataset=args.old_dataset)

        for src in bear_dirs:
            print(f"  Source: {src}")
            new_bear = scan_new_images(src, 'bear', verify_caches,
                                       exclude_species=exclude)
            new_bear = compute_hashes(new_bear)
            new_bear = dedup_vs_existing(new_bear, existing_bear_hashes, args.phash_threshold)
            new_bear = dedup_within_new(new_bear, args.phash_threshold)
            all_new_bear.extend(new_bear)
            for img in new_bear:
                if img['phash']:
                    existing_bear_hashes[img['fp_str']] = img['phash']

    if not all_new_nonbear and not all_new_bear:
        print("\nNo new images to merge. Exiting.")
        return

    # ── Split ───────────────────────────────────────────────────
    total_new = len(all_new_nonbear) + len(all_new_bear)
    print(f"\n--- Splitting {total_new} new images ---")

    nb_train = nb_val = nb_test = []
    if all_new_nonbear:
        nb_train, nb_val, nb_test = stratified_split(
            all_new_nonbear, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, args.seed)
        nb_train, nb_val, nb_test = check_leakage(nb_train, nb_val, nb_test,
                                                   args.phash_threshold)

    bear_train = bear_val = bear_test = []
    if all_new_bear:
        bear_train, bear_val, bear_test = stratified_split(
            all_new_bear, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, args.seed)
        bear_train, bear_val, bear_test = check_leakage(bear_train, bear_val, bear_test,
                                                         args.phash_threshold)

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SPLIT SUMMARY {'(DRY RUN)' if args.dry_run else ''}")
    print(f"{'='*60}")
    if all_new_nonbear:
        print(f"  non_bear: train={len(nb_train)}, val={len(nb_val)}, test={len(nb_test)}")
    if all_new_bear:
        print(f"  bear:     train={len(bear_train)}, val={len(bear_val)}, test={len(bear_test)}")

    if args.dry_run:
        print(f"\n[DRY RUN] Re-run without --dry-run to copy files.")
        return

    # ── Copy ────────────────────────────────────────────────────
    print(f"\n--- Copying to {args.dataset} ---")
    total_copied = 0
    manifest = []   # track all added file paths for reversibility

    for split, imgs, label in [
        ('train', nb_train,    'non_bear'),
        ('val',   nb_val,      'non_bear'),
        ('test',  nb_test,     'non_bear'),
        ('train', bear_train,  'bear'),
        ('val',   bear_val,    'bear'),
        ('test',  bear_test,   'bear'),
    ]:
        if not imgs:
            continue
        n = copy_to_dataset(imgs, split, label, args.dataset,
                            dry_run=args.dry_run, manifest=manifest)
        print(f"  {split}/{label}: {n} images")
        total_copied += n

    # Write manifest for reverse
    import json, datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = Path(args.dataset).parent / f"merge_manifest_{ts}.json"
    with open(manifest_path, 'w') as f:
        json.dump({
            "timestamp": ts,
            "dataset": str(args.dataset),
            "total_added": len(manifest),
            "added_files": manifest,
        }, f, indent=2)

    print(f"\n  Total copied: {total_copied} images")
    print(f"  Manifest saved: {manifest_path}")
    print(f"\n{'='*60}")
    print(f"Done! Next steps:")
    print(f"  1. Verify dataset: python check_data_leakage.py --dataset {args.dataset}")
    print(f"  2. Retrain from scratch:")
    print(f"     python train_mobilenetv2_relu_novaic.py --data-dir {args.dataset}")
    print(f"\n  To REVERSE this merge (if results don't improve):")
    print(f"     python merge_new_animals.py --reverse {manifest_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()