"""
export_for_screencap.py

Copy a subset of bear and non_bear images from the server dataset
to a local folder for camera capture testing.

Uses CLIP to classify each image into scenarios, then samples evenly
across scenarios for balanced representation.

Scenarios detected:
  Bear:     primary, night_vision, fog_rain, snow, distance, occlusion, motion_blur, urban
  Non-bear: primary, IR, new_animal, camera_frame, fog_rain, snow, urban, motion_blur

Usage:
    python export_for_screencap.py \\
        --dataset bear_dataset_v_final_final \\
        --output screencap_export \\
        --n-bear 600 \\
        --n-nonbear 400 \\
        --gpu 0

    # Then zip and download
    zip -r screencap_export.zip screencap_export/
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from PIL import Image

EXTS = {'.jpg', '.jpeg', '.png'}

BEAR_SCENARIOS = [
    "primary",
    "night_vision",
    "fog_rain",
    "snow_conditions",
    "distance_far",
    "partial_occlusion",
    "motion_blur",
    "urban",
]

NONBEAR_SCENARIOS = [
    "primary",
    "IR_night",
    "new_animal",
    "camera_frame",
    "fog_rain",
    "snow_conditions",
    "urban",
    "motion_blur",
]

BEAR_PROMPTS = {
    "primary":           "a clear daytime trail camera photo of a bear in forest",
    "night_vision":      "a dark night vision infrared grayscale trail camera bear photo",
    "fog_rain":          "a foggy rainy misty trail camera photo of a bear",
    "snow_conditions":   "a snowy winter trail camera photo of a bear",
    "distance_far":      "a distant small bear far away in a trail camera photo",
    "partial_occlusion": "a bear partially hidden behind trees bushes in trail camera",
    "motion_blur":       "a blurry motion trail camera photo of a bear running",
    "urban":             "a bear near buildings road dumpster security camera footage",
}

NONBEAR_PROMPTS = {
    "primary":       "a clear daytime trail camera photo without any bear animal",
    "IR_night":      "a dark infrared night vision trail camera without bear",
    "new_animal":    "a deer elk coyote raccoon animal trail camera photo no bear",
    "camera_frame":  "a trail camera screen recording photo on monitor no bear",
    "fog_rain":      "a foggy rainy trail camera photo without bear",
    "snow_conditions": "a snowy winter trail camera photo without bear",
    "urban":         "urban city street security camera footage no bear",
    "motion_blur":   "a blurry motion trail camera photo no bear",
}


def load_clip(gpu_id=0):
    try:
        import clip
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "git+https://github.com/openai/CLIP.git",
                        "--break-system-packages", "-q"])
        import clip

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"  Loading CLIP on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess, device


def classify_scenarios(images, prompts, model, preprocess, device, batch_size=128):
    """Classify each image into one of the scenarios using CLIP."""
    import clip
    scenario_names = list(prompts.keys())
    prompt_texts   = list(prompts.values())

    with torch.no_grad():
        text_tokens  = clip.tokenize(prompt_texts).to(device)
        text_feats   = model.encode_text(text_tokens).float()
        text_feats  /= text_feats.norm(dim=-1, keepdim=True)

    assignments = {}  # fp -> scenario_name
    total = len(images)

    for i in range(0, total, batch_size):
        batch = images[i:i+batch_size]
        imgs, valid = [], []
        for fp in batch:
            try:
                imgs.append(preprocess(Image.open(fp).convert("RGB")))
                valid.append(fp)
            except Exception:
                assignments[fp] = scenario_names[0]  # fallback to primary

        if not imgs:
            continue

        with torch.no_grad():
            img_tensor = torch.stack(imgs).to(device)
            img_feats  = model.encode_image(img_tensor).float()
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
            sims = (img_feats @ text_feats.T)
            preds = sims.argmax(dim=1).cpu().numpy()

        for fp, pred in zip(valid, preds):
            assignments[fp] = scenario_names[int(pred)]

        if (i // batch_size) % 5 == 0:
            print(f"    {min(i+batch_size, total)}/{total} classified...")

    return assignments


def stratified_sample(assignments, n):
    """Sample n images evenly across scenarios."""
    by_scenario = defaultdict(list)
    for fp, sc in assignments.items():
        by_scenario[sc].append(fp)

    scenarios = list(by_scenario.keys())
    n_scenarios = len(scenarios)
    per_scenario = max(1, n // n_scenarios)
    remainder    = n - per_scenario * n_scenarios

    selected = []
    print(f"\n  Scenario distribution (targeting {n} images):")
    for i, sc in enumerate(sorted(scenarios)):
        imgs = by_scenario[sc]
        random.shuffle(imgs)
        quota = per_scenario + (1 if i < remainder else 0)
        take  = imgs[:quota]
        selected.extend(take)
        print(f"    {sc:<22}: {len(imgs):>5} available -> {len(take)} selected")

    random.shuffle(selected)
    return selected[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   required=True)
    parser.add_argument("--output",    default="screencap_export")
    parser.add_argument("--n-bear",    type=int, default=600)
    parser.add_argument("--n-nonbear", type=int, default=400)
    parser.add_argument("--gpu",       type=int, default=0)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--zip",       action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    dataset = Path(args.dataset)
    output  = Path(args.output)
    (output / "bear").mkdir(parents=True, exist_ok=True)
    (output / "non_bear").mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"EXPORT FOR SCREEN CAPTURE (CLIP stratified)")
    print(f"  Dataset:   {dataset}")
    print(f"  Output:    {output}")
    print(f"  Bear:      {args.n_bear} images (stratified by scenario)")
    print(f"  Non-bear:  {args.n_nonbear} images (stratified by scenario)")
    print(f"{'='*60}\n")

    model, preprocess, device = load_clip(args.gpu)

    for label, n, prompts, src_name in [
        ("bear",     args.n_bear,    BEAR_PROMPTS,    "bear"),
        ("non_bear", args.n_nonbear, NONBEAR_PROMPTS, "non_bear"),
    ]:
        print(f"\n--- {label} ---")
        src_dir = dataset / "train" / src_name
        all_imgs = []
        for ext in EXTS:
            all_imgs.extend(src_dir.rglob(f'*{ext}'))

        print(f"  Found {len(all_imgs)} images")

        # Classify all images for best scenario coverage
        clip_pool = all_imgs
        if len(all_imgs) > 10000:
            print(f"  Note: classifying all {len(all_imgs)} images (~{len(all_imgs)//2500} min on GPU)")

        print(f"  Classifying {len(clip_pool)} images into scenarios...")
        assignments = classify_scenarios(clip_pool, prompts, model, preprocess,
                                         device, batch_size=128)

        selected = stratified_sample(assignments, n)

        dst_dir = output / label
        copied = 0
        for fp in selected:
            dst = dst_dir / fp.name
            if not dst.exists():
                shutil.copy2(str(fp), str(dst))
            copied += 1

        print(f"  Copied {copied} images -> {dst_dir}")

    # Summary
    total = sum(1 for f in output.rglob('*') if f.is_file())
    size_mb = sum(f.stat().st_size for f in output.rglob('*') if f.is_file()) / 1e6
    print(f"\n{'='*60}")
    print(f"  Total: {total} images ({size_mb:.0f} MB)")
    print(f"{'='*60}")

    if args.zip:
        zip_path = str(output) + ".zip"
        print(f"\n  Creating {zip_path}...")
        shutil.make_archive(str(output), 'zip', str(output.parent), str(output.name))
        zip_size = Path(zip_path).stat().st_size / 1e6
        print(f"  Done: {zip_path} ({zip_size:.0f} MB)")
        print(f"  Download: scp user@server:{Path(zip_path).absolute()} ./")
    else:
        print(f"\n  Next:")
        print(f"  zip -r {output}.zip {output}/")
        print(f"  scp user@server:{output.absolute()}.zip ./")
        print(f"\n  On laptop:")
        print(f"  python capture_screen_data.py --source {output.name}/bear --display-only --n {args.n_bear} --delay 3")
        print(f"  python capture_screen_data.py --source {output.name}/non_bear --display-only --n {args.n_nonbear} --delay 3")


if __name__ == "__main__":
    main()


import os
import shutil
import random
import argparse
from pathlib import Path


PRIORITY_KEYWORDS = ['night', 'ir', 'dark', 'distance', 'far', 'snow', 'fog']
EXTS = {'.jpg', '.jpeg', '.png'}


def priority_score(fp):
    name = fp.name.lower()
    parent = fp.parent.name.lower()
    combined = name + parent
    return sum(1 for kw in PRIORITY_KEYWORDS if kw in combined)


def collect_images(src_dir, n, prioritize=True):
    """Collect n images from src_dir, prioritizing by keyword."""
    src_dir = Path(src_dir)
    if not src_dir.exists():
        print(f"  [SKIP] {src_dir} not found")
        return []

    all_imgs = []
    for split in ['train', 'val', 'test']:
        d = src_dir / split if (src_dir / split).exists() else src_dir
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            all_imgs.extend(d.rglob(ext) if split == 'train' else [])
        if d == src_dir:
            break

    # If src_dir is already a class folder (bear/ or non_bear/)
    if not all_imgs:
        for ext in EXTS:
            all_imgs.extend(src_dir.rglob(f'*{ext}'))

    if not all_imgs:
        print(f"  No images found in {src_dir}")
        return []

    print(f"  Found {len(all_imgs)} images in {src_dir}")

    if prioritize:
        random.shuffle(all_imgs)
        all_imgs.sort(key=priority_score, reverse=True)
        n_priority = sum(1 for fp in all_imgs[:n] if priority_score(fp) > 0)
        print(f"  Selecting {n} ({n_priority} match priority keywords)")
    else:
        random.shuffle(all_imgs)

    return all_imgs[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Dataset root (e.g. bear_dataset_v_final_final)")
    parser.add_argument("--output", default="screencap_export",
                        help="Output folder (default: screencap_export)")
    parser.add_argument("--n-bear", type=int, default=600,
                        help="Number of bear images (default: 600)")
    parser.add_argument("--n-nonbear", type=int, default=400,
                        help="Number of non_bear images (default: 400)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zip", action="store_true",
                        help="Create zip file after export")
    args = parser.parse_args()

    random.seed(args.seed)
    dataset = Path(args.dataset)
    output  = Path(args.output)

    bear_out    = output / "bear"
    nonbear_out = output / "non_bear"
    bear_out.mkdir(parents=True, exist_ok=True)
    nonbear_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"EXPORT FOR SCREEN CAPTURE")
    print(f"  Dataset:   {dataset}")
    print(f"  Output:    {output}")
    print(f"  Bear:      {args.n_bear} images (priority: night/IR/distance)")
    print(f"  Non-bear:  {args.n_nonbear} images")
    print(f"{'='*60}\n")

    # ── Bear ────────────────────────────────────────────────
    print("--- Bear ---")
    bear_src = dataset / "train" / "bear"
    bear_imgs = collect_images(bear_src, args.n_bear, prioritize=True)

    copied_bear = 0
    for fp in bear_imgs:
        dst = bear_out / fp.name
        if not dst.exists():
            shutil.copy2(str(fp), str(dst))
        copied_bear += 1

    print(f"  Copied {copied_bear} bear images -> {bear_out}")

    # ── Non-bear ─────────────────────────────────────────────
    print("\n--- Non-bear ---")
    nonbear_src = dataset / "train" / "non_bear"
    nonbear_imgs = collect_images(nonbear_src, args.n_nonbear, prioritize=False)

    copied_nonbear = 0
    for fp in nonbear_imgs:
        dst = nonbear_out / fp.name
        if not dst.exists():
            shutil.copy2(str(fp), str(dst))
        copied_nonbear += 1

    print(f"  Copied {copied_nonbear} non_bear images -> {nonbear_out}")

    # ── Summary ─────────────────────────────────────────────
    total = copied_bear + copied_nonbear
    size_mb = sum(f.stat().st_size for f in output.rglob('*') if f.is_file()) / 1e6

    print(f"\n{'='*60}")
    print(f"  Total: {total} images ({size_mb:.0f} MB)")
    print(f"  Output: {output}/")
    print(f"{'='*60}")

    if args.zip:
        zip_path = str(output) + ".zip"
        print(f"\n  Creating {zip_path}...")
        shutil.make_archive(str(output), 'zip', str(output.parent), str(output.name))
        zip_size = Path(zip_path).stat().st_size / 1e6
        print(f"  Done: {zip_path} ({zip_size:.0f} MB)")
        print(f"\n  Download: scp user@server:{Path(zip_path).absolute()} ./")
    else:
        print(f"\n  Next steps:")
        print(f"  # Zip and download:")
        print(f"  zip -r {output}.zip {output}/")
        print(f"  scp user@server:{output.absolute()}.zip ./")
        print(f"\n  # Or rsync:")
        print(f"  rsync -av user@server:{output.absolute()}/ ./{output.name}/")
        print(f"\n  # Then on laptop:")
        print(f"  python capture_screen_data.py --source {output.name}/bear --display-only --n {copied_bear} --delay 3")
        print(f"  python capture_screen_data.py --source {output.name}/non_bear --display-only --n {copied_nonbear} --delay 3")


if __name__ == "__main__":
    main()
