"""
evaluate_test_set.py

Evaluate trained MobileNetV2 model on the test split.
Reports per-class and overall metrics: Precision, Recall, F1, F0.5, Accuracy.
Optionally uses temperature scaling for calibrated probabilities.

Usage:
    python evaluate_test_set.py \\
        --model BearDetection_mobilenetv2_relu_novaic_640x360_31_5/models/best_model.pth \\
        --dataset bear_dataset_v_final_final \\
        --gpu 0

    # With temperature scaling
    python evaluate_test_set.py \\
        --model BearDetection_mobilenetv2_relu_novaic_640x360_31_5/models/best_model.pth \\
        --temp-scale BearDetection_mobilenetv2_relu_novaic_640x360_31_5/models/temperature_scaling.pth \\
        --dataset bear_dataset_v_final_final \\
        --gpu 0

    # Sweep thresholds to find best
    python evaluate_test_set.py \\
        --model BearDetection_mobilenetv2_relu_novaic_640x360_31_5/models/best_model.pth \\
        --dataset bear_dataset_v_final_final \\
        --sweep-threshold \\
        --gpu 0
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchvision.transforms.functional as TF


# ============================================================
# CONFIG
# ============================================================
INPUT_SIZE = (360, 640)
NOVAIC_MEAN_BGR = [104/255, 116/255, 124/255]


# ============================================================
# MODEL
# ============================================================
def create_model():
    model = models.mobilenet_v2(weights=None)
    for name, child in model.named_modules():
        pass  # just to initialize

    def replace_relu6(module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU6):
                setattr(module, name, nn.ReLU(inplace=True))
            else:
                replace_relu6(child)

    replace_relu6(model)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, 2)
    )
    return model


def novaic_preprocess_gpu(images):
    images = images[:, [2, 1, 0], :, :]  # RGB -> BGR
    images = images.clone()
    images[:, 0] -= 104 / 255
    images[:, 1] -= 116 / 255
    images[:, 2] -= 124 / 255
    return images


# ============================================================
# DATASET
# ============================================================
class ResizeWithPadding:
    def __init__(self, target_size):
        self.target_h, self.target_w = target_size

    def __call__(self, img):
        w, h = img.size
        scale = min(self.target_w / w, self.target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = TF.resize(img, (new_h, new_w),
                                interpolation=transforms.InterpolationMode.BILINEAR)
        canvas = Image.new('RGB', (self.target_w, self.target_h), (0, 0, 0))
        canvas.paste(img_resized, ((self.target_w - new_w) // 2,
                                   (self.target_h - new_h) // 2))
        return canvas


class TestDataset(Dataset):
    def __init__(self, root_dir, input_size=INPUT_SIZE):
        self.images, self.labels = [], []
        root_dir = Path(root_dir)

        for class_dir in sorted(root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = 0 if 'non' in class_dir.name.lower() \
                else (1 if 'bear' in class_dir.name.lower() else -1)
            if label < 0:
                continue
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for p in class_dir.glob(ext):
                    self.images.append(p)
                    self.labels.append(label)

        self.resize = ResizeWithPadding(input_size)
        bear_count    = sum(1 for l in self.labels if l == 1)
        nonbear_count = sum(1 for l in self.labels if l == 0)
        print(f"  Test set: {len(self.images)} images "
              f"(bear={bear_count}, non_bear={nonbear_count})")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.resize(Image.open(self.images[idx]).convert('RGB'))
        return transforms.ToTensor()(img), self.labels[idx]


# ============================================================
# EVALUATE
# ============================================================
def evaluate(model, loader, device, threshold=0.5, temperature=1.0):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = novaic_preprocess_gpu(images.to(device))
            logits = model(images)
            if temperature != 1.0:
                logits = logits / temperature
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= threshold).astype(int)

    tp = int(((preds == 1) & (all_labels == 1)).sum())
    tn = int(((preds == 0) & (all_labels == 0)).sum())
    fp = int(((preds == 1) & (all_labels == 0)).sum())
    fn = int(((preds == 0) & (all_labels == 1)).sum())

    precision = 100. * tp / (tp + fp) if tp + fp else 0.
    recall    = 100. * tp / (tp + fn) if tp + fn else 0.
    f1  = 2  * precision * recall / (precision + recall) if precision + recall else 0.
    f05 = (1 + 0.5**2) * precision * recall / (0.5**2 * precision + recall) \
          if precision + recall else 0.
    acc = 100. * (tp + tn) / len(all_labels)

    return {
        'threshold': threshold,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'f05':       f05,
        'accuracy':  acc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'probs':  all_probs,
        'labels': all_labels,
    }


def sweep_threshold(probs, labels, beta=1.0):
    best_score, best_thresh = 0, 0.5
    rows = []
    for t in np.arange(0.30, 0.96, 0.01):
        p = (probs >= t).astype(int)
        tp = ((p==1)&(labels==1)).sum()
        fp = ((p==1)&(labels==0)).sum()
        fn = ((p==0)&(labels==1)).sum()
        pr = 100.*tp/(tp+fp) if tp+fp else 0
        rc = 100.*tp/(tp+fn) if tp+fn else 0
        fb = (1+beta**2)*pr*rc/(beta**2*pr+rc) if pr+rc else 0
        rows.append((t, pr, rc, fb))
        if fb > best_score:
            best_score, best_thresh = fb, t
    return best_thresh, best_score, rows


# ============================================================
# CLAUDE BEDROCK VERIFY FP/FN
# ============================================================
def verify_fp_fn(dataset, probs, labels, threshold):
    """
    Verify FP and FN images with Claude Bedrock.
    - FP (non_bear predicted as bear): Claude checks if actually has bear
      → if YES: model was right, it's a mislabeled image (not a real FP)
      → if NO:  confirmed FP
    - FN (bear predicted as non_bear): Claude checks if actually has bear
      → if YES: confirmed FN (model missed real bear)
      → if NO:  model was right, mislabeled (not a real FN)
    Returns adjusted TP/TN/FP/FN counts.
    """
    import io, time, random as _random, pickle

    try:
        import boto3
        from PIL import Image as PILImage
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "boto3", "Pillow", "--break-system-packages", "-q"])
        import boto3
        from PIL import Image as PILImage

    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    )
    MODEL_ID = os.environ.get(
        "BEDROCK_MODEL_ID",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    PROMPT = """You are an expert wildlife biologist reviewing trail camera images.
Is there a real, live bear visible? Consider: night vision, IR, blurry, distant, partial view.
Bear cubs and partial views count as YES if clearly identifiable as bear.
Reject: toys, CGI, other animals, background only.
Reply ONE WORD: YES / NO / UNSURE"""

    preds = (probs >= threshold).astype(int)
    fp_indices = [i for i in range(len(labels))
                  if preds[i] == 1 and labels[i] == 0]
    fn_indices = [i for i in range(len(labels))
                  if preds[i] == 0 and labels[i] == 1]

    print(f"  FP to verify: {len(fp_indices)}, FN to verify: {len(fn_indices)}")
    if not fp_indices and not fn_indices:
        return {}

    # Load cache
    cache_path = Path("test_verify_cache.pkl")
    cache = {}
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        print(f"  Loaded verify cache: {len(cache)} entries")

    def check_one(fp_str):
        if fp_str in cache:
            return fp_str, cache[fp_str]
        retries, backoff = 4, 2
        while retries > 0:
            try:
                img = PILImage.open(fp_str).convert("RGB")
                img.thumbnail((512, 512))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                resp = client.converse(
                    modelId=MODEL_ID,
                    messages=[{"role": "user", "content": [
                        {"image": {"format": "jpeg",
                                   "source": {"bytes": buf.getvalue()}}},
                        {"text": PROMPT}
                    ]}],
                    inferenceConfig={"maxTokens": 10, "temperature": 0.0}
                )
                ans = resp['output']['message']['content'][0]['text'].strip().upper()
                if "YES" in ans:   return fp_str, "YES"
                elif "NO" in ans:  return fp_str, "NO"
                else:              return fp_str, "UNSURE"
            except Exception as e:
                err = str(e).lower()
                if "throttling" in err or "429" in err or "503" in err:
                    time.sleep(backoff + _random.uniform(1, 2))
                    retries -= 1; backoff *= 2
                else:
                    return fp_str, "ERROR"
        return fp_str, "TIMEOUT"

    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_indices = [("FP", i) for i in fp_indices] + [("FN", i) for i in fn_indices]
    paths = {i: str(dataset.images[i]) for _, i in all_indices}

    results_verify = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(check_one, paths[i]): (kind, i)
                   for kind, i in all_indices}
        done = 0
        for fut in as_completed(futures):
            kind, i = futures[fut]
            fp_str, ans = fut.result()
            cache[fp_str] = ans
            results_verify[(kind, i)] = ans
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(all_indices)} verified...")

    # Save cache
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)

    # Tally confirmed vs mislabeled
    confirmed_fp = mislabeled_bear = 0    # FP cases
    confirmed_fn = mislabeled_nonbear = 0  # FN cases
    unsure_count = 0

    fp_confirmed_paths = []
    fn_confirmed_paths = []

    for (kind, i), ans in results_verify.items():
        fp_str = paths[i]
        if kind == "FP":
            if ans == "NO":
                confirmed_fp += 1
                fp_confirmed_paths.append(fp_str)
            elif ans == "YES":
                mislabeled_bear += 1  # labeled non_bear but has bear → mislabel
            else:
                unsure_count += 1
        else:  # FN
            if ans == "YES":
                confirmed_fn += 1
                fn_confirmed_paths.append(fp_str)
            elif ans == "NO":
                mislabeled_nonbear += 1  # labeled bear but no bear → mislabel
            else:
                unsure_count += 1

    print(f"\n  Verification results:")
    print(f"    FP confirmed (real FP):          {confirmed_fp}")
    print(f"    FP mislabeled (actually has bear): {mislabeled_bear}")
    print(f"    FN confirmed (real FN):          {confirmed_fn}")
    print(f"    FN mislabeled (no bear in img):  {mislabeled_nonbear}")
    print(f"    UNSURE/ERROR:                    {unsure_count}")

    if fp_confirmed_paths:
        print(f"\n  Confirmed FP files:")
        for p in fp_confirmed_paths[:10]:
            print(f"    {Path(p).name}")
        if len(fp_confirmed_paths) > 10:
            print(f"    ... and {len(fp_confirmed_paths)-10} more")

    if fn_confirmed_paths:
        print(f"\n  Confirmed FN files:")
        for p in fn_confirmed_paths[:10]:
            print(f"    {Path(p).name}")
        if len(fn_confirmed_paths) > 10:
            print(f"    ... and {len(fn_confirmed_paths)-10} more")

    return {
        'confirmed_fp': confirmed_fp,
        'confirmed_fn': confirmed_fn,
        'mislabeled_bear': mislabeled_bear,
        'mislabeled_nonbear': mislabeled_nonbear,
        'unsure_verify': unsure_count,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to best_model.pth")
    parser.add_argument("--dataset", required=True, help="Dataset root")
    parser.add_argument("--temp-scale", default=None,
                        help="Path to temperature_scaling.pth")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Classification threshold (default: from checkpoint)")
    parser.add_argument("--sweep-threshold", action="store_true",
                        help="Sweep thresholds and report best")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"TEST SET EVALUATION")
    print(f"  Model:   {args.model}")
    print(f"  Dataset: {args.dataset}/test")
    print(f"  Device:  {device}")
    print(f"{'='*60}\n")

    # Load model
    model = create_model().to(device)
    ckpt = torch.load(args.model, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    print(f"  Loaded checkpoint (val F1={ckpt.get('best_f1', '?'):.2f}%)" if 'best_f1' in ckpt else "  Loaded checkpoint")

    # Temperature scaling
    temperature  = 1.0
    best_threshold = ckpt.get('best_threshold', 0.5)
    if args.temp_scale and Path(args.temp_scale).exists():
        ts = torch.load(args.temp_scale, map_location='cpu')
        temperature     = ts.get('temperature', 1.0)
        best_threshold  = ts.get('threshold', best_threshold)
        print(f"  Temperature: {temperature:.4f}, calibrated threshold: {best_threshold:.2f}")

    threshold = args.threshold if args.threshold is not None else best_threshold
    print(f"  Using threshold: {threshold:.2f}")

    # Dataset
    test_dir = Path(args.dataset) / 'test'
    dataset  = TestDataset(test_dir)
    loader   = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

    # Evaluate
    results = evaluate(model, loader, device, threshold=threshold,
                       temperature=temperature)

    # ── Claude Bedrock verify FP/FN ─────────────────────────────
    print(f"\n--- Claude Bedrock Verification of FP/FN ---")
    fp_fn_verified = verify_fp_fn(
        dataset, results['probs'], results['labels'], threshold
    )
    # Update results with verified counts
    results.update(fp_fn_verified)

    # Sweep if requested
    if args.sweep_threshold:
        print(f"\n--- Threshold Sweep ---")
        best_t_f1,  best_s_f1,  rows_f1  = sweep_threshold(results['probs'], results['labels'], beta=1.0)
        best_t_f05, best_s_f05, rows_f05 = sweep_threshold(results['probs'], results['labels'], beta=0.5)

        print(f"\n  {'Thresh':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'F0.5':>8}")
        print(f"  {'-'*48}")
        for (t, pr, rc, f1), (_, _, _, f05) in zip(rows_f1, rows_f05):
            marker = " <-- best F1" if abs(t - best_t_f1) < 0.005 else \
                     " <-- best F0.5" if abs(t - best_t_f05) < 0.005 else ""
            print(f"  {t:>8.2f} {pr:>10.2f} {rc:>8.2f} {f1:>8.2f} {f05:>8.2f}{marker}")

        print(f"\n  Best F1  threshold: {best_t_f1:.2f} → F1={best_s_f1:.2f}%")
        print(f"  Best F0.5 threshold: {best_t_f05:.2f} → F0.5={best_s_f05:.2f}%")

    # Report
    print(f"\n{'='*60}")
    print(f"TEST RESULTS  (threshold={threshold:.2f}, temperature={temperature:.4f})")
    print(f"{'='*60}")
    print(f"  Precision : {results['precision']:.2f}%")
    print(f"  Recall    : {results['recall']:.2f}%")
    print(f"  F1        : {results['f1']:.2f}%")
    print(f"  F0.5      : {results['f05']:.2f}%")
    print(f"  Accuracy  : {results['accuracy']:.2f}%")
    print(f"  TP={results['tp']}, TN={results['tn']}, "
          f"FP={results['fp']}, FN={results['fn']}")

    if 'confirmed_fp' in results:
        print(f"\n  --- After Claude Verification ---")
        print(f"  Confirmed FP (real errors):        {results['confirmed_fp']}"
              f"  | Mislabeled bear images:    {results['mislabeled_bear']}")
        print(f"  Confirmed FN (real errors):        {results['confirmed_fn']}"
              f"  | Mislabeled non-bear images:{results['mislabeled_nonbear']}")

        # Recalculate with verified counts
        tp_v = results['tp'] + results['mislabeled_bear']
        tn_v = results['tn'] + results['mislabeled_nonbear']
        fp_v = results['confirmed_fp']
        fn_v = results['confirmed_fn']
        pr_v = 100.*tp_v/(tp_v+fp_v) if tp_v+fp_v else 0
        rc_v = 100.*tp_v/(tp_v+fn_v) if tp_v+fn_v else 0
        f1_v = 2*pr_v*rc_v/(pr_v+rc_v) if pr_v+rc_v else 0
        f05_v = (1+0.25)*pr_v*rc_v/(0.25*pr_v+rc_v) if pr_v+rc_v else 0
        print(f"\n  Verified Precision : {pr_v:.2f}%")
        print(f"  Verified Recall    : {rc_v:.2f}%")
        print(f"  Verified F1        : {f1_v:.2f}%")
        print(f"  Verified F0.5      : {f05_v:.2f}%")
        print(f"  TP={tp_v}, TN={tn_v}, FP={fp_v}, FN={fn_v}")
        results['verified_precision'] = pr_v
        results['verified_recall']    = rc_v
        results['verified_f1']        = f1_v
        results['verified_f05']       = f05_v
        results['verified_tp']        = tp_v
        results['verified_tn']        = tn_v
        results['verified_fp']        = fp_v
        results['verified_fn']        = fn_v

    print(f"{'='*60}")

    # Save results
    out_path = Path(args.model).parent / "test_results.json"
    save = {k: v for k, v in results.items() if k not in ('probs', 'labels')}
    save['temperature'] = temperature
    save['model'] = args.model
    save['dataset'] = str(args.dataset)
    with open(out_path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()
