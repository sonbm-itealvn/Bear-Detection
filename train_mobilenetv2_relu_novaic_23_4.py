"""
Bear Detection Training - MobileNetV2 (ReLU) with NOVAIC-COMPATIBLE Preprocessing
===================================================================================
CRITICAL FIX: ReLU thay ReLU6 -> Loai bo Clip operator -> Novaic NT98560 quantize tot
Input Size: 640x360 (FIXED)
OPTIMIZED FOR F1-SCORE
PREPROCESSING: Match Novaic hardware exactly

v7 UPDATE: Tang cuong augmentation cho 3 FP cases:
  1. Anh sang thay doi (brightness/exposure shifts)
  2. Nguoi mac do den (dark silhouette on bright background)
  3. Motion blur (camera shake, movement)

Novaic preprocessing:
  1. Input: BGR image [0-255]
  2. Mean subtraction: [104, 116, 124] (BGR)
  3. Scale: /255
  Result: (BGR - mean) / 255

Just run: python train_mobilenetv2_relu_novaic.py
"""

# ============================================================
# GPU SETUP - MUST BE FIRST!
# ============================================================
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
if "OVERRIDE_CUDA_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["OVERRIDE_CUDA_DEVICES"]
elif os.name == 'nt' or not os.path.exists("/dev/shm"):
    pass  # Default on local/Windows: use whatever GPUs (or CPU) are present
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"  # free 11GB RTX 2080 Ti GPUs (was "0")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
if os.name == 'nt':
    os.environ["TMPDIR"] = os.environ.get("TMPDIR", os.path.join(os.environ.get("TEMP", "tmp"), "phuocthien_tmp"))
else:
    os.environ["TMPDIR"] = os.environ.get("TMPDIR", "/dev/shm/phuocthien_tmp")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision import models
import torchvision.transforms.functional as TF
import pickle
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageFilter
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import random
import json
import time
import gc
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # dark_fp/ is next to this file
from dark_fp.supcon import ProjectionHead, supcon_loss, BearNet
from dark_fp.grouping import apply_group_weights

# ============================================================
# AWS BEDROCK CONFIG (for HEM Claude verification)
# Set here or via environment variables
# ============================================================
AWS_ACCESS_KEY_ID     = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
BEDROCK_MODEL_ID      = os.environ.get(
    "BEDROCK_MODEL_ID",
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
)

# Apply to environment so boto3 picks them up automatically
if AWS_ACCESS_KEY_ID:
    os.environ["AWS_ACCESS_KEY_ID"]     = AWS_ACCESS_KEY_ID
if AWS_SECRET_ACCESS_KEY:
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
if BEDROCK_MODEL_ID:
    os.environ["BEDROCK_MODEL_ID"]      = BEDROCK_MODEL_ID

# ============================================================
# FIXED CONFIGURATION
# ============================================================
MODEL_NAME = 'mobilenetv2_relu_novaic'
INPUT_SIZE = (360, 640)  # (H, W)
INPUT_SIZE_STR = '640x360'

# Allow overriding dataset dir via command line
import argparse as _argparse
_parser = _argparse.ArgumentParser(add_help=False)
_parser.add_argument("--data-dir", default="bear_dataset_v_final_final")
_parser.add_argument("--resume", default=None)
_parser.add_argument("--no-claude-verify", "--no-verify", dest="no_claude_verify",
                     action="store_true",
                     help="Disable Claude/Bedrock verification in Hard Example Mining "
                          "(no AWS needed; misclassified examples are boosted without "
                          "mislabel-checking).")
_parser.add_argument("--cache-only-verify", dest="cache_only_verify",
                     action="store_true",
                     help="HEM uses existing hem_gemini_cache.pkl verdicts to suppress known "
                          "mislabeled examples, but NEVER calls the Claude API (no AWS). "
                          "Uncached misclassified examples are kept (treated as UNSURE).")
_parser.add_argument("--supcon", action="store_true",
                     help="Enable train-time SupCon projection head + loss (dropped at export)")
_parser.add_argument("--supcon-weight", type=float, default=0.2)
_parser.add_argument("--dark-quad-boost", type=float, default=3.0)
_parser.add_argument("--night-bear-boost", type=float, default=2.0)
_args, _ = _parser.parse_known_args()

TRAIN_DIR = str(Path(_args.data_dir) / "train")
VAL_DIR = str(Path(_args.data_dir) / "val")

BATCH_SIZE = int(os.environ.get("OVERRIDE_BATCH_SIZE", 32))  # Per GPU. Restored to original 32 on 2026-07-29 after root-causing the OOM:
                         # physical GPU4's memory usage was mirroring physical GPU7's (an unrelated,
                         # separately-busy job) in perfect lockstep -- not a batch-size problem at all.
                         # Effective = 32 x NUM_GPUS.
NUM_WORKERS = int(os.environ.get("OVERRIDE_NUM_WORKERS", 8))  # DataLoader worker count (single loader, not per-GPU)
EPOCHS = int(os.environ.get("OVERRIDE_EPOCHS", 100))           # env override for fine-tune runs
EARLY_STOPPING_PATIENCE = int(os.environ.get("OVERRIDE_PATIENCE", 15))  # stop if no improvement
LEARNING_RATE = float(os.environ.get("OVERRIDE_LR", 0.001))    # env override for fine-tune runs
CHECKPOINT_FREQUENCY = 1

# Focal Loss parameters
FOCAL_ALPHA = 0.50
FOCAL_GAMMA = 2.5   # reduced from 3.0 — high gamma amplifies noise in hard IR/night examples
LABEL_SMOOTHING = 0.05

# Optimization metric beta:
# BETA=0.5 -> F0.5 (precision-focused, penalize FP more)
# BETA=1.0 -> F1  (balanced recall and precision)
# BETA=2.0 -> F2  (recall-focused, penalize FN more)
METRIC_BETA = 1.0  # Changed to F1 for balanced model

# Decision threshold. FIXED at 0.50 as of 2026-09-02.
#
# The Novaic app decides with a plain argmax over the two logits, which IS
# threshold 0.50, and the threshold cannot be moved after training: baking an
# offset into the final Gemm bias is absorbed exactly by the NPU's per-channel
# requantization (measured: threshold 0.43 vs 0.15 changed 0/200 decisions),
# and appending a Softmax to expose probabilities broke the app's output parser.
# So 0.50 is the only operating point that exists on device, and sweeping for an
# "optimal" threshold during training only tunes a knob the device never turns --
# it inflates reported F1 and hides how the model behaves where it actually runs.
#
# Consequence: the precision/recall trade-off must now come from the loss
# (FOCAL_ALPHA, SoftF1Loss fp_weight), not from the threshold.
FIXED_THRESHOLD = float(os.environ.get("OVERRIDE_THRESHOLD", 0.50))

# Resume from checkpoint (set path or None)
RESUME_FROM = _args.resume

# OVERRIDE_RUN_DIR gives a run its own directory. Without it every run writes
# checkpoint_epoch<N>.pth and best_model.pth into the same folder and silently
# overwrites the previous run's files of the same name -- which is how the
# epoch-33 checkpoint V16 was built from was lost. It also gives each run its
# own calibration_images/, exported from the dataset that run trained on.
WORKING_DIR = Path(os.environ.get("OVERRIDE_RUN_DIR",
                                  f"BearDetection_{MODEL_NAME}_{INPUT_SIZE_STR}_17_6"))
MODEL_DIR = WORKING_DIR / "models"
CHECKPOINT_DIR = WORKING_DIR / "checkpoints"
LOG_DIR = WORKING_DIR / "logs"

# ============================================================
# NOVAIC PREPROCESSING CONSTANTS
# ============================================================
NOVAIC_MEAN_BGR = [104/255, 116/255, 124/255]  # [0.408, 0.455, 0.486]
NOVAIC_STD = [1.0, 1.0, 1.0]  # No std division in Novaic

print(f"{'='*60}")
print(f"NOVAIC-COMPATIBLE PREPROCESSING + ReLU (NO Clip)")
print(f"{'='*60}")
print(f"Mean BGR (normalized): {NOVAIC_MEAN_BGR}")
print(f"Std: {NOVAIC_STD}")
print(f"Formula: (BGR/255) - mean = (BGR - [104,116,124]) / 255")
print(f"Activation: ReLU (NOT ReLU6 - no Clip operator)")
print(f"{'='*60}\n")

# ============================================================
# GPU CHECK
# ============================================================
print(f"{'='*60}")
print(f"GPU SETUP")
print(f"{'='*60}")
print(f"CUDA available: {torch.cuda.is_available()}")
NUM_GPUS = torch.cuda.device_count()
print(f"GPUs available: {NUM_GPUS}")
for i in range(NUM_GPUS):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")
print(f"{'='*60}\n")

# ============================================================
# FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.full_like(inputs, self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = torch.nn.functional.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        if self.label_smoothing > 0:
            ce_loss = -(smooth_targets * log_probs).sum(dim=1)
            p_t = (smooth_targets * probs).sum(dim=1)
        else:
            ce_loss = torch.nn.functional.nll_loss(log_probs, targets, reduction='none')
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * focal_weight * ce_loss
        return loss.mean()


class SoftF1Loss(nn.Module):
    """FP-weighted Soft F1: penalizes false positives heavier than false negatives."""
    def __init__(self, fp_weight=2.0):
        super().__init__()
        self.fp_weight = fp_weight  # FP penalized 2x vs FN

    def forward(self, inputs, targets):
        probs = torch.softmax(inputs, dim=1)[:, 1]
        targets_float = targets.float()
        tp = (probs * targets_float).sum()
        fp = (probs * (1 - targets_float)).sum() * self.fp_weight  # FP weighted heavier
        fn = ((1 - probs) * targets_float).sum()
        soft_f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        return 1 - soft_f1


class ConfidenceMarginLoss(nn.Module):
    """
    Penalize predictions near 0.5 (uncertain zone).
    Encourages model to output confident predictions (near 0 or 1).
    Does NOT force overconfidence like removing label smoothing.
    Instead, it just pushes away from the decision boundary.
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin  # Penalize predictions within [0.5-margin, 0.5+margin]

    def forward(self, inputs, targets):
        probs = torch.softmax(inputs, dim=1)[:, 1]  # P(bear)
        # Distance from 0.5
        dist_from_boundary = torch.abs(probs - 0.5)
        # Penalize predictions close to 0.5
        penalty = torch.clamp(self.margin - dist_from_boundary, min=0)
        return penalty.mean()


class CombinedLoss(nn.Module):
    def __init__(self, focal_alpha=0.5, focal_gamma=2.0, label_smoothing=0.0,
                 f1_weight=0.3, margin_weight=0.1):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, label_smoothing=label_smoothing)
        self.soft_f1 = SoftF1Loss(fp_weight=1.0)  # symmetric — was 2.0 which suppressed recall
        self.confidence_margin = ConfidenceMarginLoss(margin=0.3)
        self.f1_weight = f1_weight
        self.margin_weight = margin_weight

    def forward(self, inputs, targets):
        return (self.focal(inputs, targets)
                + self.f1_weight * self.soft_f1(inputs, targets)
                + self.margin_weight * self.confidence_margin(inputs, targets))


# ============================================================
# IMAGE ANALYZER
# ============================================================
class ImageAnalyzer:
    @staticmethod
    def get_brightness(img):
        return np.array(img).mean()

    @staticmethod
    def get_blur_score(img):
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def get_fog_score(img):
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        return gray.std()

    @staticmethod
    def get_object_size(img):
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges.sum() / (edges.size * 255)

    @staticmethod
    def get_center_variance(img):
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        center = img_array[h//4:3*h//4, w//4:3*w//4]
        return np.var(center)


# ============================================================
# SMART AUGMENTATIONS
# ============================================================
class MutuallyExclusiveNight:
    def __init__(self, p_low=0.05, p_night=0.03, brightness_threshold=80):
        self.p_low, self.p_night, self.threshold = p_low, p_night, brightness_threshold
        self.total_p = p_low + p_night
        self.analyzer = ImageAnalyzer()

    def __call__(self, img):
        if self.analyzer.get_brightness(img) < self.threshold:
            return img
        rand = random.random()
        if rand < self.p_low:
            return ImageEnhance.Brightness(img).enhance(random.uniform(0.4, 0.7))
        elif rand < self.total_p:
            return ImageEnhance.Brightness(img).enhance(random.uniform(0.3, 0.6))
        return img


class SmartZoomIn:
    def __init__(self, zoom_range=(1.2, 1.8), size_threshold=0.3):
        self.zoom_range, self.size_threshold = zoom_range, size_threshold
        self.analyzer = ImageAnalyzer()

    def __call__(self, img):
        if self.analyzer.get_object_size(img) > self.size_threshold:
            return img
        w, h = img.size
        scale = random.uniform(*self.zoom_range)
        new_w, new_h = int(w * scale), int(h * scale)
        img_large = TF.resize(img, (new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR)
        crop_x, crop_y = random.randint(0, new_w - w), random.randint(0, new_h - h)
        return img_large.crop((crop_x, crop_y, crop_x + w, crop_y + h))


class SmartZoomOut:
    def __init__(self, zoom_range=(0.5, 1.0), size_threshold=0.05):
        self.zoom_range, self.size_threshold = zoom_range, size_threshold
        self.analyzer = ImageAnalyzer()

    def __call__(self, img):
        if self.analyzer.get_object_size(img) < self.size_threshold:
            return img
        scale = random.uniform(0.7, 1.0) if self.analyzer.get_center_variance(img) < 800 else random.uniform(*self.zoom_range)
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        img_small = TF.resize(img, (new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR)
        canvas = Image.new(img.mode, (w, h), 0)
        canvas.paste(img_small, (random.randint(0, w - new_w), random.randint(0, h - new_h)))
        return canvas


class MutuallyExclusiveZoom:
    def __init__(self, p_zoom_in=0.20, p_zoom_out=0.15):
        self.p_zoom_in, self.p_zoom_out = p_zoom_in, p_zoom_out
        self.zoom_in, self.zoom_out = SmartZoomIn(), SmartZoomOut()

    def __call__(self, img):
        rand = random.random()
        if rand < self.p_zoom_in:
            return self.zoom_in(img)
        elif rand < self.p_zoom_in + self.p_zoom_out:
            return self.zoom_out(img)
        return img


class SmartMotionBlur:
    """
    v7 UPDATE: Tang p tu 0.05 -> 0.12 va giam blur_threshold tu 300 -> 100
    de tao nhieu blur hon, giup model robust voi FP do blur
    """
    def __init__(self, p=0.12, blur_threshold=100):
        self.p, self.blur_threshold = p, blur_threshold
        self.analyzer = ImageAnalyzer()

    def __call__(self, img):
        if random.random() > self.p:
            return img
        # Bo dieu kien blur_threshold de tao blur cho ca anh sac net
        img_array = np.array(img)
        kernel_size = random.choice([11, 15, 21, 25, 31])
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1.0
        M = cv2.getRotationMatrix2D((kernel_size // 2, kernel_size // 2), random.uniform(0, 180), 1.0)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        kernel = kernel / kernel.sum()
        return Image.fromarray(cv2.filter2D(img_array, -1, kernel))


class SmartFogAugmentation:
    def __init__(self, fog_range=(0.5, 0.9), p=0.05, fog_threshold=40):
        self.fog_range, self.p, self.fog_threshold = fog_range, p, fog_threshold
        self.analyzer = ImageAnalyzer()

    def __call__(self, img):
        if random.random() > self.p or self.analyzer.get_fog_score(img) < self.fog_threshold:
            return img
        img_array = np.array(img).astype(np.float32)
        h, w = img_array.shape[:2]
        fog_coef = random.uniform(*self.fog_range)
        depth_map = np.clip(np.tile(np.linspace(0, 1, h)[:, np.newaxis], (1, w)) + np.random.randn(h, w) * 0.1, 0, 1)
        fog_color = np.array([220, 225, 230]).reshape(1, 1, 3)
        # Vectorized: broadcasting depth_map[..., None] across 3 channels
        dm = (fog_coef * depth_map)[..., np.newaxis]
        img_array = img_array * (1 - dm) + fog_color * dm
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))


class ComprehensiveOcclusion:
    def __init__(self, p=0.15):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        overlay = img_array.copy()

        occlusion_type = random.choices(['vegetation', 'shadows', 'rocks'], weights=[0.5, 0.3, 0.2])[0]

        if occlusion_type == 'vegetation':
            # cv2.line is C++ internally, loop count is small (15-25), acceptable
            for _ in range(random.randint(15, 25)):
                x1, y1 = random.randint(0, w), random.randint(0, h)
                length = random.randint(h//4, h//2)
                angle = random.uniform(-45, 45)
                x2 = int(x1 + length * np.cos(np.radians(angle)))
                y2 = int(y1 + length * np.sin(np.radians(angle)))
                color = (random.randint(15, 50), random.randint(40, 80), random.randint(10, 40))
                cv2.line(overlay, (x1, y1), (x2, y2), color, random.randint(3, 12))
            img_array = cv2.addWeighted(img_array, 0.3, overlay, 0.7, 0)
        elif occlusion_type == 'shadows':
            shadow_mask = np.zeros((h, w), dtype=bool)
            # Vectorized shadow: use meshgrid instead of per-row loop
            rows = np.arange(h)
            for _ in range(random.randint(2, 4)):
                angle_val = random.uniform(-60, 60)
                width_val = random.randint(w//10, w//4)
                start_x = random.randint(-w//4, w)
                tan_val = np.tan(np.radians(angle_val))
                x1_arr = np.clip((start_x + (rows * tan_val)).astype(int), 0, w)
                x2_arr = np.clip(x1_arr + width_val, 0, w)
                cols = np.arange(w)
                # Broadcasting: mask[i,j] = 1 if x1[i] <= j < x2[i]
                shadow_mask |= ((cols[np.newaxis, :] >= x1_arr[:, np.newaxis]) &
                                (cols[np.newaxis, :] < x2_arr[:, np.newaxis]))
            shadow_mask = gaussian_filter(shadow_mask.astype(np.float32), sigma=15)
            # Vectorized: apply shadow across all channels
            img_array = (img_array * (1 - shadow_mask[..., np.newaxis] * 0.5)).astype(np.uint8)
        else:
            # cv2.fillPoly is C++, loop count is small (3-6), acceptable
            for _ in range(random.randint(3, 6)):
                cx, cy = random.randint(0, w), random.randint(0, h)
                size = random.randint(h//10, h//5)
                n_pts = random.randint(5, 8)
                angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
                radii = np.array([random.uniform(0.7, 1.3) for _ in range(n_pts)])
                xs = (cx + size * radii * np.cos(angles)).astype(np.int32)
                ys = (cy + size * radii * np.sin(angles)).astype(np.int32)
                points = np.stack([xs, ys], axis=1)
                cv2.fillPoly(overlay, [points], (random.randint(60,100), random.randint(60,100), random.randint(60,100)))
            img_array = cv2.addWeighted(img_array, 0.3, overlay, 0.7, 0)

        return Image.fromarray(img_array)


class RandomShiftScale:
    def __init__(self, p=0.20):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        scale = random.uniform(0.8, 1.2)
        new_w, new_h = int(w * scale), int(h * scale)
        img_scaled = TF.resize(img, (new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR)
        canvas = Image.new(img.mode, (w, h), 0)
        if new_w > w:
            img_scaled = img_scaled.crop(((new_w-w)//2, (new_h-h)//2, (new_w-w)//2+w, (new_h-h)//2+h))
        shift_x = int(w * random.uniform(-0.1, 0.1))
        shift_y = int(h * random.uniform(-0.1, 0.1))
        paste_x = max(0, min(w - img_scaled.size[0], (w - img_scaled.size[0])//2 + shift_x))
        paste_y = max(0, min(h - img_scaled.size[1], (h - img_scaled.size[1])//2 + shift_y))
        canvas.paste(img_scaled, (paste_x, paste_y))
        return canvas


class RandomGaussianNoise:
    def __init__(self, p=0.08):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_array = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 10, img_array.shape)
        return Image.fromarray(np.clip(img_array + noise, 0, 255).astype(np.uint8))


# ============================================================
# NEW v7 AUGMENTATIONS - FP FIX
# ============================================================
class LightingChangeAugmentation:
    """
    FP FIX #1: Anh sang moi truong thay doi
    Mo phong: den bat/tat, may che nang, exposure shift
    Tang cuong de model khong nham lighting change la bear
    """
    def __init__(self, p=0.15):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        aug_type = random.choice(['exposure_shift', 'partial_light', 'color_cast', 'auto_exposure'])

        img_array = np.array(img).astype(np.float32)
        h, w = img_array.shape[:2]

        if aug_type == 'exposure_shift':
            # Dot ngot tang/giam brightness (den bat/tat)
            factor = random.choice([
                random.uniform(0.3, 0.6),   # toi di
                random.uniform(1.5, 2.5),   # sang len
            ])
            img_array = np.clip(img_array * factor, 0, 255)

        elif aug_type == 'partial_light':
            # Mot phan anh sang, mot phan toi (may che, cua so)
            mask = np.ones((h, w, 1), dtype=np.float32)
            direction = random.choice(['top', 'bottom', 'left', 'right'])
            split = random.uniform(0.3, 0.7)
            dark_factor = random.uniform(0.3, 0.6)
            if direction == 'top':
                mask[:int(h*split), :, :] = dark_factor
            elif direction == 'bottom':
                mask[int(h*split):, :, :] = dark_factor
            elif direction == 'left':
                mask[:, :int(w*split), :] = dark_factor
            else:
                mask[:, int(w*split):, :] = dark_factor
            # Smooth transition
            mask_smooth = gaussian_filter(mask, sigma=30)
            img_array = np.clip(img_array * mask_smooth, 0, 255)

        elif aug_type == 'color_cast':
            # Thay doi white balance (anh sang fluorescent, vang, xanh)
            cast = np.zeros(3)
            cast[random.randint(0, 2)] = random.uniform(20, 60)
            if random.random() < 0.5:
                cast[random.randint(0, 2)] -= random.uniform(10, 30)
            img_array = np.clip(img_array + cast, 0, 255)

        elif aug_type == 'auto_exposure':
            # Mo phong auto-exposure adjustment (camera tu dong dieu chinh)
            gamma = random.choice([
                random.uniform(0.4, 0.7),  # under-exposed
                random.uniform(1.5, 2.5),  # over-exposed
            ])
            img_array = np.clip(255.0 * (img_array / 255.0) ** gamma, 0, 255)

        return Image.fromarray(img_array.astype(np.uint8))


class DarkSilhouetteAugmentation:
    """
    FP FIX #2: Nguoi mac do den
    Tao dark blob/silhouette tren nen sang de model
    hoc phan biet giua nguoi do den va bear
    """
    def __init__(self, p=0.10):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img_array = np.array(img).astype(np.float32)
        h, w = img_array.shape[:2]

        aug_type = random.choice(['dark_blob', 'contrast_boost', 'backlight'])

        if aug_type == 'dark_blob':
            num_blobs = random.randint(1, 3)
            for _ in range(num_blobs):
                cx = random.randint(w//4, 3*w//4)
                cy = random.randint(h//4, 3*h//4)
                rx = random.randint(w//10, w//4)
                ry = random.randint(h//6, h//3)
                darkness = random.uniform(0.1, 0.4)
                Y, X = np.ogrid[:h, :w]
                mask = ((X - cx)**2 / (rx**2 + 1e-6) + (Y - cy)**2 / (ry**2 + 1e-6)) <= 1.0
                mask_float = gaussian_filter(mask.astype(np.float32), sigma=max(rx, ry) * 0.3)
                # Vectorized: broadcast mask across channels
                img_array *= (1 - mask_float[..., np.newaxis] * (1 - darkness))

        elif aug_type == 'contrast_boost':
            factor = random.uniform(1.5, 2.5)
            mean = img_array.mean()
            img_array = np.clip((img_array - mean) * factor + mean, 0, 255)

        elif aug_type == 'backlight':
            brightness_boost = random.uniform(1.3, 1.8)
            img_array = np.clip(img_array * brightness_boost, 0, 255)
            cx, cy = w // 2, h // 2
            rx, ry = w // 3, h // 3
            Y, X = np.ogrid[:h, :w]
            mask = ((X - cx)**2 / (rx**2 + 1e-6) + (Y - cy)**2 / (ry**2 + 1e-6))
            mask = np.clip(1.0 - np.exp(-mask * 0.5), 0, 1)
            darkness = random.uniform(0.3, 0.6)
            # Vectorized: broadcast mask across channels
            img_array *= (mask * (1 - darkness) + darkness)[..., np.newaxis]

        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))


class StrongMotionBlur:
    """
    FP FIX #3: Motion blur manh hon
    Khac voi SmartMotionBlur, aug nay khong can check blur_threshold
    va tao blur manh hon de model robust voi FP do blur
    """
    def __init__(self, p=0.10):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img_array = np.array(img)

        blur_type = random.choice(['linear', 'radial', 'defocus'])

        if blur_type == 'linear':
            kernel_size = random.choice([21, 31, 41])
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size // 2, :] = 1.0
            angle = random.uniform(0, 180)
            M = cv2.getRotationMatrix2D((kernel_size // 2, kernel_size // 2), angle, 1.0)
            kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
            kernel = kernel / (kernel.sum() + 1e-8)
            img_array = cv2.filter2D(img_array, -1, kernel)

        elif blur_type == 'radial':
            # Zoom blur (camera di chuyen toi/lui)
            h, w = img_array.shape[:2]
            cx, cy = w // 2, h // 2
            result = img_array.astype(np.float32)
            n_steps = random.randint(5, 10)
            for i in range(1, n_steps + 1):
                scale = 1.0 + i * random.uniform(0.005, 0.015)
                M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
                shifted = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                result += shifted.astype(np.float32)
            img_array = np.clip(result / (n_steps + 1), 0, 255).astype(np.uint8)

        elif blur_type == 'defocus':
            # Out of focus blur
            ksize = random.choice([5, 7, 9, 11])
            img_array = cv2.GaussianBlur(img_array, (ksize, ksize), 0)

        return Image.fromarray(img_array)


class CameraShakeAugmentation:
    """
    FP FIX #4: Camera bi rung/chan dong khi di chuyen hoac lap dat
    Mo phong: rung da huong, frame shift, wobble
    Khac voi motion blur (1 huong), shake la rung nhieu huong ngau nhien
    """
    def __init__(self, p=0.12):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img_array = np.array(img)
        h, w = img_array.shape[:2]

        shake_type = random.choice(['multi_direction', 'frame_shift', 'wobble', 'jitter_blend'])

        if shake_type == 'multi_direction':
            # Rung nhieu huong: blend nhieu ban shift nho cua anh
            result = img_array.astype(np.float32)
            n_shifts = random.randint(3, 8)
            for _ in range(n_shifts):
                dx = random.randint(-15, 15)
                dy = random.randint(-15, 15)
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                shifted = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                result += shifted.astype(np.float32)
            img_array = np.clip(result / (n_shifts + 1), 0, 255).astype(np.uint8)

        elif shake_type == 'frame_shift':
            # Frame bi dich chuyen dot ngot (camera bi day/cham)
            dx = random.randint(-30, 30)
            dy = random.randint(-30, 30)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            img_array = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            # Them motion blur nhe theo huong shift
            if abs(dx) > 5 or abs(dy) > 5:
                ksize = random.choice([5, 7, 9])
                img_array = cv2.GaussianBlur(img_array, (ksize, ksize), 0)

        elif shake_type == 'wobble':
            # Camera bi lac (rotation nho + shift)
            angle = random.uniform(-8, 8)  # xoay nhe
            scale = random.uniform(0.95, 1.05)
            cx, cy = w // 2 + random.randint(-20, 20), h // 2 + random.randint(-20, 20)
            M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
            rotated = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            # Blend voi anh goc de tao hieu ung nhua
            alpha = random.uniform(0.3, 0.7)
            img_array = cv2.addWeighted(img_array, alpha, rotated, 1 - alpha, 0)

        elif shake_type == 'jitter_blend':
            # Mo phong nhieu frame bi rung blend lai (long exposure shake)
            result = np.zeros_like(img_array, dtype=np.float32)
            n_frames = random.randint(4, 10)
            for _ in range(n_frames):
                dx = random.gauss(0, 8)
                dy = random.gauss(0, 8)
                angle = random.gauss(0, 3)
                M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
                M[0, 2] += dx
                M[1, 2] += dy
                shifted = cv2.warpAffine(img_array, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                result += shifted.astype(np.float32)
            img_array = np.clip(result / n_frames, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)


# ============================================================
# SCREEN CAPTURE AUGMENTATION - SHARED (bear + non_bear)
# ============================================================
class ScreenCaptureAugmentation:
    """
    Simulate camera viewing a monitor/screen displaying the image.
    Partner tests by showing bear images on laptop screen to camera.
    Model must recognize both bear and non_bear through screen artifacts.

    Effects: moire pattern (incl. strong IR-style), scanlines, screen bezel,
    color banding, screen glare, perspective warp, IR desaturation.
    """
    def __init__(self, p=0.10):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img_array = np.array(img).astype(np.float32)
        h, w = img_array.shape[:2]

        # Apply 1-4 random screen effects (increased from 1-3)
        effects = random.sample(
            ['moire', 'moire_strong', 'scanlines', 'bezel',
             'color_band', 'glare', 'perspective', 'ir_desaturate'],
            k=random.randint(1, 4)
        )

        for effect in effects:
            if effect == 'moire':
                freq = random.uniform(0.3, 0.8)
                Y, X = np.mgrid[0:h, 0:w]
                pattern = np.sin(Y * freq * np.pi) * np.sin(X * freq * np.pi)
                intensity = random.uniform(10, 30)
                img_array += (pattern * intensity)[..., np.newaxis]
                img_array = np.clip(img_array, 0, 255)

            elif effect == 'moire_strong':
                freq1 = random.uniform(0.4, 1.2)
                freq2 = freq1 * random.uniform(0.85, 1.15)
                Y, X = np.mgrid[0:h, 0:w]
                pattern = (np.sin(Y * freq1 * np.pi) + np.sin(X * freq2 * np.pi)) * 0.5
                intensity = random.uniform(10, 25)  # reduced from 20-50
                img_array += (pattern * intensity)[..., np.newaxis]
                img_array = np.clip(img_array, 0, 255)  # clamp immediately
                scan_spacing = random.randint(3, 8)
                img_array[0:h:scan_spacing, :] *= random.uniform(0.75, 0.90)

            elif effect == 'scanlines':
                line_spacing = random.randint(2, 5)
                darkness = random.uniform(0.7, 0.9)
                img_array[0:h:line_spacing, :] *= darkness

            elif effect == 'bezel':
                border = random.randint(10, 40)
                img_array[:border, :] = random.randint(0, 20)
                img_array[h-border:, :] = random.randint(0, 20)
                img_array[:, :border] = random.randint(0, 20)
                img_array[:, w-border:] = random.randint(0, 20)

            elif effect == 'color_band':
                levels = random.choice([16, 24, 32])
                img_array = np.round(img_array / (256/levels)) * (256/levels)

            elif effect == 'glare':
                cx = random.randint(w//4, 3*w//4)
                cy = random.randint(h//4, 3*h//4)
                radius = random.randint(min(h,w)//6, min(h,w)//3)
                Y, X = np.ogrid[:h, :w]
                dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
                glare_mask = np.clip(1.0 - dist / radius, 0, 1) ** 2
                glare_intensity = random.uniform(40, 100)
                img_array += (glare_mask * glare_intensity)[..., np.newaxis]
                img_array = np.clip(img_array, 0, 255)

            elif effect == 'perspective':
                pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
                shift = random.randint(5, 25)
                side = random.choice(['left', 'right', 'top', 'bottom'])
                pts2 = pts1.copy()
                if side == 'left':
                    pts2[0][1] += shift; pts2[2][1] -= shift
                elif side == 'right':
                    pts2[1][1] += shift; pts2[3][1] -= shift
                elif side == 'top':
                    pts2[0][0] += shift; pts2[1][0] -= shift
                elif side == 'bottom':
                    pts2[2][0] += shift; pts2[3][0] -= shift
                M = cv2.getPerspectiveTransform(pts1, pts2)
                img_array = cv2.warpPerspective(
                    img_array.astype(np.uint8), M, (w, h),
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)
                ).astype(np.float32)

            elif effect == 'ir_desaturate':
                # IR camera effect: near-grayscale + slight green tint
                # Simulates night-vision screen recording
                gray = 0.299*img_array[:,:,0] + 0.587*img_array[:,:,1] + 0.114*img_array[:,:,2]
                blend = random.uniform(0.5, 0.85)
                img_array = img_array * (1 - blend) + gray[..., np.newaxis] * blend
                # Slight greenish tint (IR through screen)
                img_array[:,:,1] *= random.uniform(1.0, 1.1)
                img_array *= random.uniform(0.75, 0.95)

        # Overall screen effect
        img_array = img_array * random.uniform(0.85, 0.95) + random.uniform(5, 15)
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))


# ============================================================
# BEAR-ONLY AUGMENTATIONS - RECALL FIX
# ============================================================
class NightVisionSimulation:
    """
    RECALL FIX #1: Bear trong night vision
    Mo phong: camera night vision (IR), giam brightness, gan grayscale, IR noise
    """
    def __init__(self, p=0.15):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img_array = np.array(img).astype(np.float32)
        nv_type = random.choice(['ir_grayscale', 'very_dark', 'low_contrast', 'ir_green'])

        if nv_type == 'ir_grayscale':
            gray = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
            blend = random.uniform(0.6, 0.9)
            # Vectorized: broadcast gray across channels
            img_array = img_array * (1 - blend) + gray[..., np.newaxis] * blend
            img_array *= random.uniform(0.3, 0.6)
            img_array = np.clip(img_array + np.random.normal(0, random.uniform(8, 20), img_array.shape), 0, 255)

        elif nv_type == 'very_dark':
            img_array *= random.uniform(0.15, 0.35)
            img_array = np.clip(img_array + np.random.normal(0, random.uniform(10, 25), img_array.shape), 0, 255)

        elif nv_type == 'low_contrast':
            mean = img_array.mean()
            factor = random.uniform(0.3, 0.5)
            img_array = (img_array - mean) * factor + mean * 0.5
            img_array = np.clip(img_array + np.random.normal(0, random.uniform(5, 15), img_array.shape), 0, 255)

        elif nv_type == 'ir_green':
            gray = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
            # Vectorized: build 3-channel multiplier
            factors = np.array([random.uniform(0.1, 0.3), random.uniform(0.5, 0.8), random.uniform(0.1, 0.3)])
            img_array = gray[..., np.newaxis] * factors[np.newaxis, np.newaxis, :]
            img_array *= random.uniform(0.4, 0.7)
            img_array = np.clip(img_array + np.random.normal(0, random.uniform(5, 15), img_array.shape), 0, 255)

        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))


class DistanceSimulation:
    """
    RECALL FIX #2: Bear o xa (nho trong frame)
    Zoom out bear, them background padding.
    Background is built from a corner crop of the original image (not the
    full frame) so it doesn't retain a large blurred ghost of the bear
    itself -- blurring the whole original left a recognizable bear-shaped
    silhouette behind the small pasted copy, letting the model "cheat" via
    that ghost instead of learning to detect a genuinely small, distant bear.
    Corners are the least likely region to contain the (typically centered)
    subject, so cropping from there keeps the background bear-free while
    still looking like real surrounding terrain from the same photo.
    """
    def __init__(self, p=0.12):
        self.p = p

    def _corner_background(self, img, w, h, blur_radius):
        cw, ch = max(1, w // 4), max(1, h // 4)
        corners = [
            (0, 0, cw, ch), (w - cw, 0, w, ch),
            (0, h - ch, cw, h), (w - cw, h - ch, w, h),
        ]
        cx0, cy0, cx1, cy1 = random.choice(corners)
        patch = img.crop((cx0, cy0, cx1, cy1))
        canvas = patch.resize((w, h), Image.BILINEAR)
        return canvas.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    def __call__(self, img):
        if random.random() > self.p:
            return img

        w, h = img.size
        scale = random.uniform(0.2, 0.55)  # extended range: 0.3-0.6 -> 0.2-0.55
        new_w, new_h = int(w * scale), int(h * scale)
        img_small = TF.resize(img, (new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR)

        # Background: corner-crop of original, blurred (realistic forest/environment,
        # bear-free). Weighted heavily towards corner_soft since black canvas is unrealistic
        bg_type = random.choices(
            ['corner_soft', 'corner_heavy', 'noise', 'black'],
            weights=[0.5, 0.3, 0.15, 0.05]
        )[0]

        if bg_type == 'black':
            canvas = Image.new('RGB', (w, h), (0, 0, 0))
        elif bg_type == 'noise':
            canvas = Image.fromarray(np.random.randint(0, 50, (h, w, 3), dtype=np.uint8))
        elif bg_type == 'corner_soft':
            canvas = self._corner_background(img, w, h, blur_radius=15)
        else:  # corner_heavy
            canvas = self._corner_background(img, w, h, blur_radius=30)
            # Darken slightly to simulate depth
            canvas_arr = np.array(canvas).astype(np.float32)
            canvas_arr *= random.uniform(0.5, 0.85)
            canvas = Image.fromarray(canvas_arr.astype(np.uint8))

        paste_x = random.randint(0, w - new_w)
        paste_y = random.randint(0, h - new_h)
        canvas.paste(img_small, (paste_x, paste_y))
        return canvas


class HeavyOcclusionSimulation:
    """
    RECALL FIX #3: Bear bi che nhieu (30-60%)
    """
    def __init__(self, p=0.12):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img_array = np.array(img)
        h, w = img_array.shape[:2]
        occ_type = random.choice(['bars', 'large_block', 'gradient_fade'])

        if occ_type == 'bars':
            for _ in range(random.randint(3, 8)):
                color = random.randint(0, 60)
                if random.random() < 0.5:
                    y = random.randint(0, h - 1)
                    t = random.randint(h // 15, h // 6)
                    img_array[y:min(h, y+t), :] = color
                else:
                    x = random.randint(0, w - 1)
                    t = random.randint(w // 15, w // 6)
                    img_array[:, x:min(w, x+t)] = color

        elif occ_type == 'large_block':
            bw = int(w * random.uniform(0.3, 0.6))
            bh = int(h * random.uniform(0.3, 0.6))
            color = random.randint(0, 80)
            corner = random.choice(['tl', 'tr', 'bl', 'br'])
            if corner == 'tl': img_array[:bh, :bw] = color
            elif corner == 'tr': img_array[:bh, w-bw:] = color
            elif corner == 'bl': img_array[h-bh:, :bw] = color
            else: img_array[h-bh:, w-bw:] = color

        elif occ_type == 'gradient_fade':
            d = random.choice(['left', 'right', 'top', 'bottom'])
            ratio = random.uniform(0.3, 0.6)
            img_f = img_array.astype(np.float32)
            if d == 'left':
                fw = int(w * ratio)
                # Vectorized: create 1D gradient, broadcast across rows and channels
                grad = np.linspace(0, 1, fw).reshape(1, fw, 1)
                img_f[:, :fw] *= grad
            elif d == 'right':
                fw = int(w * ratio)
                grad = np.linspace(1, 0, fw).reshape(1, fw, 1)
                img_f[:, w-fw:] *= grad
            elif d == 'top':
                fh = int(h * ratio)
                grad = np.linspace(0, 1, fh).reshape(fh, 1, 1)
                img_f[:fh, :] *= grad
            else:
                fh = int(h * ratio)
                grad = np.linspace(1, 0, fh).reshape(fh, 1, 1)
                img_f[h-fh:, :] *= grad
            img_array = img_f.astype(np.uint8)

        return Image.fromarray(img_array)




# ============================================================
# SAME-CLASS MIXUP (only mix bear with bear, non_bear with non_bear)
# ============================================================
def mixup_data_same_class(x, y, alpha=0.2):
    """Mixup that only mixes within same class - prevents blending bear with non_bear."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)

    # Create same-class shuffle index
    index = torch.zeros(batch_size, dtype=torch.long, device=x.device)
    for cls in [0, 1]:
        cls_mask = (y == cls).nonzero(as_tuple=True)[0]
        if len(cls_mask) > 1:
            shuffled = cls_mask[torch.randperm(len(cls_mask), device=x.device)]
            index[cls_mask] = shuffled
        elif len(cls_mask) == 1:
            index[cls_mask] = cls_mask

    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y, lam  # y_a == y_b since same class


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# RESIZE WITH PADDING
# ============================================================
class ResizeWithPadding:
    def __init__(self, target_size):
        self.target_h, self.target_w = target_size

    def __call__(self, img):
        w, h = img.size
        scale = min(self.target_w / w, self.target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = TF.resize(img, (new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR)
        canvas = Image.new('RGB', (self.target_w, self.target_h), (0, 0, 0))
        canvas.paste(img_resized, ((self.target_w - new_w) // 2, (self.target_h - new_h) // 2))
        return canvas


# ============================================================
# NOVAIC-COMPATIBLE TRANSFORM
# ============================================================
class NoviacTransform:
    """
    Transform that matches Novaic hardware preprocessing exactly.

    Novaic does:
      1. Read image as BGR
      2. Subtract mean BGR [104, 116, 124]
      3. Divide by 255

    In PyTorch (which uses RGB), we:
      1. Convert RGB to BGR
      2. Normalize with mean_bgr/255 and std=1
    """
    def __init__(self):
        self.mean_bgr = torch.tensor([104/255, 116/255, 124/255])

    def __call__(self, img):
        tensor_rgb = transforms.ToTensor()(img)  # [C, H, W] RGB, range [0, 1]
        tensor_bgr = tensor_rgb[[2, 1, 0], :, :]  # BGR
        tensor_bgr[0] -= self.mean_bgr[0]  # B
        tensor_bgr[1] -= self.mean_bgr[1]  # G
        tensor_bgr[2] -= self.mean_bgr[2]  # R
        return tensor_bgr


# ============================================================
# DATASETS
# ============================================================
class BearDatasetTrain(Dataset):
    def __init__(self, root_dir, input_size):
        self.root_dir = Path(root_dir)
        self.images, self.labels = [], []

        for class_dir in self.root_dir.iterdir():
            if not class_dir.is_dir():
                continue
            label = 0 if 'non' in class_dir.name.lower() else (1 if 'bear' in class_dir.name.lower() else -1)
            if label < 0:
                continue
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                for p in class_dir.glob(ext):
                    self.images.append(p)
                    self.labels.append(label)

        # Ablation runs train on a fixed random subset so one arm fits in a few
        # hours on a single shared GPU. Seeded, so every arm sees the same images.
        frac = float(os.environ.get("OVERRIDE_TRAIN_FRACTION", 1.0))
        if frac < 1.0:
            keep = sorted(random.Random(0).sample(range(len(self.images)),
                                                  int(len(self.images) * frac)))
            self.images = [self.images[i] for i in keep]
            self.labels = [self.labels[i] for i in keep]
            print(f"  OVERRIDE_TRAIN_FRACTION={frac}: training on {len(self.images):,} images")

        self.spatial = transforms.Compose([
            ResizeWithPadding(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=45, expand=False, fill=0),
        ])

        # Shared augmentations: apply to BOTH bear and non_bear
        self.shared_augmentations = [
            RandomShiftScale(), MutuallyExclusiveZoom(), SmartFogAugmentation(),
            MutuallyExclusiveNight(), ComprehensiveOcclusion(), SmartMotionBlur(),
            RandomGaussianNoise(),
            # ScreenCaptureAugmentation re-enabled 2026-08-10: acceptance testing is
            # reverting to camera-filming-a-screen methodology, so moire/scanline/
            # bezel/glare/perspective-warp artifacts are back in test data.
            ScreenCaptureAugmentation(p=0.30),
        ]

        # FP-fix augmentations: apply ONLY to non_bear (label=0)
        self.nonbear_only_augmentations = [
            LightingChangeAugmentation(p=0.15),
            DarkSilhouetteAugmentation(p=0.10),
            StrongMotionBlur(p=0.10),
            CameraShakeAugmentation(p=0.12),
        ]

        # Recall-fix augmentations: apply ONLY to bear (label=1)
        # Keep p moderate — too high causes model to see distorted bear more than clean bear
        self.bear_only_augmentations = [
            NightVisionSimulation(p=0.15),   # back to 0.15 — 0.20 was too aggressive
            DistanceSimulation(p=0.15),      # back to 0.15
            HeavyOcclusionSimulation(p=0.12),
        ]

        # OVERRIDE_SYMMETRIC_AUG=1 applies the night-vision and distance
        # simulations to non_bear as well. Bear-only, they make "IR look" and
        # "small sharp subject on a blurred background" class features -- the
        # same mistake the asymmetric ScreenCaptureAugmentation made with moire.
        # On 26/09 the remaining dark-fur false positives were exactly these two
        # groups: distant animals and close-up IR shots.
        if os.environ.get("OVERRIDE_SYMMETRIC_AUG") == "1":
            self.shared_augmentations += [NightVisionSimulation(p=0.15),
                                          DistanceSimulation(p=0.15)]
            self.bear_only_augmentations = [HeavyOcclusionSimulation(p=0.12)]
            print("  OVERRIDE_SYMMETRIC_AUG=1: NightVision + Distance applied to BOTH classes")

        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
        self.novaic_transform = NoviacTransform()

        bear_count = sum(1 for l in self.labels if l == 1)
        nonbear_count = sum(1 for l in self.labels if l == 0)
        print(f"\nTrain: {len(self.images):,} images (bear={bear_count:,}, non_bear={nonbear_count:,})")
        print(f"  Shared augmentations: {len(self.shared_augmentations)}")
        print(f"  Non-bear only: {len(self.nonbear_only_augmentations)} (FP-fix)")
        print(f"  Bear only: {len(self.bear_only_augmentations)} (recall-fix)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        img = self.spatial(img)

        # Shared augmentations for all
        for aug in self.shared_augmentations:
            img = aug(img)

        # FP-fix augmentations ONLY for non_bear
        if label == 0:
            for aug in self.nonbear_only_augmentations:
                img = aug(img)

        # Recall-fix augmentations ONLY for bear
        if label == 1:
            for aug in self.bear_only_augmentations:
                img = aug(img)

        # Ensure valid pixel range before ColorJitter
        img_arr = np.array(img)
        if np.isnan(img_arr).any() or np.isinf(img_arr).any():
            img_arr = np.zeros_like(img_arr)
        img = Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))

        # Convert to tensor RGB [0,1], apply ColorJitter on tensor (CPU)
        tensor_rgb = transforms.ToTensor()(img)
        tensor_rgb = self.color_jitter(tensor_rgb)

        # Return RGB tensor - Novaic BGR transform done on GPU in train_epoch
        return tensor_rgb, label

    def get_labels_for_sampler(self):
        return self.labels


class _CleanView(Dataset):
    """The train images with NO augmentation (letterbox + tensor only).
    OVERRIDE_HEM_CLEAN=1 mines hard examples on this view ("HEM-B"): measured in
    bear_experiments_3009 on the device, mining fixed images cut false
    positives 44% (2.5% -> 1.4%) for -1.5 recall, while mining augmented
    images scores each example on a random distortion of it. Indices match the
    train dataset, so HEM weights stay aligned."""
    def __init__(self, train_ds):
        self.images, self.labels = train_ds.images, train_ds.labels
        self.resize = train_ds.spatial.transforms[0]       # ResizeWithPadding

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.resize(Image.open(self.images[idx]).convert('RGB'))
        return transforms.ToTensor()(img), self.labels[idx]


class BearDatasetVal(Dataset):
    def __init__(self, root_dir, input_size):
        self.root_dir = Path(root_dir)
        self.images, self.labels = [], []

        for class_dir in self.root_dir.iterdir():
            if not class_dir.is_dir():
                continue
            label = 0 if 'non' in class_dir.name.lower() else (1 if 'bear' in class_dir.name.lower() else -1)
            if label < 0:
                continue
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                for p in class_dir.glob(ext):
                    self.images.append(p)
                    self.labels.append(label)

        self.resize = ResizeWithPadding(input_size)
        self.novaic_transform = NoviacTransform()

        print(f"Val: {len(self.images):,} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.resize(Image.open(self.images[idx]).convert('RGB'))
        tensor_rgb = transforms.ToTensor()(img)  # RGB [0,1] - Novaic on GPU
        return tensor_rgb, self.labels[idx]


# ============================================================
# MODEL - CRITICAL: ReLU thay ReLU6
# ============================================================
# OVERRIDE_BACKBONE=<timm model name> swaps the torchvision MobileNetV2 for a
# timm ImageNet backbone (2026-10-05 pilot). Compile feasibility on the NT98560,
# measured by backbone_feasibility.py at 640x360: MobileNetV2 24.1 ms,
# mobileone_s1 28.3 ms, regnetx_008 30.0 ms; none needs a CPU layer. The NPU
# runs dense convs at 36-75% utilisation and depthwise ones at 14-22%.
# Empty = the original MobileNetV2 path, untouched.
BACKBONE = os.environ.get("OVERRIDE_BACKBONE", "")


class _TimmFeatures(nn.Module):
    """timm forward_features (pre-pool map) behind a `.features` attribute, so
    BearNet -- adaptive_avg_pool + classifier -- wraps it like MobileNetV2."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.forward_features(x)


def create_timm_model(name):
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    import timm
    m = timm.create_model(name, pretrained=True, num_classes=0)
    n = 0

    def swap(module):
        nonlocal n
        for cname, child in module.named_children():
            if isinstance(child, (nn.ReLU6, nn.Hardtanh)):
                setattr(module, cname, nn.ReLU(inplace=True))
                n += 1
            else:
                swap(child)
    swap(m)
    # Ops the NT98560 cannot run on its NN engine (no sigmoid ACT mode on 56x).
    bad = sorted({type(x).__name__ for x in m.modules()
                  if isinstance(x, (nn.SiLU, nn.Hardswish, nn.Hardsigmoid, nn.Sigmoid, nn.GELU))})
    if bad:
        raise RuntimeError(f"{name} contains {bad}: not mappable to the NT98560 NN engine")
    wrapper = nn.Module()
    wrapper.features = _TimmFeatures(m)
    wrapper.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(m.num_features, 2))
    print(f"  Backbone: timm {name} (ImageNet), {n} ReLU6 -> ReLU, "
          f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f}M params")
    return wrapper


def create_model():
    """
    MobileNetV2 voi ReLU thay ReLU6.
    ReLU6 tao Clip operator -> Novaic NT98560 quantize FAIL
    (cos similarity sut tu 0.998 xuong 0.621 tai features.5)
    """
    if BACKBONE:
        return create_timm_model(BACKBONE)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # CRITICAL: Replace ALL ReLU6 -> ReLU
    relu6_count = 0
    def replace_relu6(module):
        nonlocal relu6_count
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU6):
                setattr(module, name, nn.ReLU(inplace=True))
                relu6_count += 1
            else:
                replace_relu6(child)

    replace_relu6(model)
    print(f"  Replaced {relu6_count} ReLU6 -> ReLU (no Clip operator)")

    # Verify
    for name, m in model.named_modules():
        if isinstance(m, nn.ReLU6):
            raise RuntimeError(f"ReLU6 still found at {name}!")
    print("  VERIFIED: No ReLU6 remaining")

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, 2)
    )
    return model


# ============================================================
# OPTIMAL THRESHOLD FINDER
# ============================================================
def find_optimal_threshold(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            images = novaic_preprocess_gpu(images)
            outputs = _logits(model(images))
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    BETA = METRIC_BETA
    best_score, best_thresh = 0, 0.5
    for thresh in np.arange(0.30, 0.96, 0.01):
        preds = (all_probs >= thresh).astype(int)
        tp = ((preds == 1) & (all_labels == 1)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fbeta = (1 + BETA**2) * precision * recall / (BETA**2 * precision + recall) if (precision + recall) > 0 else 0

        if fbeta > best_score:
            best_score = fbeta
            best_thresh = thresh

    return best_thresh, best_score


# ============================================================
# HARD EXAMPLE MINING
# ============================================================
class HardExampleMiner:
    """
    Track misclassified training images with asymmetric boosting.
    FP (non_bear predicted as bear) boosted 3x heavier than FN.
    Optionally verifies misclassified images with Gemini to avoid
    boosting mislabeled images.
    """
    def __init__(self, num_samples, fp_boost=5.0, fn_boost=2.0, decay=0.7,
                 gemini_verify=False, gemini_cache_file="hem_gemini_cache.pkl",
                 extra_caches=None, cache_only=False):
        self.num_samples = num_samples
        self.fp_boost = fp_boost
        self.fn_boost = fn_boost
        self.decay = decay
        # FP and FN are tracked in SEPARATE arrays (not a single merged
        # magnitude-weighted array) because a training sample's ground-truth
        # label never changes, so a given index can only ever accumulate
        # fp_counts (label=non_bear) or fn_counts (label=bear) -- never
        # both -- making the boost type unambiguous without guessing from
        # a decayed aggregate's magnitude.
        self.fp_counts = np.zeros(num_samples, dtype=np.float32)
        self.fn_counts = np.zeros(num_samples, dtype=np.float32)
        self.gemini_verify = gemini_verify
        self.cache_only = cache_only          # use cache verdicts but NEVER call the API
        # Load caches for either real-verify OR cache-only mode
        load_cache = gemini_verify or cache_only
        self.gemini_cache_file = Path(gemini_cache_file)
        # {image_path_str: "YES"/"NO"/"UNSURE"/"ERROR"}
        self.gemini_cache = {}

        # Load primary cache
        if load_cache and self.gemini_cache_file.exists():
            with open(self.gemini_cache_file, 'rb') as f:
                self.gemini_cache = pickle.load(f)
            print(f"  HEM Gemini cache loaded: {len(self.gemini_cache)} entries from {self.gemini_cache_file.name}")

        # Load extra caches (read-only) — merge into self.gemini_cache
        if load_cache and extra_caches:
            for cache_path in extra_caches:
                cp = Path(cache_path)
                if not cp.exists():
                    print(f"  HEM: extra cache {cp.name} not found, skipping")
                    continue
                with open(cp, 'rb') as f:
                    extra = pickle.load(f)
                # extra cache may have entries as dict {gemini, folder_label, ...}
                # or as plain string. Normalize to plain string.
                merged_count = 0
                for k, v in extra.items():
                    if k in self.gemini_cache:
                        continue
                    if isinstance(v, dict):
                        ans = v.get('gemini', 'UNSURE')
                    else:
                        ans = str(v)
                    self.gemini_cache[k] = ans
                    merged_count += 1
                print(f"  HEM: merged {merged_count} entries from {cp.name}")

        # Load collected_bears gemini_cache.pkl files
        # Format: {filename: "YES"/"NO"/"UNKNOWN"} with relative filenames
        # Resolve to full paths via bear_frames folder
        if load_cache:
            collected_root = Path("/data/nas07_new/PersonalData/phuocthien/collected_bears")
            if collected_root.exists():
                collected_merged = 0
                for gemini_pkl in collected_root.glob("*/bear_frames/gemini_cache.pkl"):
                    bear_frames_dir = gemini_pkl.parent
                    with open(gemini_pkl, 'rb') as f:
                        bear_cache = pickle.load(f)
                    for fname, ans in bear_cache.items():
                        full_path = str(bear_frames_dir / fname)
                        if full_path in self.gemini_cache:
                            continue
                        # Normalize: "UNKNOWN" -> "UNSURE", keep YES/NO
                        if ans == "UNKNOWN":
                            ans = "UNSURE"
                        self.gemini_cache[full_path] = ans
                        collected_merged += 1
                if collected_merged > 0:
                    print(f"  HEM: merged {collected_merged} entries from collected_bears gemini_cache")
            else:
                print(f"  HEM: collected_bears not found at {collected_root}, skipping")

        if gemini_verify and (extra_caches or True):
            # Save merged cache
            with open(self.gemini_cache_file, 'wb') as f:
                pickle.dump(self.gemini_cache, f)
            print(f"  HEM: total cache size: {len(self.gemini_cache)}")

    def _gemini_check_batch(self, image_paths):
        """Check uncached images with Gemini. Returns {path_str: answer}."""
        import time, random as _random, base64, io
        try:
            import boto3
            from PIL import Image as PILImage
        except ImportError:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install",
                            "boto3", "Pillow", "--break-system-packages", "-q"])
            import boto3
            from PIL import Image as PILImage

        # None, not "": boto3 treats an empty string as a real key, signs with
        # it and never falls back to ~/.aws/credentials -- so with no env vars
        # set, every HEM verify call failed and ended up as TIMEOUT/ERROR.
        AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID") or None
        AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY") or None
        AWS_REGION     = os.environ.get("AWS_REGION", "us-east-1")
        MODEL_ID       = os.environ.get(
            "BEDROCK_MODEL_ID",
            "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )

        client = boto3.client(
            service_name="bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
        )

        PROMPT = """You are an expert wildlife biologist reviewing trail camera images.
Look at this image carefully. Is there a real, live bear visible in the image?
Consider difficult conditions: night vision, IR, thermal, blurry, distant, partial view.
Bear cubs and partial views (just paw, fur, or silhouette) still count as YES if clearly identifiable as bear.
Reject only: toys, CGI/animated bears, other animals (boar, dog, raccoon, etc), or no animal at all.

Reply with EXACTLY ONE WORD:
- "YES" if there is a real bear (any size, any view, even partial).
- "NO" if there is no bear, or only other animals, or only background.
- "UNSURE" only if image quality makes it truly impossible to tell.
"""

        MAX_WORKERS = 10  # Bedrock has per-minute rate limits, keep conservative

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def check_one(fp_str):
            retries, backoff = 5, 3
            while retries > 0:
                try:
                    # Load + resize image, encode as base64
                    img = PILImage.open(fp_str).convert("RGB")
                    img.thumbnail((512, 512))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                    response = client.converse(
                        modelId=MODEL_ID,
                        messages=[{
                            "role": "user",
                            "content": [
                                {
                                    "image": {
                                        "format": "jpeg",
                                        "source": {"bytes": buf.getvalue()}
                                    }
                                },
                                {"text": PROMPT}
                            ]
                        }],
                        inferenceConfig={"maxTokens": 10, "temperature": 0.0}
                    )

                    r = response['output']['message']['content'][0]['text'].strip().upper()
                    if "YES" in r:    return fp_str, "YES"
                    elif "NO" in r:   return fp_str, "NO"
                    else:             return fp_str, "UNSURE"

                except Exception as e:
                    err = str(e).lower()
                    if "throttling" in err or "too many" in err or "429" in err or "503" in err:
                        time.sleep(backoff + _random.uniform(1, 3))
                        retries -= 1
                        backoff *= 2
                    elif "quota" in err or "limit exceeded" in err:
                        return fp_str, "QUOTA_EXHAUSTED"
                    else:
                        return fp_str, "ERROR"
            return fp_str, "TIMEOUT"

        results = {}
        quota_exhausted = False
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(check_one, fp): fp for fp in image_paths}
            for fut in as_completed(futures):
                fp_str, ans = fut.result()
                if ans == "QUOTA_EXHAUSTED":
                    quota_exhausted = True
                    ans = "UNSURE"
                results[fp_str] = ans

        if quota_exhausted:
            print(f"  HEM Claude: quota exhausted — unverified images marked UNSURE (boost kept)")
        return results

    def update(self, model, dataset, device, batch_size=56, threshold=0.5):
        """Run inference on training set to find hard examples.
        Uses single GPU (cuda:0) with small batch to avoid OOM.
        DataParallel is unwrapped here — HEM doesn't need multi-GPU.

        `threshold` must match the boundary the model is actually judged at,
        or HEM boosts examples that are not really errors. That used to mean
        passing the validation-swept opt_thresh; since 2026-09-02 it means
        FIXED_THRESHOLD (0.50), because argmax on device IS 0.50 and no other
        threshold can be deployed -- see the FIXED_THRESHOLD comment at the top.
        """
        single_model = model.module if isinstance(model, nn.DataParallel) else model
        single_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        single_model = single_model.to(single_device)
        single_model.eval()
        if os.environ.get("OVERRIDE_HEM_CLEAN") == "1":
            dataset = _CleanView(dataset)

        # The full-dataset HEM pass is decode-bound (GPU ~20%) and took 90 min
        # to 2h15m with 4 workers on the loaded node (26-27/09). Keep the total
        # with the training loader's workers under ~16: NFS lock hangs above that.
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=int(os.environ.get("OVERRIDE_HEM_WORKERS", 4)),
                            pin_memory=False, persistent_workers=False)

        idx = 0
        fp_count, fn_count = 0, 0
        # Track misclassified indices and paths for Gemini verification
        misclassified = []  # list of (dataset_idx, fp_str, label)

        with torch.no_grad():
            for images, labels in loader:
                images = novaic_preprocess_gpu(images.to(single_device))
                probs = torch.softmax(_logits(single_model(images)), dim=1)[:, 1].cpu()
                preds = (probs >= threshold).long()
                bs = labels.size(0)

                fp_mask = ((preds == 1) & (labels == 0))
                fn_mask = ((preds == 0) & (labels == 1))

                if self.gemini_verify or self.cache_only:
                    for i in range(bs):
                        if fp_mask[i] or fn_mask[i]:
                            misclassified.append((
                                idx + i,
                                str(dataset.images[idx + i]),
                                int(labels[i])
                            ))
                else:
                    self.fp_counts[idx:idx+bs] += fp_mask.numpy().astype(np.float32)
                    self.fn_counts[idx:idx+bs] += fn_mask.numpy().astype(np.float32)

                fp_count += fp_mask.sum().item()
                fn_count += fn_mask.sum().item()
                idx += bs

        if (self.gemini_verify or self.cache_only) and misclassified:
            # Separate into: needs API call vs already reliably cached
            RECHECK_STATUSES = {"TIMEOUT", "ERROR", "QUOTA_EXHAUSTED"}

            # Async verification (OVERRIDE_HEM_ASYNC=1, cache-only runs): a separate
            # process (hem_gemini_async.py) asks Gemini about the false positives
            # this round could not verify and writes its verdicts to a side cache,
            # read back here every round. Nothing blocks training: Bedrock's
            # 10 req/min quota stalled a round for 4 h, and Gemini through agy
            # manages ~400 images/h. Only FALSE POSITIVES are queued and only a
            # Gemini "YES" is ever recorded: batched Gemini's "yes" holds but its
            # "no" does not, and no judge is reliable on the FN side (Bedrock
            # says "no bear" to 97% of screen-captured bear frames).
            async_on = os.environ.get("OVERRIDE_HEM_ASYNC") == "1"
            async_cache = Path("hem_async_gemini_cache.pkl")
            if async_on and async_cache.exists():
                try:
                    with open(async_cache, "rb") as f:
                        side = pickle.load(f)
                    n_new = 0
                    for k, v in side.items():
                        cur = self.gemini_cache.get(k)
                        if cur is None or str(cur).upper() in RECHECK_STATUSES | {"UNSURE"}:
                            self.gemini_cache[k] = v
                            n_new += 1
                    print(f"  HEM async: {len(side)} side verdicts, {n_new} newly applied")
                except Exception as e:      # a half-written file must not kill training
                    print(f"  HEM async: side cache unreadable ({e}), skipped this round")

            uncached = []
            for _, fp, _ in misclassified:
                cached_ans = self.gemini_cache.get(fp)
                if cached_ans is None or str(cached_ans).upper() in RECHECK_STATUSES:
                    uncached.append(fp)

            if async_on:
                pending = [fp for _, fp, label in misclassified
                           if label == 0 and (self.gemini_cache.get(fp) is None or
                                              str(self.gemini_cache.get(fp)).upper() in RECHECK_STATUSES)]
                tmp = Path("hem_async_queue.json.tmp")
                tmp.write_text(json.dumps({"written": time.strftime("%Y-%m-%d %H:%M:%S"),
                                           "fp_pending": pending}))
                os.replace(tmp, "hem_async_queue.json")
                print(f"  HEM async: queued {len(pending)} unverified false positives for Gemini")

            if uncached and self.gemini_verify:
                print(f"  HEM: Verifying {len(uncached)} with Claude "
                      f"({len(misclassified) - len(uncached)} cached, "
                      f"{sum(1 for _, fp, _ in misclassified if str(self.gemini_cache.get(fp,'')).upper() in RECHECK_STATUSES)} recheck TIMEOUT/ERROR)...")
                new_results = self._gemini_check_batch(uncached)
                self.gemini_cache.update(new_results)
                with open(self.gemini_cache_file, 'wb') as f:
                    pickle.dump(self.gemini_cache, f)
            elif self.cache_only:
                print(f"  HEM [cache-only]: {len(misclassified) - len(uncached)} cached verdicts, "
                      f"{len(uncached)} uncached -> UNSURE (kept, NO API call)")

            # Apply boost only to correctly-labeled misclassified
            confirmed_mislabel = 0
            for ds_idx, fp_str, label in misclassified:
                ans = self.gemini_cache.get(fp_str, "UNSURE")
                # Correctly labeled: Gemini agrees with folder label
                # bear (label=1) + Gemini=YES -> model wrong, boost
                # non_bear (label=0) + Gemini=NO -> model wrong, boost
                # Mislabeled: Gemini disagrees -> don't boost, zero out
                gemini_agrees = (label == 1 and ans == "YES") or \
                                (label == 0 and ans == "NO")
                gemini_disagrees = (label == 1 and ans == "NO") or \
                                   (label == 0 and ans == "YES")

                if gemini_disagrees:
                    self.fp_counts[ds_idx] = 0  # suppress mislabeled
                    self.fn_counts[ds_idx] = 0
                    confirmed_mislabel += 1
                elif gemini_agrees or ans == "UNSURE":
                    # label is fixed per sample, so exactly one of these applies
                    is_fp = (label == 0)  # non_bear misclassified as bear
                    if is_fp:
                        self.fp_counts[ds_idx] += 1.0
                    else:
                        self.fn_counts[ds_idx] += 1.0

            print(f"  HEM Gemini: {confirmed_mislabel} mislabeled suppressed "
                  f"(not boosted), {len(misclassified) - confirmed_mislabel} boosted")
        else:
            pass  # already updated in loop above

        self.fp_counts *= self.decay
        self.fn_counts *= self.decay
        print(f"  HEM: FP={fp_count}, FN={fn_count} (FP boosted 3x heavier)")
        return fp_count + fn_count

    def get_boosted_weights(self, base_weights):
        """Return updated sampling weights with boosted hard examples.

        fp_counts and fn_counts are tracked in separate arrays (see __init__),
        so which boost factor applies is known directly per-sample instead of
        being guessed from a decayed aggregate's magnitude -- a sample's
        ground-truth label never changes, so it can only ever accumulate one
        of the two (fp_counts if label=non_bear, fn_counts if label=bear).
        """
        boosted = base_weights.copy()
        fp_boost_factor = 1.0 + np.minimum(self.fp_counts, 1.0) * (self.fp_boost - 1.0)
        fn_boost_factor = 1.0 + np.minimum(self.fn_counts, 1.0) * (self.fn_boost - 1.0)
        boost = fp_boost_factor * fn_boost_factor  # only one factor is ever > 1.0 per sample
        hard_mask = (self.fp_counts > 0.01) | (self.fn_counts > 0.01)
        if hard_mask.any():
            boosted[hard_mask] *= boost[hard_mask]
            n_fp = int((self.fp_counts > 0.01).sum())
            n_fn = int((self.fn_counts > 0.01).sum())
            print(f"  HEM: Boosting {hard_mask.sum()} hard examples "
                  f"({n_fp} FP, {n_fn} FN, max: {boost.max():.1f}x)")
        return boosted


# ============================================================
# NOVAIC PREPROCESSING ON GPU
# ============================================================
def novaic_preprocess_gpu(images):
    """
    Apply Novaic BGR preprocessing on GPU (batch operation).
    Input: RGB tensor [B, 3, H, W] on device, range [0, 1]
    Output: BGR tensor with mean subtracted
    """
    images = images[:, [2, 1, 0], :, :]  # RGB -> BGR
    images = images.clone()
    images[:, 0] -= 104 / 255  # B
    images[:, 1] -= 116 / 255  # G
    images[:, 2] -= 124 / 255  # R
    return images


# ============================================================
# LOGITS HELPER — normalizes BearNet dual-output to logits-only
# ============================================================
def _logits(o):
    """If BearNet returns (logits, emb) when projection is set, extract logits.
    When --supcon is OFF, BearNet returns a plain tensor; this is a no-op."""
    return o[0] if isinstance(o, tuple) else o


# ============================================================
# TRAINING & VALIDATION
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device, epoch, use_mixup=True):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    # BatchNorm updates its running stats DURING the forward pass, before any
    # loss or gradient check can run. One non-finite batch therefore wrote NaN
    # into every running_mean/var and the model became a constant predictor in
    # eval -- the 05/10 MobileNetV2 and RegNetX pilots (batch 24) both died in
    # epoch 1 this way, with finite weights. Snapshot the buffers each step and
    # put them back when the step is rejected.
    buffers = [b for b in model.buffers() if b.is_floating_point()]
    n_rejected = 0

    for images, labels in tqdm(loader, desc=f"Epoch {epoch}"):
        images, labels = images.to(device), labels.to(device)
        images = novaic_preprocess_gpu(images)  # RGB->BGR on GPU
        optimizer.zero_grad()
        saved = [b.detach().clone() for b in buffers]

        def reject():
            nonlocal n_rejected
            with torch.no_grad():
                for b, v in zip(buffers, saved):
                    b.copy_(v)
            optimizer.zero_grad()
            n_rejected += 1

        if use_mixup and random.random() < 0.3:
            mixed_images, labels_a, labels_b, lam = mixup_data_same_class(images, labels, alpha=0.2)
            out = model(mixed_images)                        # BearNet: logits or (logits, emb)
            if isinstance(out, tuple):
                outputs, emb = out
            else:
                outputs, emb = out, None
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            # Add SupCon on original labels (same-class mixup keeps labels valid)
            if emb is not None:
                loss = loss + _args.supcon_weight * supcon_loss(emb, labels, temperature=0.1)
            # Use mixed output for accuracy (not original) to avoid double-counting
            correct += outputs.argmax(1).eq(labels).sum().item()
        else:
            out = model(images)                              # BearNet: logits or (logits, emb)
            if isinstance(out, tuple):
                outputs, emb = out
            else:
                outputs, emb = out, None
            loss = criterion(outputs, labels)
            if emb is not None:
                loss = loss + _args.supcon_weight * supcon_loss(emb, labels, temperature=0.1)
            correct += outputs.argmax(1).eq(labels).sum().item()

        if not torch.isfinite(loss) or not torch.isfinite(outputs).all():
            reject()
            continue

        loss.backward()

        # Safety: skip update if gradients are nan/inf
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            reject()
            continue

        optimizer.step()
        loss_sum += loss.item()
        total += labels.size(0)

    if n_rejected:
        print(f"  WARNING: {n_rejected} non-finite batches rejected this epoch "
              f"(BN running stats restored)")
    return loss_sum / len(loader), 100. * correct / total


# OVERRIDE_SELECT_LIST=<txt of val image paths>: the "novel" val subset (no
# train image above CLIP cosine 0.90 after the 09/10 scene re-split). validate()
# then also reports novel_f1/_recall/_fpr, and OVERRIDE_SELECT=novel_f1 makes it
# the checkpoint-selection metric. Overall val F1 was shown to mostly measure
# memory of near-duplicates (eval_novel_val.py).
NOVEL_MASK = None
SELECT_KEY = os.environ.get("OVERRIDE_SELECT", "fbeta")
ACT_MEAN_MAX = float(os.environ.get("OVERRIDE_ACT_MEAN_MAX", "inf"))


def validate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    loss_sum = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            images = novaic_preprocess_gpu(images)  # RGB->BGR on GPU
            outputs = _logits(model(images))
            loss_sum += criterion(outputs, labels).item()
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= threshold).astype(int)

    tp = int(((preds == 1) & (all_labels == 1)).sum())
    tn = int(((preds == 0) & (all_labels == 0)).sum())
    fp = int(((preds == 1) & (all_labels == 0)).sum())
    fn = int(((preds == 0) & (all_labels == 1)).sum())

    precision = float(100. * tp / (tp + fp)) if tp + fp else 0.0
    recall = float(100. * tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    BETA = METRIC_BETA
    fbeta = float((1+BETA**2) * precision * recall / (BETA**2 * precision + recall)) if (precision + recall) > 0 else 0.0
    # Also always compute F0.5 for reference
    f05 = float((1+0.5**2) * precision * recall / (0.5**2 * precision + recall)) if (precision + recall) > 0 else 0.0

    novel = {}
    if NOVEL_MASK is not None and len(NOVEL_MASK) == len(preds):
        m = NOVEL_MASK
        ntp = int(((preds == 1) & (all_labels == 1) & m).sum())
        nfp = int(((preds == 1) & (all_labels == 0) & m).sum())
        nfn = int(((preds == 0) & (all_labels == 1) & m).sum())
        nneg = int(((all_labels == 0) & m).sum())
        nr = 100. * ntp / max(1, ntp + nfn)
        npr = 100. * ntp / max(1, ntp + nfp)
        novel = {'novel_recall': float(nr), 'novel_precision': float(npr),
                 'novel_f1': float(2 * nr * npr / (nr + npr)) if nr + npr > 0 else 0.0,
                 'novel_fpr': float(100. * nfp / max(1, nneg))}
    return {
        **novel,
        'loss': float(loss_sum / len(loader)),
        'accuracy': float(100. * (tp + tn) / len(all_labels)),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f05': f05,
        'fbeta': fbeta,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'threshold': float(threshold)
    }


# ============================================================
# ACTIVATION OUTLIER MONITOR
# ============================================================
def measure_activation_outliers(model, loader, device, n_batches=6):
    """Per-ReLU  max|a| / percentile-99.9  on a few validation batches.

    This is an early-warning signal for post-training quantization, which is
    how the model is actually deployed. A high ratio means a handful of extreme
    activations set the tensor's range, so a min/max calibrator (what the Novaic
    compiler does) stretches the scale to fit them and crushes the real signal
    into a few INT8 levels.

    It is not hypothetical: the 26/08 model scored 44.5 at features.10.conv.1.2
    versus 4.0-5.5 for the two models before it, was fine in FP32, and collapsed
    on device into a constant predictor. FP32 validation cannot see this coming,
    so measure it every epoch instead of discovering it after deployment.

    Rule of thumb from the models measured so far: <10 healthy, 10-20 watch,
    >20 expect PTQ trouble.
    """
    m = model.module if isinstance(model, nn.DataParallel) else model
    m.eval()
    stats = {}

    def hook(name):
        def fn(_mod, _inp, out):
            a = out.detach().abs().flatten().float()
            if a.numel() == 0:
                return
            # subsample before quantile: torch.quantile caps input size
            s = a[:: max(1, a.numel() // 100000)]
            mx = a.max().item()
            p999 = torch.quantile(s, 0.999).item()
            prev = stats.get(name, (0.0, 0.0))
            stats[name] = (max(prev[0], mx), max(prev[1], p999))
        return fn

    handles = [mod.register_forward_hook(hook(n))
               for n, mod in m.named_modules() if isinstance(mod, nn.ReLU)]
    try:
        with torch.no_grad():
            for i, (images, _) in enumerate(loader):
                if i >= n_batches:
                    break
                m(novaic_preprocess_gpu(images.to(device)))
    finally:
        for h in handles:
            h.remove()

    ratios = {n: (mx / p if p > 0 else 0.0) for n, (mx, p) in stats.items()}
    if not ratios:
        return {'max_ratio': 0.0, 'worst_layer': '', 'mean_ratio': 0.0}
    worst = max(ratios, key=ratios.get)
    return {
        'max_ratio': float(ratios[worst]),
        'worst_layer': worst,
        'mean_ratio': float(np.mean(list(ratios.values()))),
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME} @ {INPUT_SIZE_STR}")
    print(f"CRITICAL: ReLU (NO ReLU6/Clip)")
    print(f"Preprocessing: NOVAIC-COMPATIBLE (BGR, mean=[104,116,124]/255)")
    print(f"Loss: Focal(a={FOCAL_ALPHA}, g={FOCAL_GAMMA}, ls={LABEL_SMOOTHING}) + 0.3*SoftF1(symmetric) [margin DISABLED]")
    print(f"Metric: F{METRIC_BETA} (balanced recall/precision), threshold range 0.30-0.95")
    print(f"Config: F1-balanced, asymmetric HEM (FP {5.0}x / FN {3.0}x), temperature scaling")
    print(f"{'='*60}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = BearDatasetTrain(TRAIN_DIR, INPUT_SIZE)
    val_dataset = BearDatasetVal(VAL_DIR, INPUT_SIZE)
    if os.environ.get("OVERRIDE_SELECT_LIST"):
        novel_names = {Path(l.strip()).name for l in open(os.environ["OVERRIDE_SELECT_LIST"]) if l.strip()}
        NOVEL_MASK = np.array([Path(p).name in novel_names for p in val_dataset.images])
        print(f"  Novel val subset: {int(NOVEL_MASK.sum()):,} of {len(NOVEL_MASK):,} val images "
              f"(bear {int(sum(1 for p, l in zip(NOVEL_MASK, val_dataset.labels) if p and l == 1)):,}); "
              f"selection metric: {SELECT_KEY}, act_mean cap {ACT_MEAN_MAX}")

    labels = np.array(train_dataset.get_labels_for_sampler())
    base_weights = (1.0 / np.bincount(labels))[labels].astype(np.float64)
    base_weights = apply_group_weights(train_dataset.images, base_weights,
                                       _args.dark_quad_boost, _args.night_bear_boost)
    sampler = WeightedRandomSampler(base_weights, len(base_weights), replacement=True)

    # Scale batch size by number of GPUs
    effective_batch_size = BATCH_SIZE * max(1, NUM_GPUS)
    print(f"  Batch size: {BATCH_SIZE} x {max(1, NUM_GPUS)} GPUs = {effective_batch_size}")

    loader_kwargs = {
        'num_workers': NUM_WORKERS,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': False,
    }
    if NUM_WORKERS > 0:
        loader_kwargs['prefetch_factor'] = 3

    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=effective_batch_size, shuffle=False, **loader_kwargs)

    _backbone = create_model()
    _proj = ProjectionHead() if _args.supcon else None
    model = BearNet(_backbone, _proj).to(device)
    if _args.supcon:
        print(f"  SupCon ENABLED: weight={_args.supcon_weight}, "
              f"boosts dark_quad={_args.dark_quad_boost} night_bear={_args.night_bear_boost}")

    # Hard Example Mining - asymmetric: FP boosted 3x heavier than FN
    # Env overrides let a fast fine-tune disable HEM (set HEM_START_EPOCH huge):
    # the per-HEM-epoch full-dataset inference pass (~112k imgs) is the dominant
    # cost on short runs and is not needed just to test an augmentation change.
    HEM_START_EPOCH = int(os.environ.get("OVERRIDE_HEM_START", 5))
    HEM_FREQUENCY = int(os.environ.get("OVERRIDE_HEM_FREQ", 3))
    cache_only_verify = _args.cache_only_verify
    # cache-only takes precedence; otherwise Claude verify unless --no-verify
    use_claude_verify = (not _args.no_claude_verify) and (not cache_only_verify)
    if cache_only_verify:
        print("  HEM: CACHE-ONLY verify (--cache-only-verify) — uses hem_gemini_cache.pkl "
              "verdicts to suppress known mislabels; NO Claude API calls (no AWS).")
    elif not use_claude_verify:
        print("  HEM: Claude/Bedrock verification DISABLED (--no-verify) — "
              "boosting all misclassified examples, no cache, no mislabel-checking.")
    hem = HardExampleMiner(
        num_samples=len(train_dataset), fp_boost=5.0, fn_boost=3.0,
        decay=0.7, gemini_verify=use_claude_verify, cache_only=cache_only_verify,
        gemini_cache_file="hem_gemini_cache.pkl",
        extra_caches=[
            "hard_examples_gemini_cache.pkl",
            "random_check_cache.pkl",
        ],
    )

    criterion = CombinedLoss(
        focal_alpha=FOCAL_ALPHA,
        focal_gamma=FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTHING,
        f1_weight=0.3,
        margin_weight=0.0  # DISABLED: prevents threshold from working effectively
    )

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    start_epoch, best_f1 = 1, 0.0
    best_threshold = 0.5
    early_stopping_counter = 0
    best_score_any = 0.0      # best selection score regardless of the act_mean cap

    if RESUME_FROM and Path(RESUME_FROM).exists():
        ckpt = torch.load(RESUME_FROM, map_location=device)
        model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()})
        optimizer.load_state_dict(ckpt.get('optimizer_state_dict', optimizer.state_dict()))
        start_epoch = ckpt['epoch'] + 1
        best_f1 = ckpt.get('best_f1', 0)
        best_score_any = best_f1
        best_threshold = ckpt.get('best_threshold', 0.5)
        print(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.2f}%\n")

    # Wrap with DataParallel AFTER loading checkpoint (before training)
    if NUM_GPUS > 1:
        model = nn.DataParallel(model)
        print(f"  DataParallel: using {NUM_GPUS} GPUs")

    # Helper to get underlying model (with or without DataParallel)
    def get_model():
        return model.module if isinstance(model, nn.DataParallel) else model

    history = []
    start_time = time.time()

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, use_mixup=True)

        # Fixed at the device's actual operating point -- see FIXED_THRESHOLD.
        # (find_optimal_threshold() is kept for diagnostics but must not drive
        # model selection or HEM: a threshold we cannot deploy is not a knob.)
        opt_thresh = FIXED_THRESHOLD
        val = validate(model, val_loader, criterion, device, threshold=opt_thresh)

        scheduler.step(val['fbeta'])

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val: Prec={val['precision']:.2f}%, Recall={val['recall']:.2f}%, F1={val['f1']:.2f}%, F0.5={val['f05']:.2f}%, F{METRIC_BETA}={val['fbeta']:.2f}%")
        print(f"  Threshold={opt_thresh:.2f} | LR={current_lr:.6f}")
        print(f"  TP={val['tp']}, TN={val['tn']}, FP={val['fp']}, FN={val['fn']}")

        # PTQ early-warning -- see measure_activation_outliers().
        act = measure_activation_outliers(model, val_loader, device)
        flag = ('  <-- CANH BAO: PTQ co the sup do' if act['max_ratio'] > 20 else
                '  (theo doi)' if act['max_ratio'] > 10 else '')
        print(f"  Act outlier: max={act['max_ratio']:.1f} @ {act['worst_layer']}"
              f"  mean={act['mean_ratio']:.1f}{flag}")

        # Hard Example Mining: update sampler weights periodically
        if epoch >= HEM_START_EPOCH and epoch % HEM_FREQUENCY == 0:
            hem.update(model, train_dataset, device, threshold=opt_thresh)
            boosted_weights = hem.get_boosted_weights(base_weights)
            sampler = WeightedRandomSampler(boosted_weights, len(boosted_weights), replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, sampler=sampler, **loader_kwargs)

        history.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'threshold': float(opt_thresh),
            'act_max_ratio': act['max_ratio'],
            'act_worst_layer': act['worst_layer'],
            'act_mean_ratio': act['mean_ratio'],
            **val
        })

        with open(LOG_DIR / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        score = val.get(SELECT_KEY, val['fbeta'])
        if 'novel_f1' in val:
            print(f"  Novel val: R={val['novel_recall']:.2f}%  P={val['novel_precision']:.2f}%  "
                  f"F1={val['novel_f1']:.2f}%  FPR={val['novel_fpr']:.2f}%  [select on {SELECT_KEY}]")
        # Early stopping follows the selection metric whatever act_mean says;
        # the act_mean cap only decides whether best_model.pth is written. (The
        # first V19 run, 09/10, kept act_mean ~11-12 on RegNetX, so a cap that
        # also froze best_f1 would never have let early stopping fire.)
        improved = score > best_score_any
        if improved:
            best_score_any = score
        if score > best_f1 and act['mean_ratio'] >= ACT_MEAN_MAX:
            print(f"  (not saved as best: act_mean {act['mean_ratio']:.1f} >= {ACT_MEAN_MAX})")
        if score > best_f1 and act['mean_ratio'] < ACT_MEAN_MAX:
            best_f1 = score
            best_threshold = opt_thresh
            early_stopping_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': get_model().state_dict(),
                'best_f1': best_f1,
                'best_threshold': best_threshold,
                'config': {
                    'backbone': BACKBONE or 'torchvision_mobilenet_v2',
                    'activation': 'ReLU',
                    'mean_bgr': [104, 116, 124],
                    'scale': 255,
                    'preprocessing': 'novaic_compatible',
                }
            }, MODEL_DIR / 'best_model.pth')
            print(f"  >>> Best model! F1: {best_f1:.2f}% @ threshold={best_threshold:.2f}")

        if epoch % CHECKPOINT_FREQUENCY == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': get_model().state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'best_threshold': best_threshold,
                'config': {'backbone': BACKBONE or 'torchvision_mobilenet_v2'},
            }, CHECKPOINT_DIR / f'checkpoint_epoch{epoch}.pth')

        # Early stopping (same metric as checkpoint selection, ignoring the cap)
        if not improved:
            early_stopping_counter += 1
            if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n  Early stopping: no improvement for {EARLY_STOPPING_PATIENCE} epochs.")
                print(f"  Best F{METRIC_BETA}: {best_f1:.2f}% @ epoch {epoch - EARLY_STOPPING_PATIENCE}")
                break
        else:
            early_stopping_counter = 0

        gc.collect()
        # NOTE: intentionally NOT calling torch.cuda.empty_cache() here.
        # empty_cache() returns memory to the OS pool, allowing other users
        # to allocate it. Keeping memory allocated prevents mid-training OOM
        # from GPU sharing. PyTorch's caching allocator reuses it internally.

    # Save config
    with open(MODEL_DIR / 'config.json', 'w') as f:
        json.dump({
            'best_f1': best_f1,
            'best_threshold': float(best_threshold),
            'model': MODEL_NAME,
            'input_size': INPUT_SIZE_STR,
            'preprocessing': 'novaic_compatible',
            'mean_bgr': [104, 116, 124],
            'scale': 255,
            'activation': 'ReLU (NO ReLU6/Clip)',
            'focal_gamma': FOCAL_GAMMA,
            'focal_alpha': FOCAL_ALPHA,
            'f_beta': METRIC_BETA,
            'confidence_margin': False,
        }, f, indent=2)

    # ============================================================
    # TEMPERATURE SCALING (Post-training probability calibration)
    # ============================================================
    # Learn a single parameter T to calibrate probabilities on val set.
    # Does NOT change model weights. Only divides logits by T before softmax.
    # Makes probability reflect true confidence -> threshold cuts more accurately.
    print(f"\n{'='*60}")
    print(f"Temperature Scaling (probability calibration)")
    print(f"{'='*60}")

    class TemperatureScaler(nn.Module):
        def __init__(self):
            super().__init__()
            self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        def forward(self, logits):
            return logits / self.temperature

    temp_scaler = TemperatureScaler().to(device)
    temp_optimizer = optim.LBFGS([temp_scaler.temperature], lr=0.01, max_iter=50)
    nll_criterion = nn.CrossEntropyLoss()

    # Collect all val logits
    get_model().eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = novaic_preprocess_gpu(images.to(device))
            logits = _logits(get_model()(images))
            all_logits.append(logits)
            all_targets.append(labels.to(device))
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    # Optimize temperature
    def temp_eval():
        temp_optimizer.zero_grad()
        scaled = temp_scaler(all_logits)
        loss = nll_criterion(scaled, all_targets)
        loss.backward()
        return loss

    temp_optimizer.step(temp_eval)
    optimal_temp = temp_scaler.temperature.item()
    print(f"  Optimal temperature: {optimal_temp:.4f}")

    # Re-find threshold with calibrated probabilities
    calibrated_probs = torch.softmax(all_logits / optimal_temp, dim=1)[:, 1].cpu().numpy()
    cal_labels = all_targets.cpu().numpy()

    BETA = METRIC_BETA
    best_cal_score, best_cal_thresh = 0, 0.5
    for thresh in np.arange(0.30, 0.96, 0.01):
        preds = (calibrated_probs >= thresh).astype(int)
        tp = ((preds == 1) & (cal_labels == 1)).sum()
        fp = ((preds == 1) & (cal_labels == 0)).sum()
        fn = ((preds == 0) & (cal_labels == 1)).sum()
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        fb = (1 + BETA**2) * pr * rc / (BETA**2 * pr + rc) if pr + rc else 0
        if fb > best_cal_score:
            best_cal_score = fb
            best_cal_thresh = thresh

    # DIAGNOSTIC ONLY. The device decides by argmax (= 0.50) and the threshold
    # cannot be baked into the exported ONNX, so this calibrated value must not
    # be used to pick a model or to set a deployment threshold -- it only shows
    # how far the calibrated probabilities sit from the fixed operating point.
    print(f"  Calibrated threshold: {best_cal_thresh:.2f} (F{METRIC_BETA}={best_cal_score:.4f}) "
          f"[DIAGNOSTIC ONLY -- deployment runs at {FIXED_THRESHOLD:.2f}]")
    print(f"  Before calibration: threshold={best_threshold:.2f}")

    # Save temperature
    torch.save({'temperature': optimal_temp, 'threshold': best_cal_thresh},
               MODEL_DIR / 'temperature_scaling.pth')

    # Update config with calibrated values
    with open(MODEL_DIR / 'config.json', 'r') as f:
        config = json.load(f)
    config['temperature'] = optimal_temp
    config['calibrated_threshold'] = float(best_cal_thresh)
    with open(MODEL_DIR / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  Saved: {MODEL_DIR / 'temperature_scaling.pth'}")
    CALIB_DIR = WORKING_DIR / "calibration_images"
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    CALIB_COUNT = 500  # Total calibration images (50% hard + 50% random)

    print(f"\nExporting HYBRID calibration images to {CALIB_DIR}...")

    resize = ResizeWithPadding(INPUT_SIZE)
    train_images = train_dataset.images
    train_labels = train_dataset.labels

    # fp_counts and fn_counts never overlap for a given sample (fixed
    # ground-truth label), so summing them is a safe combined hardness score
    hard_scores = hem.fp_counts + hem.fn_counts

    bear_indices = [i for i, l in enumerate(train_labels) if l == 1]
    nonbear_indices = [i for i, l in enumerate(train_labels) if l == 0]

    # Sort by difficulty (descending)
    bear_sorted = sorted(bear_indices, key=lambda i: hard_scores[i], reverse=True)
    nonbear_sorted = sorted(nonbear_indices, key=lambda i: hard_scores[i], reverse=True)

    # 50 hard + 50 random per class = 100 per class = 200 total... scaled to 500
    num_hard_per_class = CALIB_COUNT // 4   # 125
    num_rand_per_class = CALIB_COUNT // 4   # 125

    hard_bear = bear_sorted[:num_hard_per_class]
    hard_nonbear = nonbear_sorted[:num_hard_per_class]

    remain_bear = bear_sorted[num_hard_per_class:]
    remain_nonbear = nonbear_sorted[num_hard_per_class:]

    random.seed(42)
    rand_bear = random.sample(remain_bear, min(num_rand_per_class, len(remain_bear)))
    rand_nonbear = random.sample(remain_nonbear, min(num_rand_per_class, len(remain_nonbear)))

    selected = hard_bear + hard_nonbear + rand_bear + rand_nonbear
    random.shuffle(selected)

    hard_set = set(hard_bear + hard_nonbear)
    calib_count = 0
    hard_count = 0
    for idx in selected:
        try:
            img = Image.open(train_images[idx]).convert('RGB')
            img = resize(img)
            label_str = "bear" if train_labels[idx] == 1 else "non_bear"
            is_hard = idx in hard_set
            type_str = "hard" if is_hard else "rand"
            if is_hard:
                hard_count += 1
            dst = CALIB_DIR / f"calib_{calib_count:04d}_{label_str}_{type_str}.jpg"
            img.save(dst, quality=95)
            calib_count += 1
        except Exception:
            continue

    print(f"  Exported {calib_count} calibration images:")
    print(f"  - {hard_count} Hard examples (top misclassified by HEM)")
    print(f"  - {calib_count - hard_count} Random examples (core distribution)")
    print(f"  Size: {INPUT_SIZE_STR} (resized with padding)")
    print(f"  Format: RGB JPG (Novaic applies BGR + mean subtraction)")

    print(f"\n{'='*60}")
    print(f"Done! Best F1: {best_f1:.2f}% @ threshold={best_threshold:.2f}")
    print(f"Time: {(time.time()-start_time)/3600:.1f}h")
    print(f"Next: python train_qat_mobilenetv2_relu_novaic.py")
    print(f"{'='*60}")