"""
Collect bear images by scenario using SAHI + Multi-GPU + yt-dlp + BioCLIP (Strict Priority Logic).
"""

import os
import argparse
import shutil
import subprocess
import sys
import math
import time
import random
import multiprocessing as mp
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

os.environ["TORCHDYNAMO_DISABLE"] = "1"

if sys.platform.startswith("linux"):
    TMPDIR = "/dev/shm/phuocthien_tmp"
else:
    TMPDIR = os.path.join(os.path.expanduser("~"), "tmp_bear_collect")
os.environ["TMPDIR"] = TMPDIR
os.makedirs(TMPDIR, exist_ok=True)

# ============================================================
# SCENARIO CONFIG
# ============================================================
SCENARIO_CONFIG = {
    "night_vision": {
        "queries": [
            "trail camera bear night vision", "bear infrared camera night",
            "black bear night trail cam", "grizzly bear trail camera night",
            "bear caught trail camera night 2024", "bear night vision compilation trail cam",
            "bear IR camera footage night", "bears at night trail camera",
            "brown bear night trail camera", "kodiak bear night camera",
            "cub bear night trail cam", "mama bear cubs night camera",
            "bear foraging night trail cam", "bear prowling night camera",
            "bear feeder night camera", "bear bait station night camera",
            "bear visiting camera at night", "bear sniffing trail camera night",
            "bear night forest encounter camera", "trail camera bear after dark",
            "bear in dark woods camera", "bear nocturnal trail camera",
            "bear midnight trail camera", "bear pre-dawn trail cam",
            "bear night camera 2025", "wildlife night camera black bear",
        ],
        "filter": "night",
        "max_per_query": 12,
    },
    "IR": {
        "queries": [
            "bear infrared thermal camera", "trail camera bear IR illumination",
            "bear thermal imaging wildlife", "infrared bear detection camera",
            "bear trail cam infrared night 2024",
            "FLIR bear wildlife footage", "thermal bear tracking hunting",
            "bear heat signature camera", "bear thermal scope hunting",
            "thermal imaging bear at night", "bear observation thermal camera",
            "IR wildlife camera bear compilation", "infrared game camera bear",
            "bear night vision thermal comparison", "bear black-flash IR camera",
            "bear no-glow infrared trail cam", "bear red glow IR camera",
            "thermal drone bear wildlife", "bear thermal sighting forest",
        ],
        "filter": "night",
        "max_per_query": 11,
    },
    "snow_conditions": {
        "queries": [
            "bear trail camera snow winter", "bear in snow caught on camera",
            "grizzly bear snow trail cam", "bear winter wildlife camera",
            "polar bear trail camera snow", "bear snow forest camera 2024",
            "bear walking in snow trail cam",
            "brown bear snow trail cam", "kodiak bear snow winter",
            "black bear snowy forest camera", "grizzly winter footage camera",
            "bear tracks in snow camera", "bear emerging from den snow",
            "bear hibernation camera winter", "bear snowstorm trail camera",
            "bear snowfall wildlife cam", "bear den snow entrance camera",
            "bear snow mountain wildlife", "bear alpine snow trail cam",
            "bear snowy meadow camera", "bear snow river trail camera",
            "polar bear arctic camera footage", "bear winter landscape camera",
        ],
        "filter": "snow",
        "max_per_query": 12,
    },
    "low_light": {
        "queries": [
            "bear trail camera dusk dawn", "bear low light wildlife camera",
            "bear twilight trail cam", "bear dim light camera footage",
            "bear early morning trail camera",
            "bear dawn trail camera", "bear sunset trail cam",
            "bear sunrise wildlife camera", "bear golden hour trail cam",
            "bear evening trail camera", "bear late afternoon camera",
            "bear overcast trail camera", "bear cloudy day trail cam",
            "bear rainy day trail camera", "bear foggy morning camera",
            "bear misty forest trail cam", "bear stormy weather camera",
            "bear shadow forest camera", "bear deep woods trail cam",
            "bear canopy shade wildlife camera", "bear dim forest trail camera",
        ],
        "filter": "dark",
        "max_per_query": 11,
    },
    "distance_10m": {
        "queries": [
            "distant bear trail camera", "bear far away forest camera",
            "bear long distance wildlife cam", "small bear trail camera footage",
            "bear meadow far trail cam", "bear field distance camera",
            "wide angle bear trail camera", "bear tiny in frame trail cam",
            "bear background trail camera", "bear far view wildlife cam",
            "bear distant mountain camera", "bear crossing distant field camera",
            "bear across valley trail cam", "bear on hillside far camera",
            "bear distant ridge trail camera", "bear opposite bank camera",
            "bear pasture far trail camera", "bear clearing distant camera",
            "bear aerial view wildlife", "drone footage distant bear",
        ],
        "filter": "none",
        "max_per_query": 11,
    },
    "partial_occlusion": {
        "queries": [
            "bear behind trees trail cam", "bear partially hidden camera",
            "bear bushes trail camera", "bear through vegetation camera",
            "bear behind log trail camera", "bear behind rock trail cam",
            "bear in tall grass camera", "bear through branches trail cam",
            "bear half hidden forest camera", "bear obscured vegetation camera",
            "bear peeking trail camera", "bear in thicket camera",
            "bear in brush trail camera", "bear hidden trees wildlife cam",
            "bear through foliage trail cam", "bear tall grass trail cam",
            "bear ferns forest camera", "bear underbrush trail camera",
            "bear behind fence camera", "bear partially visible trail cam",
            "bear edge of frame trail cam",
        ],
        "filter": "none",
        "max_per_query": 11,
    },
    "in_urban": {
        "queries": [
            "bear backyard security camera", "urban bear caught on camera",
            "bear neighborhood camera footage", "bear dumpster security cam",
            "bear residential area camera",
            "bear ring doorbell camera", "bear nest doorbell bear",
            "bear porch camera night", "bear front door security camera",
            "bear arlo camera bear", "bear wyze camera wildlife",
            "bear garbage can camera", "bear pool backyard camera",
            "bear garage security camera", "bear driveway camera footage",
            "bear deck security camera", "bear on patio camera",
            "bear garden camera footage", "bear in yard security cam",
            "bear street security cam", "bear parking lot camera",
            "bear subdivision camera bear", "bear suburban wildlife camera",
            "bear Tahoe home camera", "bear Colorado backyard camera",
            "bear raiding trash camera", "bear opening door camera",
            "bear bird feeder home camera",
        ],
        "filter": "none",
        "max_per_query": 11,
    },
    "camera_frame": {
        "queries": [
            "bear trail camera screen recording", "bear footage on phone screen",
            "trail cam bear video on monitor", "showing bear trail camera footage on screen",
            "reaction to bear trail cam", "reviewing trail cam bear footage",
            "trail camera review bear", "bear trail cam compilation screen",
            "bear video on laptop screen", "bear footage on tv screen",
            "news report bear camera footage", "bear security footage news",
            "bear video playback monitor", "bear trail cam reaction video",
            "bear camera footage review", "bear found on game camera",
            "bear trail cam found amazing", "bear trail camera youtuber",
            "trail cam bear discovery",
        ],
        "filter": "none",
        "max_per_query": 11,
    },
    "motion_blur": {
        "queries": [
            "bear running trail camera", "fast bear caught on camera",
            "bear charging trail cam", "bear moving fast wildlife camera",
            "bear sprinting trail camera", "bear running full speed camera",
            "bear chase trail cam", "bear fleeing forest camera",
            "bear running away trail cam", "bear bolting wildlife camera",
            "bear jumping trail camera", "bear climbing fast camera",
            "bear leaping trail cam", "bear fighting motion camera",
            "bear play fighting camera", "bear tumbling trail cam",
            "bear chasing deer camera", "bear chasing prey wildlife",
            "bear escape trail camera", "bear startled running camera",
            "bear rushing by trail cam", "bear cubs running trail cam",
        ],
        "filter": "none",
        "max_per_query": 11,
    },
    "rain_fog": {
        "queries": [
            "bear in rain trail camera", "bear rainy day wildlife cam",
            "bear wet fur trail cam", "bear stormy forest camera",
            "bear foggy morning trail cam", "bear in mist wildlife camera",
            "bear through fog forest camera", "bear heavy rain trail cam",
            "bear drizzle wildlife camera", "bear overcast rain camera",
        ],
        "filter": "dark",
        "max_per_query": 12,
    },
    "multi_bear": {
        "queries": [
            "multiple bears trail camera", "bear family trail cam",
            "mother bear cubs trail camera", "bear siblings wildlife camera",
            "two bears trail camera", "bear group forest camera",
            "bear cubs playing trail cam", "mom bear with cubs camera",
            "bears fighting trail camera", "bears meeting trail cam",
        ],
        "filter": "none",
        "max_per_query": 12,
    },
    "bear_cubs_small": {
        "queries": [
            "bear cub trail camera", "bear cubs forest camera",
            "small bear cub wildlife cam", "baby bear trail cam",
            "newborn bear cub camera", "yearling bear trail camera",
            "bear cub climbing tree camera", "bear cub alone trail cam",
            "tiny bear cub wildlife", "bear cub playing camera",
        ],
        "filter": "none",
        "max_per_query": 12,
    },
    "atypical_coat": {
        # 2026-08-10: benchmark_results_MobileNetV2_QAT_v28_videos.csv analysis found the model
        # heavily misses non-black-coat bears (polar bear, spirit/Kermode bear) -- training data is
        # skewed to typical North American black/brown trail-cam bears.
        "queries": [
            "polar bear swimming ocean", "polar bear wildlife camera",
            "polar bear arctic footage", "polar bear hunting seal camera",
            "polar bear walking snow footage", "polar bear close up wildlife",
            "spirit bear kermode bear footage", "white black bear kermode",
            "cinnamon black bear footage", "blonde black bear wildlife camera",
            "brown phase black bear footage", "white phase black bear camera",
            "glacier bear blue bear footage", "albino bear wildlife camera",
            "polar bear documentary footage", "polar bear cub wildlife camera",
        ],
        "filter": "none",
        "max_per_query": 12,
    },
    "aerial_drone": {
        # 2026-08-10: video d1bX6ruurQo (drone overhead shot of a bear at a trash can, well-lit
        # daylight) had only 18.8% recall despite easy lighting -- top-down drone angle is absent
        # from typical eye-level trail-cam training data.
        "queries": [
            "drone footage bear wildlife", "aerial view bear forest",
            "drone bear sighting video", "bear from above drone camera",
            "overhead drone bear footage", "bear top view aerial camera",
            "drone chasing bear video", "bear spotted by drone",
            "aerial drone wildlife bear", "drone footage grizzly bear",
            "drone footage black bear forest", "bear drone news footage",
        ],
        "filter": "none",
        "max_per_query": 12,
    },
    "water_submersion": {
        # 2026-08-10: video XlMZTYqXVBA (bear swimming in a backyard pool at night, only head
        # above water, fisheye lens, colored pool lighting) -- water occlusion is a distinct
        # occlusion type not covered by ComprehensiveOcclusion (foliage/object occlusion only).
        "queries": [
            "bear swimming pool backyard camera", "bear swimming lake footage",
            "bear swimming river trail camera", "bear in water wildlife camera",
            "bear crossing river camera", "bear swimming ocean footage",
            "bear wading water trail cam", "bear pool security camera",
            "bear swimming camera footage", "bear fishing river camera",
            "bear splashing water trail cam", "bear swimming closeup wildlife",
        ],
        "filter": "none",
        "max_per_query": 12,
    },
    "window_glass": {
        # 2026-08-18: FN-table review, video 6bgvjPZvcKc (9 FN) -- filmed from inside a house
        # through a window, bear on the porch/windowsill outside, backlit against bright daylight,
        # window-frame mullions crossing the bear. Distinct from partial_occlusion (foliage/objects)
        # and from in_urban (no glass/backlight combination there).
        "queries": [
            "bear looking through window trail camera", "bear at window house camera",
            "bear peeking through window", "bear on porch seen through window",
            "bear at back door window camera", "bear through screen door camera",
            "bear looking in window house video", "bear outside window backyard camera",
            "bear at window sill camera", "bear seen from inside house window",
            "bear pressed against window glass", "bear through glass door camera",
        ],
        "filter": "none",
        "max_per_query": 12,
    },
    "indoor_store": {
        # 2026-08-18: FN-table review, video xApoL_UdFvM (19 FN) -- bear inside a convenience
        # store, cluttered shelving background, overhead security-cam angle. Distinct from
        # in_urban (outdoor backyard/porch cameras only).
        "queries": [
            "bear inside store security camera", "bear breaks into store camera",
            "bear inside gas station camera", "bear in shop security footage",
            "bear indoors security camera overhead", "bear breaks into house camera",
            "bear inside garage security camera", "bear inside cabin camera",
            "bear inside building security footage", "bear raids store camera",
            "bear inside convenience store video", "bear indoor overhead camera angle",
        ],
        "filter": "none",
        "max_per_query": 12,
    },
    "mesh_occlusion": {
        # 2026-08-18: FN-table review, video SSkxoPiNG3Y (22 FN) -- night IR, bear partially
        # occluded by hammock netting/mesh in the foreground. Distinct from ComprehensiveOcclusion
        # (foliage/solid objects), this is a fine mesh/net pattern crossing the bear.
        "queries": [
            "bear tangled in net camera", "bear behind fence mesh camera",
            "bear through screen mesh trail camera", "bear caught in hammock camera",
            "bear behind chain link fence camera", "bear through net wildlife camera",
            "bear on hammock backyard camera", "bear behind netting trail cam",
            "bear behind wire mesh camera", "bear in trellis netting camera",
        ],
        "filter": "night",
        "max_per_query": 12,
    },
    "indoor_night_breakin": {
        # 2026-08-18: FN-table review, video _qlckdZwy2M (25 FN) -- night, bear breaking into a
        # kitchen, extreme close-up. Distinct from indoor_store (daytime/overhead retail context)
        # and from night_vision (outdoor forest/yard scenes).
        "queries": [
            "bear breaks into kitchen night camera", "bear inside house night camera",
            "bear breaks into home night footage", "bear in kitchen security camera night",
            "bear indoors night trail camera", "bear raids kitchen night video",
            "bear breaks into cabin night camera", "bear inside house dark camera",
            "bear close up night camera indoors", "bear enters house night security cam",
        ],
        "filter": "night",
        "max_per_query": 12,
    },
    "backlit_night_distant": {
        # 2026-08-18: FN-table review, video Swmd8TZA32c (4 FN) -- backyard security camera at
        # night, bear small and distant (15-25m), backlit by a motion-sensor light causing flare,
        # low contrast. Distinct from night_vision (closer/clearer IR) and low_light (no flare).
        "queries": [
            "bear backyard camera motion light night", "bear porch light flare camera",
            "bear silhouette night camera light glare", "bear distant backyard camera night",
            "bear security light glare trail camera", "bear far away motion light camera",
            "bear backlit night security camera", "bear yard camera bright light night",
            "bear night camera lens flare", "bear distant silhouette security light",
        ],
        "filter": "night",
        "max_per_query": 12,
    },
    "distant_forest_clutter": {
        # 2026-08-18: FN-table review, video RdudVAoOgRM (2 FN) -- bear small and distant, at the
        # frame's edge on a log, cluttered forest background with dappled sunlight. Distinct from
        # bear_cubs_small (subject size from age, not distance) and aerial_drone (camera angle).
        "queries": [
            "bear far away forest trail camera", "distant bear trail camera trees",
            "small bear far trail camera forest", "tiny bear distant trail camera woods",
            "bear background trail camera dappled sunlight", "bear edge of frame trail camera",
            "bear barely visible trail camera forest", "bear distant clearing trail camera",
            "bear far off trail camera woods", "small distant bear wildlife camera forest",
        ],
        "filter": "none",
        "max_per_query": 12,
    },
}

# ============================================================
# PHYSICAL FILTERS & JUNK DETECTION
# ============================================================
def apply_filter(img, mode):
    if mode == "none": return True
    if mode == "night":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 1].mean() < 40 and hsv[:, :, 2].mean() < 180 or hsv[:, :, 2].mean() < 70
    if mode == "snow":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        bottom_half = hsv[hsv.shape[0]//2:, :, :]
        return ((bottom_half[:, :, 1] < 60) & (bottom_half[:, :, 2] > 150)).mean() > 0.15
    if mode == "dark":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() < 100
    return True

def is_junk_frame(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 10 or gray.std() < 8: return True
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    if hist.max() / hist.sum() > 0.60: return True
    if cv2.Canny(gray, 50, 150).sum() / (gray.size * 255) < 0.01: return True
    if cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].mean() > 80 and cv2.Laplacian(gray, cv2.CV_64F).std() > 30: return True
    return False

def phash(image, hash_size=16):
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    return (gray[:, 1:] > gray[:, :-1]).flatten().astype(np.uint8)

# ============================================================
# DOWNLOAD VIDEOS
# ============================================================
def download_videos(queries, video_dir, max_per_query=3, max_duration=900, cookies_file=None):
    video_dir.mkdir(parents=True, exist_ok=True)
    if not shutil.which("yt-dlp"):
        subprocess.run(["pip", "install", "yt-dlp", "--break-system-packages", "-q"])

    total_downloaded = 0; failures = 0
    for query in queries:
        safe = query.replace(" ", "_")[:50]
        existing = list(video_dir.glob(f"{safe}_*.mp4"))
        if len(existing) >= max_per_query:
            print(f"    [BỎ QUA] Đã có đủ {len(existing)} video cho: '{query}'")
            total_downloaded += len(existing); continue

        print(f"    [TẢI LÔ] Tải {max_per_query} video cho: '{query}'")
        out_tmpl = str(video_dir / f"{safe}_%(autonumber)03d.%(ext)s")
        cmd = [
            "yt-dlp", "-i", "-f", "bestvideo[height<=720][ext=mp4]/best[ext=mp4]/best",
            "--match-filter", f"duration<{max_duration}", f"ytsearch{max_per_query}:{query}",
            "--sleep-requests", "2", "--sleep-interval", "4", "--max-sleep-interval", "10",
            "--retries", "5", "--socket-timeout", "30", "--no-warnings", "-o", out_tmpl,
        ]
        # 2026-08-10: YouTube's "n challenge" now requires a JS runtime to solve, or most
        # formats silently disappear ("Only images are available for download").
        if shutil.which("deno") or os.path.exists("/root/.deno/bin/deno"):
            deno_path = shutil.which("deno") or "/root/.deno/bin/deno"
            cmd.extend(["--js-runtimes", f"deno:{deno_path}", "--remote-components", "ejs:github"])
        if cookies_file and os.path.exists(cookies_file): cmd.extend(["--cookies", cookies_file])

        try: result = subprocess.run(cmd, capture_output=True, text=True)
        except subprocess.TimeoutExpired: failures += 1; time.sleep(10); continue

        if result.returncode != 0: failures += 1; time.sleep(5) 
        else: total_downloaded += max_per_query; time.sleep(random.uniform(3.0, 7.0))

    total_now = len(list(video_dir.glob("*.mp4")))
    print(f"    Thống kê: Hiện có {total_now} video trong thư mục ({failures} lỗi)")
    return total_now

def download_from_urls(url_file, video_dir):
    video_dir.mkdir(parents=True, exist_ok=True)
    with open(url_file) as f: urls = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    for i, url in enumerate(urls):
        vp = video_dir / f"custom_{i:03d}.mp4"
        if vp.exists(): continue
        subprocess.run(["yt-dlp", "-f", "best[height<=720]", "-o", str(vp), url], capture_output=True)

# ============================================================
# EXTRACT FRAMES
# ============================================================
def extract_frames(video_dir, frames_dir, fps=1, gpus=[0]):
    frames_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(video_dir.glob("*.mp4"))
    unique_gpus = list(set(gpus)) if gpus else [0]
    print(f"    Extracting from {len(videos)} videos at {fps} fps")
    has_nvdec = False
    try: has_nvdec = "cuda" in subprocess.run(["ffmpeg", "-hwaccels"], capture_output=True, text=True).stdout.lower()
    except Exception: pass

    def process_video(task_args):
        vp, gpu_id = task_args
        prefix = vp.stem
        out_pat = str(frames_dir / f"{prefix}_%05d.jpg")
        if next(frames_dir.glob(f"{prefix}_*.jpg"), None): return
        if has_nvdec:
            cmd = ["ffmpeg", "-y", "-hwaccel", "cuda", "-hwaccel_device", str(gpu_id), "-i", str(vp), "-vf", f"fps={fps}", "-q:v", "2", out_pat]
            if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
                subprocess.run(["ffmpeg", "-y", "-i", str(vp), "-vf", f"fps={fps}", "-q:v", "2", out_pat], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(["ffmpeg", "-y", "-i", str(vp), "-vf", f"fps={fps}", "-q:v", "2", out_pat], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    tasks = [(vp, unique_gpus[i % len(unique_gpus)]) for i, vp in enumerate(videos)]
    max_workers = max(1, min(len(tasks), len(unique_gpus) * 2))
    with ThreadPoolExecutor(max_workers=max_workers) as executor: list(executor.map(process_video, tasks))
    total = len(list(frames_dir.glob("*.jpg")))
    print(f"    Total frames: {total}")
    return total

# ============================================================
# GPU WORKER (YOLO)
# ============================================================
def gpu_worker(gpu_id, scenario, frame_paths, output_dir, filter_mode, confidence, dedup_threshold):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    device = "cuda:0"
    if torch.cuda.is_available(): torch.cuda.set_device(0)

    try:
        from sahi.models.yolov8 import Yolov8DetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "sahi==0.11.15", "ultralytics", "--break-system-packages", "-q"])
        from sahi.models.yolov8 import Yolov8DetectionModel
        from sahi.predict import get_sliced_prediction

    print(f"    [GPU {gpu_id}] Loading model on {device}...")
    try: detection_model = Yolov8DetectionModel(model_path='yolov8m.pt', confidence_threshold=confidence, device=device)
    except Exception as e: print(f"    [GPU {gpu_id}] Model error: {e}"); return

    # A second run with the same (scenario, gpu_id) restarted `kept` at 0 and
    # silently overwrote earlier files sharing the same index -- including,
    # in one incident, 40 images a human had already verified as containing a
    # bear. Start past whatever indices already exist, and seed kept_hashes
    # from the existing files so cross-run dedup still works.
    existing = sorted(output_dir.glob(f"{scenario}_gpu{gpu_id}_*.jpg"))
    kept = 0
    for fp in existing:
        try:
            kept = max(kept, int(fp.stem.split("_")[-2]) + 1)
        except (IndexError, ValueError):
            pass
    kept_hashes = []
    for fp in existing:
        im = cv2.imread(str(fp))
        if im is not None:
            kept_hashes.append(phash(im))
    for i, fp in enumerate(frame_paths):
        if (i + 1) % 500 == 0: print(f"    [GPU {gpu_id}] Scanned {i+1}/{len(frame_paths)} | kept: {kept}")
        img = cv2.imread(str(fp))
        if img is None or is_junk_frame(img) or not apply_filter(img, filter_mode): continue
        h_img, w_img = img.shape[:2]
        slice_size = min(640, min(h_img, w_img))
        try: result = get_sliced_prediction(img, detection_model, slice_height=slice_size, slice_width=slice_size, overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0)
        except Exception: continue
        has_bear, best_conf = False, 0.0
        for pred in result.object_prediction_list:
            if pred.category.name.lower() == "bear":
                has_bear = True
                if pred.score.value > best_conf: best_conf = pred.score.value
        if not has_bear: continue
        h = phash(img)
        if any(np.sum(h != prev_h) < dedup_threshold for prev_h in kept_hashes): continue
        kept_hashes.append(h)
        dst = output_dir / f"{scenario}_gpu{gpu_id}_{kept:05d}_conf{best_conf:.2f}.jpg"
        cv2.imwrite(str(dst), img); kept += 1
    print(f"    [GPU {gpu_id}] Done: {kept} bear images for {scenario}")

# ============================================================
# CROSS-GPU DEDUP
# ============================================================
def process_single_hash(fp):
    img = cv2.imread(str(fp)); return (fp, phash(img)) if img is not None else (fp, None)

def cross_gpu_dedup(output_dir, threshold=10, gpu_id=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    files = sorted(output_dir.glob("*.jpg"))
    if len(files) < 2: return 0
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): torch.cuda.set_device(0)

    print(f"    Step 4: Cross-GPU dedup on physical GPU {gpu_id}...")
    cache_file = output_dir / "hash_cache.pkl"
    hash_cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f: hash_cache = pickle.load(f)
        except Exception: pass

    hashes, valid_files, files_to_hash = [], [], []
    for fp in files:
        if fp.name in hash_cache: valid_files.append(fp); hashes.append(hash_cache[fp.name])
        else: files_to_hash.append(fp)

    if files_to_hash:
        with ThreadPoolExecutor(max_workers=16) as executor:
            for fp, h in executor.map(process_single_hash, files_to_hash):
                if h is not None: valid_files.append(fp); hashes.append(h); hash_cache[fp.name] = h

    N = len(hashes)
    if N < 2: return 0
    hash_tensor = torch.tensor(np.array(hashes), dtype=torch.uint8, device=device)
    keep_mask = torch.ones(N, dtype=torch.bool, device=device)
    for i in range(N):
        if not keep_mask[i]: continue
        dist = (hash_tensor[i+1:] != hash_tensor[i]).sum(dim=1)
        keep_mask[i+1:] = keep_mask[i+1:] & (~(dist < threshold))

    removed = 0
    for i, keep in enumerate(keep_mask.cpu().numpy()):
        if not keep:
            valid_files[i].unlink()
            if valid_files[i].name in hash_cache: del hash_cache[valid_files[i].name]
            removed += 1
    with open(cache_file, "wb") as f: pickle.dump(hash_cache, f)
    print(f"      -> Removed {removed} cross-GPU duplicates. {N - removed} images kept.")
    return removed

# ============================================================
# STEP 5: BioCLIP POST-FILTERING (THE GOLDEN VETO V3)
# ============================================================
def post_filter_clip(output_dir, gpu_id=0, batch_size=128, threshold=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    files = sorted(output_dir.glob("*.jpg"))
    if not files: return 0

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): torch.cuda.set_device(0)

    print(f"\n    Step 5: Post-filtering using BioCLIP & Strict Veto on physical GPU {gpu_id}...")
    scenario_name = output_dir.parent.name
    is_dark_scenario = scenario_name in ["night_vision", "IR", "low_light", "rain_fog"]

    if threshold is None: threshold = 0.15 if is_dark_scenario or scenario_name in ["distance_10m", "partial_occlusion"] else 0.20

    cache_file = output_dir / "clip_passed_cache.pkl"
    clip_passed_cache = set()
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f: clip_passed_cache = pickle.load(f)
        except Exception: pass

    files_to_process = [fp for fp in files if fp.name not in clip_passed_cache]
    if not files_to_process: print("      -> All images have already passed. Skipping."); return 0

    try: import open_clip
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "open_clip_torch", "timm", "--break-system-packages", "-q"]); import open_clip
    from PIL import Image

    model_id = 'hf-hub:imageomics/bioclip'
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(model_id, device=device)
        tokenizer = open_clip.get_tokenizer(model_id)
    except Exception as e: print(f"      -> Fatal Error loading BioCLIP: {e}"); return 0

    # THÊM NHÃN "VĂN BẢN/PHỤ ĐỀ" ĐỂ ĐẬP CHẾT VIDEO SHORTS
    classes = [
        "American black bear",                          # 0
        "brown bear",                                   # 1
        "grizzly bear",                                 # 2
        "red fox",                                      # 3
        "coyote",                                       # 4
        "gray wolf",                                    # 5
        "deer",                                         # 6
        "elk",                                          # 7
        "moose",                                        # 8
        "domestic dog",                                 # 9
        "cat",                                          # 10
        "raccoon",                                      # 11
        "porcupine",                                    # 12
        "wild boar",                                    # 13
        "human",                                        # 14
        "empty night forest",                           # 15
        "animated cartoon drawing",                     # 16
        "CGI 3D render",                                # 17
        "stuffed toy",                                  # 18
        "a video screenshot with large text subtitles", # 19 <-- VŨ KHÍ MỚI
    ]
    bear_indices = [0, 1, 2]
    reject_indices = [16, 17, 18, 19] # Nhóm 1: Rác mạng / Chữ / Đồ giả
    hard_reject_indices = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15] # Nhóm 2: Nai, Cáo, Nhím, Sói... (Lưu ý: Thả Lợn rừng số 13 ra để bảo kê)
    
    with torch.no_grad():
        text_inputs = tokenizer(classes).to(device)
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp().item()

    trash_dir = output_dir.parent / "_false_positives_trash"
    trash_dir.mkdir(exist_ok=True)
    review_dir = output_dir.parent / "_needs_manual_review"
    review_dir.mkdir(exist_ok=True)
    removed, kept = 0, 0

    for i in range(0, len(files_to_process), batch_size):
        batch_files = files_to_process[i:i + batch_size]
        images, valid_files = [], []
        
        for fp in batch_files:
            try: img = Image.open(fp).convert("RGB"); images.append(preprocess(img)); valid_files.append(fp)
            except Exception: continue
        
        if not images: continue

        with torch.no_grad():
            image_tensor = torch.stack(images).to(device)
            image_features = model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * (image_features @ text_features.T)
            sims = logits.softmax(dim=-1).cpu().numpy()

        for fp, sims_img in zip(valid_files, sims):
            bear_score = sum(sims_img[idx] for idx in bear_indices)
            reject_score = sum(sims_img[idx] for idx in reject_indices)
            
            hard_reject_score = 0
            best_hard_idx = -1
            if hard_reject_indices:
                hard_reject_score = max(sims_img[idx] for idx in hard_reject_indices)
                best_hard_idx = max(hard_reject_indices, key=lambda idx: sims_img[idx])

            try: yolo_conf = float(fp.name.split("conf")[-1].split(".jpg")[0])
            except Exception: yolo_conf = 0.0

            reason = None
            
            # =======================================================
            # LOGIC XÉT XỬ TỐI THƯỢNG (SỬA LỖI TRÌNH TỰ)
            # =======================================================
            
            # 1. Chém đầu đồ giả & Phụ đề to đùng (Bất chấp YOLO)
            if reject_score > 0.35 or reject_score > bear_score:
                reason = "animated_cgi_text"
            
            # 2. Chém đầu con vật khác: Nhím, Nai, Sói... (Bất chấp YOLO)
            elif hard_reject_score > 0.50 and hard_reject_score > bear_score:
                reason = classes[best_hard_idx].replace(" ", "_")[:30]
            
            # 3. Yolo Veto: Nếu lọt qua 2 khiên trên, BioCLIP đang mơ hồ (ảnh tối mờ / giống lợn rừng). Lúc này YOLO ra tay!
            elif is_dark_scenario and yolo_conf >= 0.40:
                pass # Bảo kê thành công! Gấu sống!
                
            # 4. Gấu điểm thấp + YOLO thấp -> Chém
            elif bear_score < threshold:
                reason = f"weak_bear_{bear_score:.2f}"

            # Xử lý tên file tránh tràn (Errno 36)
            name_str = fp.name
            base_name = name_str[name_str.find(scenario_name):] if scenario_name in name_str and not name_str.startswith(scenario_name) else name_str

            if reason is None: 
                clip_passed_cache.add(base_name); kept += 1
            else:
                # Vớt vát vào Review nếu YOLO thấy nét (>= 0.30) và không phải Đồ giả / Có chữ
                if yolo_conf >= 0.30 and reason != "animated_cgi_text":
                    new_name = f"REVIEW_{reason}_b{bear_score:.2f}_y{yolo_conf:.2f}_{base_name}"
                    if len(new_name) > 240: new_name = new_name[:235] + ".jpg"
                    shutil.move(str(fp), str(review_dir / new_name))
                else:
                    new_name = f"NOT_BEAR_{reason}_b{bear_score:.2f}_y{yolo_conf:.2f}_{base_name}"
                    if len(new_name) > 240: new_name = new_name[:235] + ".jpg"
                    shutil.move(str(fp), str(trash_dir / new_name))
                removed += 1

    with open(cache_file, "wb") as f: pickle.dump(clip_passed_cache, f)
    print(f"      -> BioCLIP Priority Logic removed {removed} FPs, kept {kept}. Total images: {len(files) - removed}")
    return removed

# ============================================================
# MAIN ORCHESTRATOR
# ============================================================
def setup_environment():
    missing = []
    checks = {"sahi": "sahi==0.11.15", "ultralytics": "ultralytics"}
    for pkg, install_spec in checks.items():
        if subprocess.run([sys.executable, "-c", f"import {pkg}"], capture_output=True).returncode != 0: missing.append(install_spec)
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", *missing, "--break-system-packages", "-q"])

def main():
    setup_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="all")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0])
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--dedup", type=int, default=10)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--clip-threshold", type=float, default=None)
    parser.add_argument("--custom-urls", default=None)
    parser.add_argument("--cookies", default=None)
    parser.add_argument("--output-root", default="collected_bears")
    args = parser.parse_args()

    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        if r.returncode == 0:
            print("\n  GPU memory status:")
            gpu_info = {}
            for line in r.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    idx, free, total = int(parts[0]), int(parts[1]), int(parts[2])
                    gpu_info[idx] = (free, total)
                    marker = "  [USED BY SCRIPT]" if idx in args.gpus else ""
                    print(f"    GPU {idx}: {free:>6} MB free / {total} MB total{marker}")
    except Exception: pass

    active = list(SCENARIO_CONFIG.keys()) if args.scenario == "all" else [args.scenario]
    if args.scenario != "all" and args.scenario not in SCENARIO_CONFIG: return

    print(f"{'='*70}\nBEAR COLLECTION: SAHI + Multi-GPU + Tensor Dedup + BioCLIP\n{'='*70}")

    total_all = 0
    for sc in active:
        config = SCENARIO_CONFIG[sc]
        base_dir = Path(args.output_root) / sc
        video_dir, frames_dir, final_dir = base_dir / "videos", base_dir / "all_frames", base_dir / "bear_frames"
        final_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}\nSCENARIO: {sc.upper()}\n{'='*60}")

        if not args.skip_download:
            if args.custom_urls: download_from_urls(args.custom_urls, video_dir)
            else: download_videos(config["queries"], video_dir, max_per_query=config.get("max_per_query", 3), cookies_file=args.cookies)

        if not args.skip_extract: extract_frames(video_dir, frames_dir, fps=args.fps, gpus=args.gpus)

        all_frames = sorted(frames_dir.glob("*.jpg"))
        if not all_frames: continue

        current_conf = args.confidence
        if sc in ["night_vision", "IR", "low_light", "rain_fog"]:
            current_conf = max(args.confidence, 0.25)
            print(f"\n  [INFO] Bumping YOLO confidence to {current_conf} for dark scenario.")

        print(f"\n  Step 3: SAHI bear detection ({len(all_frames)} frames)...")
        num_processes = len(args.gpus)
        chunk_size = math.ceil(len(all_frames) / num_processes)
        chunks = [all_frames[i:i + chunk_size] for i in range(0, len(all_frames), chunk_size)]
        ctx = mp.get_context('spawn'); processes = []
        for i, chunk in enumerate(chunks):
            if i >= num_processes: break
            p = ctx.Process(target=gpu_worker, args=(args.gpus[i], sc, chunk, final_dir, config["filter"], current_conf, args.dedup))
            p.start(); processes.append(p)
        for p in processes: p.join()

        cross_gpu_dedup(final_dir, threshold=args.dedup, gpu_id=args.gpus[0])
        if not args.skip_clip: post_filter_clip(final_dir, gpu_id=args.gpus[0], threshold=args.clip_threshold)

        final_count = len(list(final_dir.glob("*.jpg")))
        total_all += final_count
        print(f"\n  Result: {final_count} bear images finalized for {sc}")

    print(f"\n{'='*60}\nTOTAL: {total_all} images\n{'='*60}")

if __name__ == "__main__":
    main()