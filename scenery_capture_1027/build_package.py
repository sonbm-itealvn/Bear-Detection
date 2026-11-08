"""Build capture_package_1027 for filming on a screen with the target camera (fix for the
27/10 V19 false positives on Japanese-garden rocks; see
bear_project/docs/reports/2026-10-27_v19_fp/REPORT.md).

non_bear  scenery: Wikimedia Commons stills + frames of 4K Japan scenery videos (YouTube), Japanese
          gardens, rock/zen gardens, bamboo, mossy boulders / forest. YOLOv8m-screened for bears,
          phash-deduped, ranked by V19. Video frames are split by WHOLE video (25% of videos test).
          set=train (to be merged after filming) or set=test (held out: never merged).
bear      train bears among rocks / in forests (CLIP ViT-L/14 text retrieval), so that
          "filmed off a screen" never becomes a non-bear cue.
Everything is letterboxed to 1920x1080 and shown in one shuffled order.
python scenery_capture_1027/build_package.py --gpu 2"""
import argparse, csv, json, os, random, shutil, sys, zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/data/nas07_new/PersonalData/phuocthien")
sys.path.insert(0, str(ROOT))
SRC = ROOT / "scenery_capture_1027"
OUT = ROOT / "capture_package_1027"
CAMERA_PREFIXES = ("acc", "cap1009", "screencap", "bear_MFDC", "non_bear_MFDC", "MFDC", "capture",
                   "bear_full_capture", "bear_capture", "bear_pasteaug")


def letterbox(src, dst, W=1920, H=1080):
    im = Image.open(src).convert("RGB")
    k = min(W / im.width, H / im.height)
    im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
    c = Image.new("RGB", (W, H), (0, 0, 0))
    c.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
    c.save(dst, quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--n-train", type=int, default=1100)
    ap.add_argument("--n-test", type=int, default=350)
    ap.add_argument("--n-bear", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.gpu in ("0", "7"):
        sys.exit("GPUs 0 and 7 belong to the user")
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if a.gpu == "cpu" else a.gpu
    dev = "cpu" if a.gpu == "cpu" else "cuda"
    import imagehash, torch
    from ultralytics import YOLO
    from det.model import load_det
    from det.dataset import letterbox as lb640, to_tensor
    rng = random.Random(a.seed)
    meta = {}
    if (SRC / "commons_meta.jsonl").exists():
        meta = {f"{json.loads(l)['group']}__{json.loads(l)['pageid']}.jpg": json.loads(l) for l in open(SRC / "commons_meta.jsonl")}
    vids = {v["id"]: v for v in json.load(open(SRC / "videos.json"))}
    files = sorted((SRC / "commons").glob("*.jpg")) + sorted((SRC / "frames").glob("*.jpg"))
    vid_of = lambda f: f.name.split("__")[1] if f.parent.name == "frames" else None
    print("candidates", len(files), flush=True)

    # 1. drop anything YOLO thinks is a bear (any confidence >= 0.15); unreadable files too
    yolo = YOLO(str(ROOT / "yolov8m.pt"))
    clean = []
    for f in files:
        try:
            im = Image.open(f).convert("RGB")
        except Exception:
            continue
        r = yolo.predict(np.asarray(im)[:, :, ::-1], imgsz=1280, conf=0.15, device="cpu" if dev == "cpu" else 0, verbose=False)[0]
        if not any(int(c) == 21 for c in r.boxes.cls.cpu().numpy()):
            clean.append(f)
    print("after YOLO bear screen", len(clean), flush=True)

    # 2. phash dedup (DCT phash distance <= 8)
    keep, hashes = [], []
    for f in clean:
        h = imagehash.phash(Image.open(f).convert("L"))
        if all(h - g > 8 for g in hashes):
            keep.append(f); hashes.append(h)
    print("after dedup", len(keep), flush=True)

    # 3. V19 FP32 p(bear) on the 640x360 letterbox (the camera's view)
    net = load_det(ROOT / "det_runs/s16_v2/best.pth").to(dev).eval()
    p = {}
    with torch.no_grad():
        for i in range(0, len(keep), 32):
            x = torch.stack([to_tensor(lb640(Image.open(f).convert("RGB"))) for f in keep[i:i + 32]]).to(dev)
            for f, l in zip(keep[i:i + 32], net(x)[0].float()):
                p[f] = float(torch.softmax(l, 0)[1])
    ps = np.array([p[f] for f in keep])
    print(f"V19 p(bear) on clean scenery: >=0.5 {int((ps >= .5).sum())}, >=0.2 {int((ps >= .2).sum())}, median {np.median(ps):.3f}", flush=True)

    # 4. split 75/25 per group, then select: all hard (p >= 0.2) first, random fill
    by_g = {}
    for f in keep:
        by_g.setdefault(f.name.split("__")[0], []).append(f)
    tr, te = [], []
    for g, fs in by_g.items():
        stills = [f for f in fs if vid_of(f) is None]
        rng.shuffle(stills); k = int(round(len(stills) * 0.25))
        te += stills[:k]; tr += stills[k:]
        gv = sorted({vid_of(f) for f in fs if vid_of(f)})       # video frames: hold out WHOLE videos
        rng.shuffle(gv); test_v = set(gv[:max(1, int(round(len(gv) * 0.25)))]) if gv else set()
        te += [f for f in fs if vid_of(f) in test_v]; tr += [f for f in fs if vid_of(f) and vid_of(f) not in test_v]
    def pick(fs, n):
        fs = sorted(fs, key=lambda f: -p[f])
        hard = [f for f in fs if p[f] >= 0.2]
        rest = [f for f in fs if p[f] < 0.2]; rng.shuffle(rest)
        return (hard + rest)[:n]
    tr, te = pick(tr, a.n_train), pick(te, a.n_test)

    # 5. bears among rocks / in forests (CLIP text retrieval over the 03/10 embeddings)
    from transformers import CLIPModel, CLIPTokenizer
    name = "openai/clip-vit-large-patch14"
    m = CLIPModel.from_pretrained(name, cache_dir=str(ROOT / ".cache/huggingface")).eval()
    tok = CLIPTokenizer.from_pretrained(name, cache_dir=str(ROOT / ".cache/huggingface"))
    with torch.no_grad():
        t = torch.nn.functional.normalize(m.get_text_features(**tok(
            ["a photo of a bear among rocks", "a photo of a bear in a forest", "a photo of a black bear near boulders"],
            padding=True, return_tensors="pt")), dim=1).numpy()
    E, F = [], []
    for k in range(3):
        z = np.load(ROOT / f"audit_nonbear_1003/clip_{k}of3.npz", allow_pickle=True)
        ok = z["ok"] & (z["labels"] == "bear")
        E.append(z["E"][ok].astype(np.float32)); F += [str(x) for x in z["files"][ok]]
    E = np.concatenate(E); sc = (E @ t.T).max(1)
    train_bear = {f.name: f for f in (ROOT / "bear_dataset_v_final_final/train/bear").iterdir()}
    bears = []
    for i in np.argsort(-sc):
        n = Path(F[i]).name
        if n in train_bear and not n.startswith(CAMERA_PREFIXES) and train_bear[n] not in bears:
            bears.append(train_bear[n])
        if len(bears) == a.n_bear:
            break

    # 6. write the package
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "train").mkdir(parents=True)
    items = [("train", "non_bear", f) for f in tr] + [("test", "non_bear", f) for f in te] + [("train", "bear", f) for f in bears]
    rng.shuffle(items)
    rows = []
    for i, (st, truth, f) in enumerate(items, 1):
        name = f"train_{i:05d}.jpg"
        letterbox(f, OUT / "train" / name)
        mm = meta.get(f.name, {})
        if vid_of(f):
            v = vids.get(vid_of(f), {})
            mm = {"page": f"https://www.youtube.com/watch?v={vid_of(f)}", "license": "YouTube (research use)", "artist": v.get("title", "")[:120]}
        group = f.name.split("__")[0] if truth == "non_bear" else "bear_rocks_forest"
        rows.append({"name": name, "set": st, "truth": truth, "group": group,
                     "source": str(f.relative_to(ROOT)), "v19_p_bear": round(p.get(f, -1), 4),
                     "commons_page": mm.get("page", ""), "license": mm.get("license", ""), "artist": mm.get("artist", "")})
    with open(OUT / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(OUT / "train/labels.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["file", "truth", "group", "source"])
        for r in rows:
            w.writerow([r["name"], r["truth"], r["group"], r["source"]])
    with open(OUT / "ATTRIBUTION.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["file", "commons_page", "license", "artist"])
        for r in rows:
            if r["commons_page"]:
                w.writerow([r["name"], r["commons_page"], r["license"], r["artist"]])
    zp = ROOT / "capture_package_1027.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                z.write(f, f"capture_package_1027/{f.relative_to(OUT)}")
    import collections
    print("PACKAGE", len(rows), dict(collections.Counter((r["set"], r["truth"]) for r in rows)),
          dict(collections.Counter(r["group"] for r in rows)), "zip", zp, zp.stat().st_size // 2**20, "MB")


if __name__ == "__main__":
    main()
