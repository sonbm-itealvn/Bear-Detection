"""Build the animal/people half of the 29/10 capture package and merge it with
capture_package_1027 (rocks / gardens / bamboo) into ONE shuffled package for a single
filming session -> capture_package_1029/ + capture_package_1029.zip.

Animal half (fix for the 28/10 V19 false positives; REPORT.md section 5):
  non_bear  crow / cat_closeup / dog_dark / person_forest: Commons stills + YouTube frames,
            from screen.json (dedup, V19 p(bear)), minus rejected.txt (images I judged by eye to
            show a bear among the YOLO-flagged ones). Per group: hard (V19 p >= 0.2) first, then
            random. Video frames split by WHOLE video (25% of videos test), stills 75/25.
  bear      train bears in matching looks (close-up black bears, bears standing upright, bears
            lying down, bears walking in grass), CLIP ViT-L/14 text retrieval, excluding the
            bears already in capture_package_1027.
Everything is letterboxed to 1920x1080; package-1027 files are copied as they are.
python animals_capture_1029/build_package.py"""
import argparse, collections, csv, json, os, random, shutil, sys, zipfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("/data/nas07_new/PersonalData/phuocthien")
sys.path.insert(0, str(ROOT))
SRC = ROOT / "animals_capture_1029"
P1 = ROOT / "capture_package_1027"
OUT = ROOT / "capture_package_1029"
GROUPS = ("crow", "cat_closeup", "dog_dark", "person_forest")
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
    ap.add_argument("--n-train", type=int, default=250, help="per group")
    ap.add_argument("--n-test", type=int, default=75, help="per group")
    ap.add_argument("--n-bear", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    rng = random.Random(a.seed)
    scr = json.load(open(SRC / "screen.json"))
    rej = set(l.strip() for l in open(SRC / "rejected.txt") if l.strip()) if (SRC / "rejected.txt").exists() else set()
    meta = {}
    for l in open(SRC / "commons_meta.jsonl"):
        x = json.loads(l); meta[f"{x['group']}__{x['pageid']}.jpg"] = x
    vids = {v["id"]: v for v in json.load(open(SRC / "videos.json"))}
    files = [ROOT / k for k in scr if k not in rej]
    p = {ROOT / k: v["v19_p"] for k, v in scr.items()}
    vid_of = lambda f: f.name.split("__")[1] if f.parent.name == "frames" else None
    print("screened", len(scr), "rejected by eye", len(rej), "usable", len(files), flush=True)

    tr, te = [], []
    for g in GROUPS:
        fs = [f for f in files if f.name.split("__")[0] == g]
        stills = [f for f in fs if vid_of(f) is None]; rng.shuffle(stills)
        k = int(round(len(stills) * 0.25)); gte, gtr = stills[:k], stills[k:]
        gv = sorted({vid_of(f) for f in fs if vid_of(f)}); rng.shuffle(gv)
        test_v = set(gv[:max(1, int(round(len(gv) * 0.25)))]) if gv else set()
        gte += [f for f in fs if vid_of(f) in test_v]; gtr += [f for f in fs if vid_of(f) and vid_of(f) not in test_v]
        def pick(xs, n):
            xs = sorted(xs, key=lambda f: -p[f]); hard = [f for f in xs if p[f] >= 0.2]
            rest = [f for f in xs if p[f] < 0.2]; rng.shuffle(rest)
            # at most half of a group's quota from one source video, so one video cannot dominate
            out, per = [], collections.Counter()
            for f in hard + rest:
                v = vid_of(f) or "still"
                if v != "still" and per[v] >= max(15, n // 4):
                    continue
                out.append(f); per[v] += 1
                if len(out) == n:
                    break
            return out
        a_tr, a_te = pick(gtr, a.n_train), pick(gte, a.n_test)
        print(f"{g:14s} pool train {len(gtr):4d} test {len(gte):4d} -> {len(a_tr)} / {len(a_te)}; "
              f"hard(p>=0.2) {sum(p[f] >= .2 for f in a_tr)} / {sum(p[f] >= .2 for f in a_te)}", flush=True)
        tr += a_tr; te += a_te

    from transformers import CLIPModel, CLIPTokenizer
    name = "openai/clip-vit-large-patch14"
    m = CLIPModel.from_pretrained(name, cache_dir=str(ROOT / ".cache/huggingface")).eval()
    tok = CLIPTokenizer.from_pretrained(name, cache_dir=str(ROOT / ".cache/huggingface"))
    with torch.no_grad():
        t = torch.nn.functional.normalize(m.get_text_features(**tok(
            ["a close-up photo of a black bear", "a photo of a bear standing on its hind legs",
             "a photo of a bear lying down", "a photo of a black bear walking in the grass"],
            padding=True, return_tensors="pt")), dim=1).numpy()
    E, F = [], []
    for k in range(3):
        z = np.load(ROOT / f"audit_nonbear_1003/clip_{k}of3.npz", allow_pickle=True)
        ok = z["ok"] & (z["labels"] == "bear")
        E.append(z["E"][ok].astype(np.float32)); F += [str(x) for x in z["files"][ok]]
    S = np.concatenate(E) @ t.T
    used = {Path(r["source"]).name for r in csv.DictReader(open(P1 / "manifest.csv")) if r["truth"] == "bear"}
    train_bear = {f.name: f for f in (ROOT / "bear_dataset_v_final_final/train/bear").iterdir()}
    bears, per_q = [], a.n_bear // S.shape[1]
    for j in range(S.shape[1]):                      # equal share per query
        n = 0
        for i in np.argsort(-S[:, j]):
            nm = Path(F[i]).name
            if nm in train_bear and nm not in used and not nm.startswith(CAMERA_PREFIXES):
                bears.append(train_bear[nm]); used.add(nm); n += 1
                if n == per_q:
                    break

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "train").mkdir(parents=True)
    items = [("1029_animals", "train", "non_bear", f.name.split("__")[0], f) for f in tr] + \
            [("1029_animals", "test", "non_bear", f.name.split("__")[0], f) for f in te] + \
            [("1029_animals", "train", "bear", "bear_closeup_pose", f) for f in bears]
    for r in csv.DictReader(open(P1 / "manifest.csv")):        # package 1, unchanged images
        items.append(("1027_scenery", r["set"], r["truth"], r["group"], P1 / "train" / r["name"], r))
    rng.shuffle(items)
    rows = []
    for i, it in enumerate(items, 1):
        pkg, st, truth, group, f = it[:5]
        name = f"img_{i:05d}.jpg"
        if pkg == "1027_scenery":
            r1 = it[5]; shutil.copyfile(f, OUT / "train" / name)
            rows.append({"name": name, "package": pkg, "set": st, "truth": truth, "group": group, "source": r1["source"],
                         "orig_name": r1["name"], "v19_p_bear": r1["v19_p_bear"], "page": r1["commons_page"],
                         "license": r1["license"], "artist": r1["artist"]})
            continue
        letterbox(f, OUT / "train" / name)
        mm = meta.get(f.name, {})
        if vid_of(f):
            v = vids.get(vid_of(f), {})
            mm = {"page": f"https://www.youtube.com/watch?v={vid_of(f)}", "license": "YouTube (research use)", "artist": v.get("title", "")[:120]}
        rows.append({"name": name, "package": pkg, "set": st, "truth": truth, "group": group, "source": str(f.relative_to(ROOT)),
                     "orig_name": "", "v19_p_bear": round(p.get(f, -1), 4), "page": mm.get("page", ""),
                     "license": mm.get("license", ""), "artist": mm.get("artist", "")})
    with open(OUT / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(OUT / "train/labels.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["file", "truth", "group", "source"])
        for r in rows:
            w.writerow([r["name"], r["truth"], r["group"], r["source"]])
    with open(OUT / "ATTRIBUTION.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["file", "source_page", "license", "artist"])
        for r in rows:
            if r["page"]:
                w.writerow([r["name"], r["page"], r["license"], r["artist"]])
    zp = ROOT / "capture_package_1029.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                z.write(f, f"capture_package_1029/{f.relative_to(OUT)}")
    print("PACKAGE", len(rows), dict(collections.Counter((r["package"], r["set"], r["truth"]) for r in rows)),
          dict(collections.Counter(r["group"] for r in rows)), "zip", zp, zp.stat().st_size // 2**20, "MB", flush=True)


if __name__ == "__main__":
    main()
