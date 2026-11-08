"""Screen the 28/10 animal/people candidates (animals_capture_1029/{commons,frames}) and write
screen.json: per image YOLOv8m max 'bear' confidence (>= 0.15 -> flagged for review BY EYE,
never dropped automatically: on dark small mammals YOLO's bear class fires on ~25%), DCT-phash
dedup (<= 8) and V19 FP32 p(bear) on the 640x360 letterbox. CPU-only (GPU2 fault).
Also writes review_flagged_<k>.jpg contact sheets of the flagged images."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path("/data/nas07_new/PersonalData/phuocthien")
sys.path.insert(0, str(ROOT))
SRC = ROOT / "animals_capture_1029"


def main():
    import os; os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import imagehash, torch
    from ultralytics import YOLO
    from det.model import load_det
    from det.dataset import letterbox as lb640, to_tensor
    files = sorted((SRC / "commons").glob("*.jpg")) + sorted((SRC / "frames").glob("*.jpg"))
    print("candidates", len(files), flush=True)
    keep, hashes = [], []
    for f in files:
        try:
            h = imagehash.phash(Image.open(f).convert("L"))
        except Exception:
            continue
        if all(h - g > 8 for g in hashes):
            keep.append(f); hashes.append(h)
    print("after dedup", len(keep), flush=True)
    yolo = YOLO(str(ROOT / "yolov8m.pt"))
    res = {}
    for i, f in enumerate(keep):
        r = yolo.predict(np.asarray(Image.open(f).convert("RGB"))[:, :, ::-1], imgsz=1280, conf=0.15, device="cpu", verbose=False)[0]
        cls, cf = r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()
        res[str(f.relative_to(ROOT))] = {"yolo_bear": float(cf[cls == 21].max()) if (cls == 21).any() else 0.0,
                                         "yolo_classes": sorted({r.names[int(c)] for c in cls})}
        if i % 500 == 0:
            print("yolo", i, flush=True)
    net = load_det(ROOT / "det_runs/s16_v2/best.pth").eval()
    with torch.no_grad():
        for i in range(0, len(keep), 32):
            x = torch.stack([to_tensor(lb640(Image.open(f).convert("RGB"))) for f in keep[i:i + 32]])
            for f, l in zip(keep[i:i + 32], net(x)[0].float()):
                res[str(f.relative_to(ROOT))]["v19_p"] = float(torch.softmax(l, 0)[1])
    json.dump(res, open(SRC / "screen.json", "w"), indent=0)
    flagged = sorted([k for k, v in res.items() if v["yolo_bear"] >= 0.15], key=lambda k: -res[k]["yolo_bear"])
    ps = np.array([v["v19_p"] for v in res.values()])
    print(f"flagged {len(flagged)}; V19 p>=0.5 {int((ps >= .5).sum())}, >=0.2 {int((ps >= .2).sum())}", flush=True)
    W, H = 384, 216
    for s in range(0, len(flagged), 30):
        sheet = Image.new("RGB", (W * 6, (H + 16) * 5), (20, 20, 20))
        for k, n in enumerate(flagged[s:s + 30]):
            x, y = (k % 6) * W, (k // 6) * (H + 16)
            sheet.paste(Image.open(ROOT / n).convert("RGB").resize((W, H)), (x, y + 16))
            ImageDraw.Draw(sheet).text((x + 4, y + 2), f"#{s + k} {Path(n).name[:34]} yb={res[n]['yolo_bear']:.2f}", fill=(255, 220, 80))
        sheet.save(SRC / f"review_flagged_{s // 30:02d}.jpg", quality=85)
    (SRC / "flagged.txt").write_text("\n".join(flagged) + "\n")


if __name__ == "__main__":
    main()
