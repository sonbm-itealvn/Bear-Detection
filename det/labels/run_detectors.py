"""Box proposals for the pseudo-label pipeline (spec §2).
YOLOv8m (all COCO classes, conf >= 0.05) at imgsz 640 and 1280 with horizontal-flip TTA,
plus 640-px SAHI-style tiles for images >= 1280 px wide and the 'small' domain; OWLv2 with
four bear queries. Writes one JSON line per image; resumable (skips sha1s already written).
CUDA_VISIBLE_DEVICES is set from --gpu BEFORE torch/ultralytics are imported (ultralytics
grabs GPU 0 otherwise)."""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/nas07_new/PersonalData/phuocthien")
QUERIES = ["a bear", "a polar bear", "a giant panda", "a bear cub"]
OWL = "google/owlv2-base-patch16-ensemble"


def tiles(w, h, size=640, overlap=0.2):
    def starts(n):
        if n <= size:
            return [0]
        step = int(size * (1 - overlap))
        return sorted(set(list(range(0, n - size, step)) + [n - size]))
    return [(x, y, min(x + size, w), min(y + size, h)) for y in starts(h) for x in starts(w)]


def unflip(b, width):
    x1, y1, x2, y2 = b
    return [width - x2, y1, width - x1, y2]


def nms_rows(rows, thr=0.6):
    """Per-class NMS over [x1,y1,x2,y2,score,cls] rows."""
    from det.labels.fusion import iou
    if not rows:
        return []
    a = np.array(rows, float)
    keep = []
    for c in np.unique(a[:, 5]):
        idx = np.where(a[:, 5] == c)[0]
        idx = idx[np.argsort(-a[idx, 4])]
        while len(idx):
            i = idx[0]
            keep.append(i)
            if len(idx) == 1:
                break
            idx = idx[1:][iou(a[i:i + 1, :4], a[idx[1:], :4])[0] < thr]
    return a[sorted(keep)].tolist()


class Detectors:
    def __init__(self):
        import torch
        from ultralytics import YOLO
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.torch = torch
        self.yolo_model = YOLO(str(ROOT / "yolov8m.pt"))
        hf = str(ROOT / ".cache/huggingface")
        self.proc = Owlv2Processor.from_pretrained(OWL, cache_dir=hf)
        self.owl = Owlv2ForObjectDetection.from_pretrained(OWL, cache_dir=hf).to("cuda").eval()

    def _yolo(self, img, x0=0, y0=0, sz=640):
        rows = []
        for flip in (False, True):
            im = img.transpose(0) if flip else img                    # 0 = FLIP_LEFT_RIGHT
            r = self.yolo_model.predict(np.asarray(im)[:, :, ::-1], imgsz=sz, conf=0.05, device=0, verbose=False)[0]
            for b, s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                b = unflip(b, im.width) if flip else list(b)
                rows.append([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0, float(s), int(c)])
        return rows

    def yolo(self, img, small):
        w, h = img.size
        rows = self._yolo(img, sz=640) + self._yolo(img, sz=1280)
        if w >= 1280 or small:
            for (x0, y0, x1, y1) in tiles(w, h):
                rows += self._yolo(img.crop((x0, y0, x1, y1)), x0, y0, 640)
        return nms_rows(rows)

    def owlv2(self, img):
        torch = self.torch
        w, h = img.size
        with torch.no_grad():
            inp = self.proc(text=[QUERIES], images=img, return_tensors="pt").to("cuda")
            o = self.owl(**inp)
            side = max(w, h)                                         # OWLv2 pads to a square (bottom/right)
            res = self.proc.post_process_object_detection(o, threshold=0.05,
                                                          target_sizes=torch.tensor([[side, side]]).cuda())[0]
        return [[max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3]), float(s), int(l)]
                for b, s, l in zip(res["boxes"].cpu().tolist(), res["scores"].cpu().tolist(), res["labels"].cpu().tolist())]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train"); ap.add_argument("--cls", default="bear")
    ap.add_argument("--gpu", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--files", help="list of image paths (overrides --split/--cls)")
    a = ap.parse_args(argv)
    if a.gpu in ("0", "7"):
        sys.exit("GPUs 0 and 7 belong to the user -- refusing")
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    import torch
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    from det.labels.domains import domain_of
    from det.labels.store import load_rgb, sha1_file
    if a.files:
        files = [Path(l.strip()) for l in open(a.files) if l.strip()]
    else:
        d = ROOT / "bear_dataset_v_final_final" / a.split / a.cls
        files = sorted(d.iterdir())
    if a.limit:
        files = files[:a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["sha1"] for l in open(out) if l.strip()} if out.exists() else set()
    det = Detectors()
    n = 0
    with open(out, "a") as fo:
        for p in files:
            sha = sha1_file(p)
            if sha in done:
                continue
            img = load_rgb(p)
            dom = domain_of(p.name)
            fo.write(json.dumps({"path": str(p), "sha1": sha, "w": img.width, "h": img.height, "domain": dom,
                                 "yolo": det.yolo(img, dom == "small"), "owl": det.owlv2(img)}) + "\n")
            fo.flush()
            n += 1
            if n % 500 == 0:
                print(f"  {n} / {len(files) - len(done)}", flush=True)
    print(f"done: {n} new images -> {out}")


if __name__ == "__main__":
    main()
