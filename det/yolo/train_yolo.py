"""Train candidate (b): YOLOv8n with ReLU (Novaic cannot run SiLU efficiently), nc=1,
fine-tuned from COCO yolov8n.pt (all shape-compatible weights transfer; activations differ,
so the whole net is fine-tuned).
python -m det.yolo.train_yolo --gpu 2 --data det_runs/yolo_ds/data.yaml --out det_runs/yolo_v8n"""
import argparse, os, sys
from pathlib import Path

ROOT = Path("/data/nas07_new/PersonalData/phuocthien")
BASE_YAML = "yolov8.yaml"


def make_model(out_dir):
    """ReLU YOLOv8n (nc=1) with COCO weights loaded where shapes match."""
    import ultralytics, yaml
    from ultralytics import YOLO
    src = Path(ultralytics.__file__).parent / "cfg/models/v8" / BASE_YAML
    cfg = yaml.safe_load(open(src))
    cfg["nc"] = 1
    cfg["activation"] = "nn.ReLU()"
    dst = Path(out_dir) / "yolov8n_relu.yaml"          # file name carries the 'n' scale
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
    y = YOLO(str(dst), task="detect")
    y.load(str(ROOT / "yolov8n.pt"))
    return y


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--data", default=str(ROOT / "det_runs/yolo_ds/data.yaml"))
    ap.add_argument("--out", default=str(ROOT / "det_runs/yolo_v8n"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=1.0, help="ultralytics dataset fraction (smoke runs)")
    a = ap.parse_args(argv)
    if a.gpu in ("0", "7"):
        sys.exit("GPUs 0 and 7 belong to the user -- refusing")
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    import torch
    if not torch.cuda.is_available():
        sys.exit("CUDA not available -- refusing to fall back to CPU")
    y = make_model(a.out)
    y.train(data=a.data, imgsz=640, epochs=a.epochs, batch=a.batch, device=0, workers=a.workers,
            project=a.out, name="train", exist_ok=True, close_mosaic=10, fraction=a.fraction,
            pretrained=True, amp=True, plots=False, seed=0, deterministic=False)


if __name__ == "__main__":
    main()
