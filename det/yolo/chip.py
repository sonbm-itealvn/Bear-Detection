"""Chip check of the YOLO candidate (b'): export (with the chip-preprocessing adapter),
compile at 640x384, simulate the eval-set TEST half letterboxed to 640x384, decode the 6 raw
outputs with det.yolo.decode and compare with the FP32 predictions (preds_test.jsonl).
python -m det.yolo.chip --pt det_runs/yolo_v8n_v2/train/weights/best.pt"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/data/nas07_new/PersonalData/phuocthien")


def output_keys():
    from det.yolo.export_onnx import OUT_NAMES
    return {n: f"Conv_{n}_Y" for n in OUT_NAMES}


def letterbox_384_jpeg(src, dst):
    from det.yolo.eval_yolo import to_input  # noqa: F401  (same geometry, kept for symmetry)
    from det.yolo.geometry import IN_W, IN_H, letterbox_params
    with Image.open(src) as im:
        im = im.convert("RGB")
        nw, nh, px, py = letterbox_params(*im.size)
        c = Image.new("RGB", (IN_W, IN_H), (0, 0, 0))
        c.paste(im.resize((nw, nh), Image.BILINEAR), (px, py))
        c.save(dst, "JPEG", quality=95)


def stage(name, tag, paths):
    import novaic_eval as ne
    d = ne.NV / "input/data" / f"{name}_{tag}"
    if d.exists():
        shutil.rmtree(d)
    (d / "test_images").mkdir(parents=True)
    staged = []
    for i, p in enumerate(paths):
        n = f"{i:05d}__{Path(p).stem}.jpg"
        letterbox_384_jpeg(p, d / "test_images" / n)
        staged.append(n)
    (d / "test_img_list.txt").write_text("\n".join(staged) + "\n")
    (d / "test_ground_truth.txt").write_text("\n".join("0" for _ in staged) + "\n")
    (d / "label.txt").write_text("non_bear\nbear\n")
    return staged


def read(name, tag, img):
    import novaic_eval as ne
    from det import sim_tools as st
    from det.yolo.export_onnx import OUT_NAMES
    base = st._find_img_dir(name, tag, img)
    shapes = {}
    for lvl, (h, w) in zip((3, 4, 5), ((48, 80), (24, 40), (12, 20))):
        shapes[f"box_p{lvl}"] = (64, h, w); shapes[f"cls_p{lvl}"] = (1, h, w)
    out = []
    for n in OUT_NAMES:
        a = np.fromfile(base / "float" / st.dump_name(output_keys()[n]), np.float32)
        out.append(a[:int(np.prod(shapes[n]))].reshape(shapes[n]))
    return out


def main():
    import torch
    import novaic_eval as ne
    from det import sim_tools as st
    from det.labels.store import LabelStore
    from det.verify_chip import compare_detections
    from det.yolo.decode import decode
    from det.yolo.export_onnx import NovaicYolo, export_onnx, load_trained
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--name", default="yolo_v8n_v2_chip")
    ap.add_argument("--shards", type=int, default=8)
    a = ap.parse_args()
    run = Path(a.pt).parent.parent
    ne.NET_H = 384
    md = ne.NV / "input/model/customer" / a.name
    if not (ne.NV / "output" / a.name / "sdk/nvt_model.bin").exists():
        md.mkdir(parents=True, exist_ok=True)
        export_onnx(NovaicYolo(load_trained(a.pt)).eval(), md / "deploy_float.onnx")
        r = subprocess.run([sys.executable, "-u", "novaic_eval.py", "--name", a.name, "--n-eval", "400", "--jobs", "4",
                            "--in-hp", "1", "--calib-mode", "0", "--mean-fmt", "bgr", "--net-h", "384", "--stage", "compile"],
                           cwd=ROOT, capture_output=True, text=True)
        print(f"[compile] rc={r.returncode} {r.stdout[-300:]}", flush=True)
        if r.returncode:
            sys.exit(r.stderr[-1000:])
    binf = ne.NV / "output" / a.name / "sdk/nvt_model.bin"
    prof = (ne.NV / "output" / a.name / "gentool/profile.txt").read_text(errors="ignore")
    import re
    ms = float(re.search(r"Simulation total time\s*:\s*([\d.]+)", prof).group(1))
    print(f"  compiler ms {ms}, bin {binf.stat().st_size} B", flush=True)
    fp32 = {r["path"]: r for r in map(json.loads, open(run / "preds_test.jsonl"))}
    paths = sorted(fp32)
    staged = stage(a.name, "det", paths)
    print("  sim rcs", st.run_sim(a.name, "det", staged, shards=a.shards), flush=True)
    store = LabelStore(ROOT / "det_labels/labels_v1.jsonl")
    es = json.load(open(ROOT / "det_labels/eval_set.json"))["test"]
    rows = {Path(r["path_hint"]).name: r for r in (store.get(s) for s in es["bear"] + es["non_bear"]) if r}
    chip, f32, gt, neg, raw = {}, {}, {}, set(), {}
    for s, p in zip(staged, paths):
        try:
            outs = read(a.name, "det", s)
        except FileNotFoundError:
            print("  missing", s); continue
        m = fp32[p]
        chip[p] = decode(outs, m["w"], m["h"], thr=0.05)
        f32[p] = np.asarray(m["pred"]).reshape(-1, 5)
        raw[p] = outs
        r = rows.get(Path(p).name)
        if "/non_bear/" in p:
            neg.add(p); gt[p] = np.zeros((0, 4))
        elif r is not None and r["status"] not in ("masked", "excluded"):
            gt[p] = np.array([b["xyxy"] for b in r["boxes"]]).reshape(-1, 4)
    keep = [p for p in chip if p in gt]
    rep = compare_detections({p: chip[p] for p in keep}, {p: f32[p] for p in keep}, gt, neg)
    rep.update(compiler_ms=ms, bin_bytes=binf.stat().st_size)
    print("  detection:", rep, flush=True)
    json.dump(rep, open(run / "chip_report.json", "w"), indent=1, default=float)
    np.savez_compressed(run / "chip_raw_test20.npz", **{f"{i}_{k}": v for i, p in enumerate(keep[:20])
                                                         for k, v in zip(("bp3", "cp3", "bp4", "cp4", "bp5", "cp5"), raw[p])},
                        meta=np.array(json.dumps([{"path": p, "w": fp32[p]["w"], "h": fp32[p]["h"]} for p in keep[:20]])))


if __name__ == "__main__":
    main()
