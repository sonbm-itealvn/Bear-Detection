"""Delivery package for candidate (b'): V19 decides bear / no-bear (unchanged build), and a
separate YOLOv8n (ReLU) model draws the boxes, run only on frames where V19 says bear.
python -m det.yolo.package_yolo --tag V19_YOLOv8n_boxes_YYYYMMDD"""
import argparse, hashlib, json, re, shutil, zipfile
from pathlib import Path

ROOT = Path("/data/nas07_new/PersonalData/phuocthien")
V19_BIN_MD5 = "a51174d26206be3fab007413fcacb63f"


def _readme(tag, md5, size, ms, chip, gated, outs):
    a, b = gated["a"], gated["b"]
    frac = "\n".join(f"| `{n}` | {bits} | {fr} |" for n, bits, fr in outs)
    return f"""# Bear boxes with a second model (YOLOv8n) for NT98560: {tag}

Two models on the NPU:
1. **V19** (`deliver_ptq_20261014_V19_regnetx_ep58/nvt_model.bin`, md5 `{V19_BIN_MD5}`, unchanged) decides bear / no bear, exactly as today.
2. **This model** (`nvt_model.bin`, md5 `{md5}`, {size:,} bytes, **{ms:.2f} ms per frame** on the compiler) draws the boxes.
   Run it **only on frames where V19 says bear**.

Device time: V19 measured about 70 ms for 29.96 compiler-ms, so expect roughly +35-40 ms on bear frames only.

## Input
- 640x384, BGR, `(BGR - [104,116,124]) / 255`. This is the same mean/scale as V19, so the same preprocessing setup can be used.
- A 16:9 camera frame is resized to 640x360 and placed in the middle of a 640x384 black canvas (12 black rows top and bottom).
- Any other aspect ratio: scale to fit inside 640x384, centred, black bars.

## Outputs (6 tensors, NCHW)

| tensor | shape | meaning |
|---|---|---|
| `box_p3`, `cls_p3` | 64x48x80, 1x48x80 | stride 8 |
| `box_p4`, `cls_p4` | 64x24x40, 1x24x40 | stride 16 |
| `box_p5`, `cls_p5` | 64x12x20, 1x12x20 | stride 32 |

Quantized output format (16-bit integers; float = q * 2^-frac):

| tensor | bits | frac bits |
|---|---|---|
{frac}

- `cls`: the bear logit.
- `box`: 4 sides (left, top, right, bottom) x 16 distribution bins. The distance is the softmax-weighted mean of 0..15, in units of the stride, from the cell centre ((j+0.5)*stride, (i+0.5)*stride).

## Decoding (C reference in `c/`)
`yolo_det_decode()`:
1. keeps cells with sigmoid(cls) >= 0.3, at most the best 256;
2. runs non-maximum suppression at IoU 0.5 and keeps at most 20 boxes;
3. maps them back through the letterbox and clips them to the frame.

With `fallback=1` (use it, since V19 already said bear) the strongest cell that gives a valid box inside the frame is returned even when nothing reaches 0.3.
The C function takes FLOAT tensors: convert each int16 output q to q * 2^-frac first (table above).
`c/yolo_test_main.c` + `test_vectors/` reproduce `expected.txt`, which was built from real chip outputs (C vs Python: max 0.0006 px).

## Verification on the bit-accurate simulator

Human-reviewed test set: {chip['n'] - chip['n_neg']} bear images, {chip['n_neg']} non-bear images. AP is computed over all boxes down to score 0.05; the IoU statistic uses the shown boxes (score >= 0.3).

| | FP32 | chip |
|---|---|---|
| AP50 | {chip['ap50_fp32']:.3f} | {chip['ap50_chip']:.3f} |
| AP@[.5:.95] | {chip['ap_fp32']:.3f} | {chip['ap_chip']:.3f} |

The median IoU between the chip boxes and the FP32 boxes is {chip['median_iou_chip_vs_fp32']:.3f}.

## Compared with the single-model build (V19det, one model with a box head)

Both columns were measured on chip outputs, with boxes shown only when V19 says bear:

| | V19det (box head) | V19 + YOLOv8n (this package) |
|---|---|---|
| AP50 | {a['ap50']:.3f} | {b['ap50']:.3f} |
| AP@[.5:.95] (box tightness) | {a['ap']:.3f} | {b['ap']:.3f} |
| median IoU with the true box | {a['iou']['median']:.2f} | {b['iou']['median']:.2f} |
| extra time | +1.4 compiler-ms, every frame | +{ms:.1f} compiler-ms, bear frames only |
| extra size | +0.43 MB | +{size / 1e6:.2f} MB (second model) |

AP50 by bear size on the network input (sqrt of area): small (< 32 px) {a['ap50_size'].get('small', float('nan')):.3f} vs {b['ap50_size'].get('small', float('nan')):.3f}, medium {a['ap50_size'].get('medium', float('nan')):.3f} vs {b['ap50_size'].get('medium', float('nan')):.3f}, large {a['ap50_size'].get('large', float('nan')):.3f} vs {b['ap50_size'].get('large', float('nan')):.3f}.

Caveat: about two thirds of the reviewed boxes were accepted as proposed by the YOLOv8m labeller. That may favour a YOLO-family model on box tightness (AP@[.5:.95]).

## Files

| file | contents |
|---|---|
| `nvt_model.bin` | YOLOv8n box model |
| `deploy_float.onnx` | the float model it was compiled from |
| `gen_config.txt`, `mean_data.txt` | compile configuration |
| `quan_info_bitinfo.json` | per-layer quantization |
| `best.pt` | ultralytics weights (ReLU YOLOv8n, 1 class) |
| `c/` | C99 decoder + test program |
| `test_vectors/` | chip outputs and the expected boxes |
| `decode_ref.py`, `geometry.py` | Python reference decoder |
"""


def package_yolo(run, build_dir, model_dir, config_dir, gated_json, out_root, tag):
    run, build_dir = Path(run), Path(build_dir)
    d = Path(out_root) / tag
    if d.exists():
        shutil.rmtree(d)
    (d / "c").mkdir(parents=True)
    shutil.copy2(build_dir / "sdk/nvt_model.bin", d)
    shutil.copy2(build_dir / "gentool/quan_info_bitinfo.json", d)
    for f in ("deploy_float.onnx", "mean_data.txt"):
        if (Path(model_dir) / f).exists():
            shutil.copy2(Path(model_dir) / f, d)
    if (Path(config_dir) / "gen_config.txt").exists():
        shutil.copy2(Path(config_dir) / "gen_config.txt", d)
    shutil.copy2(run / "weights/best.pt", d / "best.pt")
    for f in ("bear_det_decode.h", "yolo_det_decode.h", "yolo_det_decode.c", "yolo_test_main.c"):
        shutil.copy2(ROOT / "det/c" / f, d / "c")
    (d / "decode_ref.py").write_text((ROOT / "det/yolo/decode.py").read_text()
                                     .replace("from det.yolo.geometry import", "from geometry import"))
    shutil.copy2(ROOT / "det/yolo/geometry.py", d / "geometry.py")
    shutil.copytree(run / "test_vectors", d / "test_vectors")
    prof = (build_dir / "gentool/profile.txt").read_text(errors="ignore")
    ms = float(re.search(r"Simulation total time\s*:\s*([\d.]+)", prof).group(1))
    binf = d / "nvt_model.bin"
    md5 = hashlib.md5(binf.read_bytes()).hexdigest()
    info = json.load(open(build_dir / "gentool/quan_info_bitinfo.json"))
    order = ["box_p3", "cls_p3", "box_p4", "cls_p4", "box_p5", "cls_p5"]
    outs = [(n, info.get(f"Conv_{n}_Y", {}).get("output0_bitdepth"), info.get(f"Conv_{n}_Y", {}).get("output0_frac_bit_num"))
            for n in order]
    (d / "README.md").write_text(_readme(tag, md5, binf.stat().st_size, ms,
                                         json.load(open(run / "chip_report.json")), json.load(open(gated_json)), outs))
    with zipfile.ZipFile(Path(out_root) / f"{tag}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(d.rglob("*")):
            if p.is_file():
                z.write(p, f"{tag}/{p.relative_to(d)}")
    print(f"packaged {d} (md5 {md5})")
    return d


def main():
    import novaic_eval as ne
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--run", default=str(ROOT / "det_runs/yolo_v8n_v2/train"))
    ap.add_argument("--name", default="yolo_v8n_v2_chip")
    ap.add_argument("--gated", default=str(ROOT / "det_runs/compare_gated_chip_test.json"))
    a = ap.parse_args()
    package_yolo(a.run, ne.NV / "output" / a.name, ne.NV / "input/model/customer" / a.name,
                 ne.NV / "config" / a.name / "cnn25", a.gated,
                 ROOT / "det_deliver", a.tag)


if __name__ == "__main__":
    main()
