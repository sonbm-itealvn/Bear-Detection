"""Delivery package for the V19 + box head build (Task 15).
python -m det.package --ckpt det_runs/s16/best.pth --tag V19det_YYYYMMDD"""
import argparse, hashlib, json, re, shutil, zipfile
from pathlib import Path

import torch

ROOT = Path("/data/nas07_new/PersonalData/phuocthien")
V19_MS, V19_COMPILER_MS = 70.0, 29.96          # device-measured vs compiler estimate for V19


def _md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def _outputs(bitinfo):
    rows = []
    for k, v in bitinfo.items():
        if v.get("is_network_out"):
            g = v.get("ori_graph_name", "")
            name = "heat" if "/head/heat/" in g else "box" if "/head/box/" in g else "output"
            rows.append((name, k, v.get("output0_bitdepth"), v.get("output0_frac_bit_num")))
    return sorted(rows, key=lambda r: ["output", "heat", "box"].index(r[0]))


def _readme(tag, md5, size, ms, hc, outs, chip, ev):
    s, (gh, gw) = hc["stride"], {16: (23, 40), 8: (45, 80)}[hc["stride"]]
    dev_lo, dev_hi = V19_MS + 1.5 * (ms - V19_COMPILER_MS), V19_MS + 2.0 * (ms - V19_COMPILER_MS)
    L = [f"# Bear detector with boxes for NT98560: {tag}", "",
         f"`nvt_model.bin`: md5 `{md5}`, {size:,} bytes, **{ms:.2f} ms per frame** (compiler estimate; "
         f"V19 alone was {V19_COMPILER_MS} ms). On the device V19 measured {V19_MS:.0f} ms; this build is expected at "
         f"about {dev_lo:.0f}-{dev_hi:.0f} ms (compiler deltas have understated the device by 1.5-2x).", "",
         "## What it is", "",
         "V19 (RegNetX-800MF, delivered 14/10) with its classifier untouched, plus a box head on the same backbone. "
         "The classifier output is bit-identical to V19 (verified below), so existing bear/no-bear logic keeps working.", "",
         "Input is unchanged: 640x360, BGR, `(BGR - [104,116,124]) / 255`, the camera frame letterboxed "
         "(scale to fit, centred, black bars).", "",
         "## Outputs", "", "| tensor | shape | bits | frac bits | meaning |", "|---|---|---|---|---|"]
    meaning = {"output": "classifier logits, index 1 = bear (same as V19)",
               "heat": f"bear-centre logit per {s}-px cell (sigmoid = probability)",
               "box": f"log of the distances from the cell centre to the left, top, right, bottom edges, in units of {s} px"}
    shape = {"output": "1x2", "heat": f"1x1x{gh}x{gw}", "box": f"1x4x{gh}x{gw}"}
    for n, k, b, f in outs:
        L.append(f"| `{n}` (`{k}`) | {shape[n]} | {b} | {f} | {meaning[n]} |")
    L += ["", "## Decoding (C reference in `c/`)", "",
          "`bear_det_decode()` turns the three outputs into boxes in ORIGINAL frame pixels:",
          "1. only when the classifier says bear (`logit[1] > logit[0]`);",
          "2. cells that are a 3x3 local maximum of `heat` with probability >= 0.3, at most 20;",
          "   if none reaches 0.3 the strongest peak is still returned (the classifier said bear);",
          f"3. box = cell centre ((j+0.5)*{s}, (i+0.5)*{s}) -/+ exp(box)*{s}, mapped back through the letterbox and clipped;",
          "4. non-maximum suppression at IoU 0.5.",
          "The C function takes FLOAT tensors: convert each int16 output q to q * 2^-frac (frac bits in the table above) first.",
          "`c/test_main.c` + `test_vectors/` reproduce `expected.txt` (chip outputs, decoded by the Python reference).", "",
          "## Verification on the bit-accurate simulator", ""]
    c, d, t, p = chip["classifier"], chip["dark"], chip["detection"], chip["c_parity"]
    L += ["| check | result | pass |", "|---|---|---|",
          f"| classifier vs V19, {c['n']} val images | max diff {c['max_abs']}, {c['argmax_changes']} decisions changed | {'yes' if c['pass'] else 'NO'} |",
          f"| dark negatives called bear | {d['fp']}/{d['n_neg']} (highest margin {d['max_neg_margin']:.2f}) | {'yes' if d['pass'] else 'NO'} |",
          f"| boxes chip vs FP32 | median IoU {t['median_iou_chip_vs_fp32']:.3f}, AP50 {t['ap50_fp32']:.3f} -> {t['ap50_chip']:.3f} | {'yes' if t['pass'] else 'NO'} |",
          f"| C decoder vs Python on chip outputs | {p['n']} vectors, max {p['max_px']:.3f} px | {'yes' if p['pass'] else 'NO'} |", ""]
    if not t["pass"]:
        L += [f"The box check misses its median-IoU target (0.95): the chip's boxes overlap the FP32 boxes with "
              f"median IoU {t['median_iou_chip_vs_fp32']:.3f}, while box accuracy against the human labels is almost "
              f"unchanged (AP50 {t['ap50_fp32']:.3f} -> {t['ap50_chip']:.3f}, AP@[.5:.95] {t['ap_fp32']:.3f} -> "
              f"{t['ap_chip']:.3f}). The IoU is measured on the boxes that are shown (score >= 0.3). The head reads the "
              "backbone's last stage at the same 8-bit precision the V19 classifier uses, and its own first layer "
              "(`lat4`) is also 8-bit; raising the backbone side would change the classifier, which this build must "
              "not do (a head-only precision change is a possible later improvement).", ""]
    if ev:
        L += ["## Box accuracy (FP32, human-reviewed test half of the eval set)", "",
              f"AP50 {ev['ap50']:.3f}, AP@[.5:.95] {ev['ap']:.3f}, recall at 0.1 false boxes per non-bear image "
              f"{ev['rec_fppi01']:.3f}, median IoU {ev['iou']['median']:.3f}, {ev['n_gt']} bears.", "",
              "| source | AP50 |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in ev["ap50_domain"].items()] + \
             ["", "| bear size on the network input (sqrt of area) | AP50 |", "|---|---|"] + \
             [f"| {k} ({ {'small': '< 32 px', 'medium': '32-96 px', 'large': '> 96 px'}[k] }) | {v:.3f} |" for k, v in ev["ap50_size"].items()] + [""]
    L += ["## Known limits", "",
          "- Box accuracy is measured on still images, including camera-filmed screen captures ('screen'), "
          "not on live trail-camera footage.",
          "- Small, distant bears are the weakest size bucket.",
          "- AP is computed over all boxes down to score 0.05; the decoder shows boxes from 0.3.", "",
          "## Files", "", "| file | contents |", "|---|---|",
          "| `nvt_model.bin` | deploy this |", "| `deploy_float.onnx` | float model it was compiled from |",
          "| `gen_config.txt`, `mean_data.txt` | compile configuration |",
          "| `quan_info_bitinfo.json` | per-layer quantization |", "| `checkpoint_det.pth` | PyTorch weights |",
          "| `c/` | C99 decoder + test program |", "| `test_vectors/` | chip outputs + expected boxes |",
          "| `decode_ref.py`, `geometry.py` | Python reference decoder |"]
    return "\n".join(L) + "\n"


def package(ckpt, build_dir, model_dir, out_root, tag, config_dir=None):
    run = Path(ckpt).parent
    d = Path(out_root) / tag
    if d.exists():
        shutil.rmtree(d)
    (d / "c").mkdir(parents=True)
    shutil.copy2(Path(build_dir) / "sdk/nvt_model.bin", d)
    shutil.copy2(Path(build_dir) / "gentool/quan_info_bitinfo.json", d)
    for f in ("deploy_float.onnx", "mean_data.txt"):
        if (Path(model_dir) / f).exists():
            shutil.copy2(Path(model_dir) / f, d)
    if config_dir and (Path(config_dir) / "gen_config.txt").exists():
        shutil.copy2(Path(config_dir) / "gen_config.txt", d)
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    torch.save({"model_state_dict": raw["model_state_dict"], "config": raw["config"]}, d / "checkpoint_det.pth")
    for f in ("bear_det_decode.h", "bear_det_decode.c", "test_main.c"):
        shutil.copy2(ROOT / "det/c" / f, d / "c")
    (d / "decode_ref.py").write_text((ROOT / "det/decode.py").read_text()
                                     .replace("from det.geometry import", "from geometry import"))
    shutil.copy2(ROOT / "det/geometry.py", d / "geometry.py")
    shutil.copytree(run / "test_vectors", d / "test_vectors")
    prof = (Path(build_dir) / "gentool/profile.txt").read_text(errors="ignore")
    ms = float(re.search(r"Simulation total time\s*:\s*([\d.]+)", prof).group(1))
    outs = _outputs(json.load(open(Path(build_dir) / "gentool/quan_info_bitinfo.json")))
    chip = json.load(open(run / "chip_report.json"))
    ev = json.load(open(run / "eval_test.json")) if (run / "eval_test.json").exists() else None
    binf = d / "nvt_model.bin"
    (d / "README.md").write_text(_readme(tag, _md5(binf), binf.stat().st_size, ms, raw["config"]["det_head"], outs, chip, ev))
    small = {"nvt_model.bin", "README.md"}
    for zname, keep in ((f"{tag}.zip", None), (f"{tag}_small.zip", small)):
        with zipfile.ZipFile(Path(out_root) / zname, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(d.rglob("*")):
                rel = p.relative_to(d)
                if p.is_file() and (keep is None or str(rel) in keep or rel.parts[0] in ("c", "test_vectors")):
                    z.write(p, f"{tag}/{rel}")
    print(f"packaged {d} (md5 {_md5(binf)})")
    return d


def main():
    import novaic_eval as ne
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--name", help="build name used by verify_chip --name")
    a = ap.parse_args()
    hc = torch.load(a.ckpt, map_location="cpu", weights_only=False)["config"]["det_head"]
    name = a.name or f"det_v19_s{hc['stride']}_{hc['up']}"
    package(a.ckpt, ne.NV / "output" / name, ne.NV / "input/model/customer" / name, ROOT / "det_deliver", a.tag,
            config_dir=ne.NV / "config" / name / "cnn25")


if __name__ == "__main__":
    main()
