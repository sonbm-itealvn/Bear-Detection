"""Measure a checkpoint the way the NT98560 actually runs it.

This replaces `ptq_minmax_check.py`, which modelled the wrong quantizer. The
device does not use per-tensor affine INT8 with a min/max scale; it uses
power-of-two fixed point (`frac_bit_num`) with **saturation**, 12-bit weights,
16-bit conv accumulators, and Cross-Layer Equalization enabled in the compiler
(`cle_en = 1`). Those two schemes fail in opposite directions, so min/max
numbers said "collapse" where the hardware is nearly lossless.

Rather than emulate that, this drives Novatek's own compiler and bit-accurate
simulator, which ship inside `novaic_toolchain/` on this machine:

    checkpoint -> ONNX -> compiler.nvtai -> simulator.nvtai -> report

The simulator runs the compiled `nvt_model.bin` through the CNN/NUE engine
emulator and writes, per image, a per-layer `cosine(fixed, float)` next to the
FP32 reference -- the real version of what `ptq_where_it_dies.py` approximated.

Two environment facts this script encodes, both learned the hard way:
  * the compiler needs `deploy_float.onnx` beside `deploy.onnx`, and `--chip 56x`;
  * `simulator/emu52X/emu` is a 32-bit x86 binary, so the host needs an i386
    loader -- see `setup_i386_emu.sh`.

Simulation costs ~65 s/image single-threaded, so --jobs shards the test list
across parallel simulator processes, each with its own output directory.

    python novaic_eval.py --checkpoint <ckpt.pth> --name ep22 --n-eval 100 --jobs 6
    python novaic_eval.py --name ep22 --stage report
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path("/data/nas07_new/PersonalData/phuocthien")
DS = BASE / "bear_dataset_v_final_final"
NV = BASE / ("novaic_toolchain/release/release/ai_tool/novatek/novaic/"
             "test-tutorial/nvtai_tool")
TC = BASE / "novaic_toolchain/release/release/ai_tool/novatek/novaic/toolchain"
BIN = TC / "closeprefix/bin"
I386 = BASE / "novaic_i386/root/usr/lib"
# folder with the pool of calibration images (500 are sampled); override with the CALIB_SRC environment variable
CALIB_SRC = Path(os.environ.get("CALIB_SRC", str(BASE / "BearDetection_mobilenetv2_relu_novaic_640x360_17_6/calibration_images")))
CHIP = "56x"
NET_H = 360   # network input height; 384 for stride-32 detectors (YOLO probe)
# Mean subtraction format. Training subtracts B=104 G=116 R=124 from a BGR
# tensor. Every build up to V17 compiled with fmt_type=1 (RGB) plus a
# newline-separated mean file while the preprocessor outputs BGR (out_fmt=3),
# so the device subtracted 124 from B and 104 from R. Found by the
# bear_experiments_3009 audit (05/10); confirmed here on V17's 398 sim images:
# FP32 with the reversed mean gives recall 88.4% vs 94.5% correct, and tracks
# the simulator (85.9%) better (margin corr 0.959 vs 0.931). So ~6 of the 8.5
# recall points "lost to quantization" were this. "bgr" is the fix; keep
# "legacy-rgb" only to reproduce old numbers.
MEAN_FMT = "bgr"
H, W = 360, 640
HEAD_CONV = False   # set by --head-conv


def env():
    e = dict(os.environ)
    # the compiler shells out to a bare `python3` that must have onnx + onnxsim:
    # put this interpreter's env first so a non-activated shell still works
    e["PATH"] = str(Path(sys.executable).parent) + ":" + e.get("PATH", "")
    e["LD_LIBRARY_PATH"] = ":".join([
        "/opt/miniconda/envs/novaic/lib",
        str(TC / "repo/proprietary/opencv/lib"),
        e.get("LD_LIBRARY_PATH", ""),
    ])
    return e


def run(cmd, log, cwd=NV):
    with open(log, "w") as f:
        return subprocess.run(cmd, cwd=cwd, env=env(), stdout=f,
                              stderr=subprocess.STDOUT).returncode


# ---------------------------------------------------------------- export
def export_timm_onnx(sd, backbone, dst):
    """Checkpoints trained with OVERRIDE_BACKBONE=<timm name>: rebuild the same
    wrapper as the trainer (features.m = timm model, classifier = Dropout+Linear),
    load strictly, reparameterize MobileOne/RepVGG branches into single convs,
    and export the pooled classifier."""
    import os as _os
    _os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    import timm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = timm.create_model(backbone, pretrained=False, num_classes=0)
            self.classifier = nn.Sequential(nn.Dropout(p=0.3),
                                            nn.Linear(self.m.num_features, 2))

        def forward(self, x):
            f = self.m.forward_features(x)
            if HEAD_CONV:
                # classifier as a 1x1 conv BEFORE the pool: exact in FP32, and the
                # 672-ch map then feeds a conv (16-bit input with in_hp) instead of
                # the pool, whose input the 56x always keeps 8-bit (step 0.5 here)
                # identity 2->2 Gemm at the end: the simulator only dumps the
                # network output when the last layer is a Gemm (a Flatten-terminated
                # graph runs but writes no float/<out>.bin -- 13/10)
                return self.idfc(torch.flatten(F.adaptive_avg_pool2d(self.head(f), 1), 1))
            return self.classifier(torch.flatten(F.adaptive_avg_pool2d(f, 1), 1))

    net = Net()
    for mod in list(net.modules()):          # same ReLU6 swap as the trainer
        for cname, child in list(mod.named_children()):
            if isinstance(child, (nn.ReLU6, nn.Hardtanh)):
                setattr(mod, cname, nn.ReLU(inplace=True))
    clean = {k.replace("module.", "").replace("features.m.", "m.", 1): v
             for k, v in sd.items() if not k.startswith("projection.")}
    net.load_state_dict(clean, strict=True)
    net.eval()
    if HEAD_CONV:
        lin = net.classifier[1]
        net.head = nn.Conv2d(lin.in_features, 2, 1)
        net.head.weight.data.copy_(lin.weight.data[:, :, None, None])
        net.head.bias.data.copy_(lin.bias.data)
        net.idfc = nn.Linear(2, 2)
        net.idfc.weight.data.copy_(torch.eye(2))
        net.idfc.bias.data.zero_()
        print("  head: Conv1x1 -> GlobalAveragePool (classifier moved before the pool)")
    if backbone.startswith(("mobileone", "repvgg")):
        from timm.utils.model import reparameterize_model
        x = torch.randn(1, 3, H, W)
        before = net(x)
        net = reparameterize_model(net)
        diff = (net(x) - before).abs().max().item()
        print(f"  reparameterized {backbone}: max |logit diff| {diff:.2e}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, torch.randn(1, 3, H, W), str(dst), opset_version=12,
                      export_params=True, do_constant_folding=True,
                      input_names=["input"], output_names=["output"])
    import onnx
    from onnxsim import simplify
    sm, ok = simplify(onnx.load(str(dst)))
    if ok:
        onnx.save(sm, str(dst))
    print(f"  exported {dst} ({backbone})")


def export_onnx(ckpt, dst):
    import torch
    import torch.nn as nn
    from torchvision import models

    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and (raw.get("config") or {}).get("det_head"):
        from det.export import export_det_onnx
        return export_det_onnx(raw, dst)
    backbone = (raw.get("config") or {}).get("backbone", "torchvision_mobilenet_v2") \
        if isinstance(raw, dict) else "torchvision_mobilenet_v2"
    if backbone != "torchvision_mobilenet_v2":
        return export_timm_onnx(raw.get("model_state_dict", raw), backbone, dst)
    sd = raw
    sd = sd.get("model_state_dict", sd)
    skip = ("quant", "dequant", "observer", "scale", "zero_point",
            "fake_quant", "activation_post_process")
    clean = {}
    for k, v in sd.items():
        if any(p in k for p in skip):
            continue
        k = k.replace("module.", "")
        clean[k[6:] if k.startswith("model.") else k] = v

    m = models.mobilenet_v2(weights=None)

    def swap(mod):
        for n, c in mod.named_children():
            if isinstance(c, nn.ReLU6):
                setattr(mod, n, nn.ReLU(inplace=True))
            else:
                swap(c)
    swap(m)
    m.classifier = nn.Sequential(nn.Dropout(p=0.3),
                                 nn.Linear(m.classifier[1].in_features, 2))
    # A QAT checkpoint has had BN folded into the convs, so it carries 52
    # conv BIASES that plain MobileNetV2 convs do not have (bias=False) and
    # no BN statistics at all. strict=False drops every one of those biases
    # into `unexpected_keys` and says nothing -- this function used to print
    # only `missing_keys`, so it would have compiled a model with all 52
    # folded biases silently discarded and reported the simulator score as if
    # it meant something. Run `export_qat_to_plain.py` first: it folds the
    # bias into the following BatchNorm's beta, where BN(x) = x + beta
    # reproduces it exactly.
    r = m.load_state_dict(clean, strict=False)
    dropped = [k for k in r.unexpected_keys if k.endswith(".bias")]
    if dropped:
        raise SystemExit(
            f"  REFUSING: {len(dropped)} folded conv biases would be dropped "
            f"(e.g. {dropped[:2]}).\n"
            f"  This looks like a QAT checkpoint. Convert it first:\n"
            f"    python export_qat_to_plain.py --qat {ckpt} --out <plain.pth>\n"
            f"  then pass the converted file to --checkpoint.")
    if r.unexpected_keys:
        print(f"  WARNING unexpected keys: {r.unexpected_keys[:5]}")
    if r.missing_keys:
        print(f"  WARNING missing keys: {r.missing_keys[:5]}")
    m.eval()

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(m, torch.randn(1, 3, H, W), str(dst), opset_version=12,
                      export_params=True, do_constant_folding=False,
                      input_names=["input"], output_names=["output"])
    try:
        import onnx
        from onnxsim import simplify
        sm, ok = simplify(onnx.load(str(dst)))
        if ok:
            onnx.save(sm, str(dst))
            print("  onnx simplified")
    except Exception as e:
        print(f"  (skipping simplify: {e})")
    print(f"  exported {dst}")


# ---------------------------------------------------------------- config
GEN_CONFIG = """[path/model_dir] = {model_dir}
[path/mean_path] = {model_dir}/mean_data.txt
[path/ref_img_dir] = {ref_dir}
[path/ref_list_path] = {ref_dir}/ref_img_list.txt
[path/out_dir] = {out_root}
[ref_data/num] = {n_calib}
[multiscale/en] = 0
[preproc/in/fmt] = 2
[preproc/in/type] = 1
[preproc/in/frac_bit_num] = 0
[preproc/in/width] = 640
[preproc/in/height] = {net_h}
[preproc/in/channel] = 2
[preproc/in/batch] = 1
[preproc/in/time] = 1
[preproc/resize/mode] = 0
[preproc/resize/width] = 640
[preproc/resize/height] = {net_h}
[preproc/meansub/en] = 1
[preproc/meansub/mode] = 0
[preproc/meansub/fmt] = 0
[preproc/meansub/fmt_type] = {mean_fmt_type}
[preproc/normalize/en] = 1
[preproc/normalize/scale] = 0.00392157
[preproc/out_fmt] = 3
[postproc/en] = 1
[precision/in_hp/conv_en] = {in_hp}
[precision/in_hp/bnscale_en] = {in_hp}
[precision/in_hp/deconv_en] = 0
[precision/in_hp/fc_en] = {in_hp}
[precision/in_hp/eltwise_en] = {in_hp}
[precision/in_hp/roipool_en] = 0
[precision/balance_weight/out_range_ratio] = {out_range_ratio:.6f}
[precision/ref_relu_outval_en] = 1
[precision/cle_en] = {cle}
[precision/calib_mode] = {calib_mode}
[precision/percentile] = {percentile:.6f}
[precision/out_hp/eltwise_en] = {out_hp_eltwise}
[precision/out_hp/conv_en] = {out_hp_conv}
[precision/avgpool_noclamp_en] = {avgpool_noclamp}
[compression/method/quan_en] = 1
[compression/method/vlc_en] = 0
[performance/mode] = 0
[performance/shrink_en] = 1
[performance/cnn_engine_mode] = 0
[performance/job_schedule_en] = 0
"""

SIM_CONFIG = """[path/label_path] = {data_dir}/label.txt
[path/model_dir] = {model_dir}
[path/test_img_dir] = {img_dir}
[path/test_gt_path] = {data_dir}/test_ground_truth.txt
[path/test_list_path] = {data_dir}/test_img_list.txt
[dataset] = {name}
[path/out_dir] = {out_dir}
[src/imgfmt] = 0
[src/is_preproc_out_bin] = 0
[src/bintype] = 1
[src/width] = 640
[src/height] = {net_h}
[src/channel] = 3
[src/batch] = 1
[src/time] = 1
[src/resize_en] = 1
[src/padding_en] = 1
[src/crop_en] = 0
[backend/convert_ext_to] = CPU
"""


def write_cfg(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


# ---------------------------------------------------------------- staging
def stage_data(name, n_eval, n_calib, seed):
    rng = random.Random(seed)
    model_dir = NV / "input/model/customer" / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "mean_data.txt").write_text(
        "104,116,124\n" if MEAN_FMT == "bgr" else "104\n116\n124\n")

    ref_dir = NV / "input/data" / f"{name}_calib/reference_images"
    if not (ref_dir / "ref_img_list.txt").exists():
        ref_dir.mkdir(parents=True, exist_ok=True)
        names = sorted(os.listdir(CALIB_SRC))
        rng.shuffle(names)
        picked = names[:n_calib]
        for f in picked:
            shutil.copy2(CALIB_SRC / f, ref_dir / f)
        (ref_dir / "ref_img_list.txt").write_text("\n".join(picked) + "\n")
    print(f"  calib images: {len(list(ref_dir.glob('*.jpg')))}")

    data_dir = NV / "input/data" / f"{name}_test"
    img_dir = data_dir / "test_images"
    if not (data_dir / "test_img_list.txt").exists():
        img_dir.mkdir(parents=True, exist_ok=True)
        items = []
        for cls, lab in (("non_bear", 0), ("bear", 1)):
            d = DS / "val" / cls
            fs = sorted(os.listdir(d))
            rng.shuffle(fs)
            for f in fs[:n_eval // 2]:
                # flatten name so class is recoverable from the filename alone
                tgt = f"{cls}__{f}"
                copy_as_rgb_jpeg(d / f, img_dir / tgt)
                items.append((tgt, lab))
        rng.shuffle(items)
        (data_dir / "test_img_list.txt").write_text(
            "\n".join(i[0] for i in items) + "\n")
        (data_dir / "test_ground_truth.txt").write_text(
            "\n".join(str(i[1]) for i in items) + "\n")
        (data_dir / "label.txt").write_text("non_bear\nbear\n")
    n = len((data_dir / "test_img_list.txt").read_text().split())
    print(f"  eval images : {n}")
    return model_dir, ref_dir, data_dir, img_dir


def letterbox_640x360(im):
    """Same letterbox as the trainers (score_darkfur.Letterbox /
    train_qat_novaic.ResizeWithPadding): scale to fit, centre, black pad."""
    from PIL import Image
    im = im.convert("RGB")
    s = min(W / im.width, H / im.height)
    nw, nh = max(1, int(im.width * s)), max(1, int(im.height * s))
    out = Image.new("RGB", (W, H), (0, 0, 0))
    out.paste(im.resize((nw, nh), Image.BILINEAR), ((W - nw) // 2, (H - nh) // 2))
    return out


def copy_as_rgb_jpeg(src, dst):
    """Stage a test image for the simulator as a LETTERBOXED 640x360 RGB JPEG.

    The simulator stretches whatever it is given to 640x360 -- [src/padding_en]=1
    does not letterbox -- while the trainers and the camera (16:9 frames) never
    stretch. 357 of the 400 V19 test images are not 16:9; on them FP32 run on a
    STRETCHED input reproduced the chip (corr 0.996, 8 of 9 'chip-only' bear
    misses), i.e. most of the measured 'quantization loss' was this artefact
    (13/10). Also re-encodes GIF/PNG/greyscale files named *.jpg, which abort a
    whole simulator shard."""
    from PIL import Image
    with Image.open(src) as im:
        letterbox_640x360(im).save(dst, "JPEG", quality=95)


def stage_compile(name, model_dir, ref_dir, n_calib, cle, orr, in_hp, logdir,
                  calib_mode=0, percentile=99.99, out_hp_eltwise=0,
                  out_hp_conv=0, avgpool_noclamp=0):
    write_cfg(NV / "config" / name / "cnn25/gen_config.txt",
              GEN_CONFIG.format(net_h=NET_H, model_dir=model_dir, ref_dir=ref_dir,
                                out_root=NV / "output", n_calib=n_calib,
                                cle=cle, out_range_ratio=orr, in_hp=in_hp,
                                calib_mode=calib_mode, percentile=percentile,
                                out_hp_eltwise=out_hp_eltwise,
                                out_hp_conv=out_hp_conv,
                                avgpool_noclamp=avgpool_noclamp,
                                mean_fmt_type=4 if MEAN_FMT == "bgr" else 1))
    src = model_dir / "deploy_float.onnx"
    shutil.copy2(src, model_dir / "deploy.onnx")
    log = logdir / f"compile_{name}.log"
    rc = run([str(BIN / "compiler.nvtai"), "--config-dir", str(NV),
              "--pattern-name", name, "--use-onnx", "--chip", CHIP], log)
    ok = (NV / "output" / name / "sdk/nvt_model.bin").exists()
    print(f"  compile rc={rc} model={'OK' if ok else 'MISSING'}  log={log}")
    return ok


def stage_sim(name, model_dir, data_dir, img_dir, jobs, logdir, per_layer=False):
    names = (data_dir / "test_img_list.txt").read_text().split()
    gts = (data_dir / "test_ground_truth.txt").read_text().split()
    shards = [[] for _ in range(jobs)]
    for i, (n, g) in enumerate(zip(names, gts)):
        shards[i % jobs].append((n, g))

    procs = []
    for k, sh in enumerate(shards):
        if not sh:
            continue
        sd = NV / "input/data" / f"{name}_test_s{k}"
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "test_img_list.txt").write_text("\n".join(x[0] for x in sh) + "\n")
        (sd / "test_ground_truth.txt").write_text("\n".join(x[1] for x in sh) + "\n")
        (sd / "label.txt").write_text("non_bear\nbear\n")

        out = NV / "output" / f"{name}_s{k}"
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(NV / "output" / name, out)

        pat = f"{name}_test_s{k}"
        write_cfg(NV / "config" / pat / "cnn25/sim_config.txt",
                  SIM_CONFIG.format(net_h=NET_H, data_dir=sd, model_dir=model_dir,
                                    img_dir=img_dir, name=pat, out_dir=out))
        # the simulator parses the compile-side gen_config too (preprocessing,
        # quantization settings), so every sim pattern dir needs a copy of it
        shutil.copy2(NV / "config" / name / "cnn25/gen_config.txt",
                     NV / "config" / pat / "cnn25/gen_config.txt")
        log = logdir / f"sim_{name}_s{k}.log"
        procs.append(subprocess.Popen(
            [str(BIN / "simulator.nvtai"), "--config-dir", str(NV),
             "--pattern-name", pat, "--chip", CHIP,
             # per_layer keeps every layer's fixed-point dump and its
             # cosine(fixed, float) line in the log; the default keeps only the
             # classifier output because a full dump is large and slow.
             "--emu-onlyRstLayer", "0" if per_layer else "1",
             "--delete-bin", "0" if per_layer else "1",
             "--openmp-en", "4", "--verbose", "3"],
            cwd=NV, env=env(), stdout=open(log, "w"), stderr=subprocess.STDOUT))
    print(f"  launched {len(procs)} simulator shards")
    rcs = [p.wait() for p in procs]
    print(f"  shard return codes: {rcs}")


# ---------------------------------------------------------------- report
LAYER_RE = re.compile(r"\[\s*(\d+)\]\((\S+)\s*\)\s+(\S+)\s+cosine\(\s*([0-9.]+)\s*\)"
                      r"\s+avg_diff\(\s*([0-9.eE+-]+)\s*\)")


def net_out_file(name):
    """Name of the classifier-output tensor dump. Det builds have 3 network outputs;
    the classifier one is the Gemm (or the identity Gemm of --head-conv builds)."""
    import json
    f = NV / "output" / name / "gentool/quant_input_output_bitinfo.json"
    if f.exists():
        outs = [k for k, v in json.load(open(f)).items() if v.get("is_network_out")]
        gemm = [k for k in outs if k.startswith("Gemm")]
        if gemm or outs:
            return f"{(gemm or outs)[0]}.bin"
    return "Gemm_output_Y.bin"


def stage_report(name, jobs):
    import numpy as np
    per_layer, rows = {}, []
    OUTF = net_out_file(name)
    for k in range(jobs):
        out = NV / "output" / f"{name}_s{k}"
        if not out.exists():
            continue
        for f in sorted((out / "simtool").glob("layer_diff_*.txt")):
            img = f.name.split("_", 3)[3].rsplit(".txt", 1)[0]
            for m in LAYER_RE.finditer(f.read_text()):
                per_layer.setdefault(m.group(3), []).append(float(m.group(4)))
            # NOTE: debug/sim_layer_<img>/float/ is NOT an FP32 reference despite
            # the name -- its Gemm_output_Y.bin holds the FIXED result dequantized
            # (verified: it equals the int16 NUE result divided by 2^frac). The
            # true FP32 golden lives in debug/golden_tensor/. Reading the wrong
            # one once turned a 38-point quantization loss into "quantization is
            # free", so both are read here and reported side by side.
            dev = out / f"debug/sim_layer_{img}/float/{OUTF}"
            ref = out / f"debug/sim_layer_{img}/golden_tensor/{OUTF}"
            if not ref.exists():
                ref = out / f"debug/golden_tensor/{OUTF}"
            if dev.exists():
                v = np.fromfile(dev, dtype=np.float32)
                g = (np.fromfile(ref, dtype=np.float32)
                     if ref.exists() else np.array([np.nan, np.nan]))
                if v.size >= 2:   # the file is zero-padded to 16 floats in some runs
                    rows.append((img, float(v[0]), float(v[1]),
                                 float(g[0]), float(g[1])))

    if not rows:
        print("no results found -- run the sim stage first")
        return

    print(f"\nimages with output: {len(rows)}")
    print(f"{'layer':<46}{'mean cos':>10}{'min cos':>10}{'n':>6}")
    for lname, cs in sorted(per_layer.items(), key=lambda kv: float(np.mean(kv[1])))[:12]:
        print(f"{lname[:44]:<46}{np.mean(cs):>10.5f}{np.min(cs):>10.5f}{len(cs):>6}")

    # class order is resolved empirically: whichever assignment of the two
    # output units to (non_bear, bear) agrees better with the filenames wins.
    truth = np.array([1 if r[0].startswith("bear__") else 0 for r in rows])
    logits = np.array([[r[1], r[2]] for r in rows])   # on-device (quantized)
    for flip in (False, True):
        pred = (logits[:, 0] > logits[:, 1]).astype(int) if flip else \
               (logits[:, 1] > logits[:, 0]).astype(int)
        tp = int(((pred == 1) & (truth == 1)).sum())
        fp_ = int(((pred == 1) & (truth == 0)).sum())
        fn = int(((pred == 0) & (truth == 1)).sum())
        tn = int(((pred == 0) & (truth == 0)).sum())
        rec = tp / max(tp + fn, 1)
        pre = tp / max(tp + fp_, 1)
        tag = "unit0=bear" if flip else "unit1=bear"
        print(f"  ON-DEVICE  {tag:<12} recall {100*rec:6.2f}%  precision "
              f"{100*pre:6.2f}%  (TP={tp} TN={tn} FP={fp_} FN={fn})")
    print("\nNOTE: the recall/precision above are the QUANTIZED, on-device "
          "results.\nCompare them against PyTorch FP32 on the same files to see "
          "what quantization costs.\nRun the simulator WITHOUT --emu-onlyRstLayer "
          "to get per-layer cosines: damage on this\narchitecture starts at "
          "features.0, not deep in the network.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--n-calib", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cle", type=int, default=1, help="precision/cle_en")
    ap.add_argument("--out-range-ratio", type=float, default=1000.0)
    # GEN_REF_CALIB_MODE from the vendor header ParseGenConfig.h:
    #   0 MAX (min/max)  1 L2  2 KLD  3 CLIP  4 PERCENTILE
    # The compiler has defaulted to 0 on every compile this project has ever
    # run -- `calib_mode` appears in none of its ~350 gen_config.txt files --
    # even though a PyTorch simulation of p99.9 clipping on this architecture
    # measured 96%/96% against 76%/95% for min/max. Unlike cust_quan injection
    # (which force-disables CLE and was abandoned for that reason), this is a
    # separate field and should leave CLE alone; verify that in the compile log.
    ap.add_argument("--calib-mode", type=int, default=0, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--percentile", type=float, default=99.99,
                    help="used when --calib-mode 4")
    # Output-side high precision on eltwise, distinct from --in-hp which is
    # input-side. The residual Add of features.9-13 is the measured damage site.
    ap.add_argument("--out-hp-eltwise", type=int, default=0, choices=[0, 1])
    # Both aimed at the measured damage site: the per-layer probe put
    # features.18 and GlobalAveragePool at cosine ~0.7 on every image, the
    # worst layers in the network and the classifier's direct input.
    ap.add_argument("--out-hp-conv", type=int, default=0, choices=[0, 1],
                    help="16-bit conv OUTPUTS (convOutHpEn); --in-hp is input-side only")
    ap.add_argument("--avgpool-noclamp", type=int, default=0, choices=[0, 1],
                    help="precision/avgpool_noclamp_en")
    ap.add_argument("--per-layer", action="store_true",
                    help="dump every layer and log per-layer cosine(fixed, float)")
    ap.add_argument("--in-hp", type=int, default=0,
                    help="precision/in_hp/* -- ask the compiler for higher-precision "
                         "layer inputs. Worth testing because the measured damage "
                         "starts at features.0, where the compiler picked 8-bit.")
    ap.add_argument("--mean-fmt", default="bgr", choices=["bgr", "legacy-rgb"],
                    help="bgr = correct (fmt_type 4, CSV mean); legacy-rgb = every build up to V17")
    ap.add_argument("--stage", default="all",
                    choices=["all", "export", "compile", "sim", "report"])
    ap.add_argument("--net-h", type=int, default=360,
                    help="network input height (default 360; YOLO probes use 384)")
    ap.add_argument("--head-conv", action="store_true",
                    help="timm backbones: export the classifier as a 1x1 conv before "
                         "GlobalAveragePool (keeps the pre-pool map 16-bit on the 56x)")
    a = ap.parse_args()
    global HEAD_CONV
    HEAD_CONV = a.head_conv
    global MEAN_FMT, NET_H
    MEAN_FMT = a.mean_fmt
    NET_H = a.net_h
    if NET_H != 360 and a.stage in ("all", "sim", "report"):
        # stage_data/copy_as_rgb_jpeg letterbox to 640x360; the simulator would STRETCH them to
        # 640xNET_H (the 13/10 artefact). Stage your own letterboxed images (det/yolo/chip.py).
        sys.exit(f"--net-h {NET_H}: only --stage export/compile are supported here; "
                 f"stage letterboxed 640x{NET_H} test images yourself (see det/yolo/chip.py)")

    logdir = BASE / "novaic_eval_logs"
    logdir.mkdir(exist_ok=True)
    model_dir, ref_dir, data_dir, img_dir = stage_data(
        a.name, a.n_eval, a.n_calib, a.seed)

    if a.stage in ("all", "export"):
        if not a.checkpoint:
            sys.exit("--checkpoint required for the export stage")
        print("[export]")
        export_onnx(a.checkpoint, model_dir / "deploy_float.onnx")
    if a.stage in ("all", "compile"):
        print("[compile]")
        if not stage_compile(a.name, model_dir, ref_dir, a.n_calib,
                             a.cle, a.out_range_ratio, a.in_hp, logdir,
                             a.calib_mode, a.percentile, a.out_hp_eltwise,
                             a.out_hp_conv, a.avgpool_noclamp):
            sys.exit("compile failed")
    if a.stage in ("all", "sim"):
        print("[sim]")
        stage_sim(a.name, model_dir, data_dir, img_dir, a.jobs, logdir, per_layer=a.per_layer)
    if a.stage in ("all", "report"):
        print("[report]")
        stage_report(a.name, a.jobs)


if __name__ == "__main__":
    main()
