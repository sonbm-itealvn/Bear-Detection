"""Chip verification of a trained det checkpoint (spec §4, Task 14):
 1. classifier invariance: logits on V19's 400 letterboxed val inputs equal the delivered
    V19 build bit for bit;
 2. dark set: 48 dark negatives stay negative (V19: 0/48), and logits equal V19's;
 3. detection: chip boxes vs FP32 boxes on the eval-set TEST half (median IoU >= 0.95,
    AP50 drop <= 0.02);
 4. C decoder parity on 20 chip output vectors (<= 1 px).
python -m det.verify_chip --ckpt det_runs/s16/best.pth"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/nas07_new/PersonalData/phuocthien")
REF = "v19_ep58_lb"
from det.eval_det import EVAL_THR  # noqa: E402  (same threshold as the FP32 eval)


# ---------------------------------------------------------------- pure checks (unit-tested)
def compare_classifier(new, ref):
    keys = sorted(set(new) & set(ref))
    d = [float(np.abs(np.asarray(new[k]) - np.asarray(ref[k])).max()) for k in keys]
    flips = sum(int(np.argmax(new[k]) != np.argmax(ref[k])) for k in keys)
    m = max(d) if d else float("nan")
    return {"n": len(keys), "max_abs": m, "argmax_changes": flips, "pass": bool(keys) and m == 0.0}


def dark_check(logits, negatives):
    marg = [float(logits[k][1] - logits[k][0]) for k in negatives if k in logits]
    fp = sum(m > 0 for m in marg)
    return {"n_neg": len(marg), "fp": int(fp), "max_neg_margin": max(marg) if marg else float("nan"),
            "pass": bool(marg) and fp == 0}


SHOW_THR = 0.3     # the decoder's display threshold; only shown boxes enter the IoU statistic


def compare_detections(chip, fp32, gt, neg=()):
    from det.labels.fusion import iou
    from det.metrics import ap_at, ap_range, iou_stats
    keys = sorted(set(chip) & set(fp32))
    ious = []
    for k in keys:
        c, f = np.asarray(chip[k]).reshape(-1, 5), np.asarray(fp32[k]).reshape(-1, 5)
        f = f[f[:, 4] >= SHOW_THR]
        if len(f):
            ious += list(iou(f[:, :4], c[:, :4]).max(1)) if len(c) else [0.0] * len(f)
    def ap(pred, fn):
        return fn([np.asarray(pred[k]).reshape(-1, 5) for k in keys],
                  [np.asarray(gt.get(k, np.zeros((0, 4)))).reshape(-1, 4) for k in keys])
    r = {"n": len(keys), "n_neg": len(set(neg) & set(keys)),
         "median_iou_chip_vs_fp32": iou_stats(ious)["median"], "p10_iou_chip_vs_fp32": iou_stats(ious)["p10"],
         "ap50_fp32": ap(fp32, lambda p, g: ap_at(p, g, 0.5)), "ap50_chip": ap(chip, lambda p, g: ap_at(p, g, 0.5)),
         "ap_fp32": ap(fp32, ap_range), "ap_chip": ap(chip, ap_range)}
    r["ap50_drop"] = r["ap50_fp32"] - r["ap50_chip"]
    r["pass"] = r["median_iou_chip_vs_fp32"] >= 0.95 and r["ap50_drop"] <= 0.02
    return r


# ---------------------------------------------------------------- orchestration
def compile_build(ckpt, name):
    from det import sim_tools as st
    if (st.NV / "output" / name / "sdk/nvt_model.bin").exists():
        print(f"  {name}: already compiled", flush=True)
        return
    for stage in ("export", "compile"):
        r = subprocess.run([sys.executable, "-u", "novaic_eval.py", "--checkpoint", str(ckpt), "--name", name,
                            "--n-eval", "400", "--jobs", "4", "--in-hp", "1", "--calib-mode", "0",
                            "--mean-fmt", "bgr", "--stage", stage], cwd=ROOT, capture_output=True, text=True)
        print(f"  [{stage}] rc={r.returncode} {r.stdout[-300:]}", flush=True)
        if r.returncode:
            sys.exit(f"{stage} failed: {r.stderr[-800:]}")


def _ref_logits(tag_glob, names):
    from det import sim_tools as st
    out = {}
    for n in names:
        hit = list((st.NV / "output").glob(f"{tag_glob}/debug/sim_layer_{n}/float/Gemm_output_Y.bin"))
        if hit:
            out[n] = np.fromfile(hit[0], np.float32)[:2]
    return out


def _chip(name, tag, staged, shapes, shards):
    from det import sim_tools as st
    rcs = st.run_sim(name, tag, staged, shards=shards)
    print(f"  sim {tag}: rcs {rcs}", flush=True)
    out = {}
    for s in staged:
        try:
            out[s] = st.read_outputs(name, tag, s, shapes)
        except FileNotFoundError as e:
            print(f"  missing {s}: {e}")
    return out


def main(argv=None):
    from det import sim_tools as st
    from det.decode import decode
    from det.geometry import GRID
    from det.labels.store import LabelStore
    from det.make_test_vectors import write_expected, write_vector
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--name", help="build name (default det_v19_s{stride}_{up}); use a NEW name per checkpoint")
    ap.add_argument("--store", default=str(ROOT / "det_labels/labels_v1.jsonl"))
    ap.add_argument("--eval-set", default=str(ROOT / "det_labels/eval_set.json"))
    a = ap.parse_args(argv)
    import torch
    raw = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    hc = raw["config"]["det_head"]
    stride = hc["stride"]
    name = a.name or f"det_v19_s{stride}_{hc['up']}"
    run = Path(a.ckpt).parent
    gh, gw = GRID[stride]
    shapes = {"output": (2,), "heat": (1, gh, gw), "box": (4, gh, gw)}
    report = {"build": name, "ckpt": str(a.ckpt)}
    compile_build(a.ckpt, name)
    report["bin_md5"] = subprocess.run(["md5sum", str(st.NV / "output" / name / "sdk/nvt_model.bin")],
                                       capture_output=True, text=True).stdout.split()[0]

    # 1. classifier invariance on V19's own letterboxed val inputs (exact bytes)
    v19_imgs = sorted((st.NV / "input/data" / f"{REF}_test/test_images").iterdir())
    ref = _ref_logits(f"{REF}_s*", [p.name for p in v19_imgs])
    imgs = [p for p in v19_imgs if p.name in ref]
    out = _chip(name, "cls", st.stage_images(name, "cls", imgs, exact=True), shapes, a.shards)
    report["classifier"] = compare_classifier({k: v["output"] for k, v in out.items()}, ref)
    print("  classifier:", report["classifier"], flush=True)

    # 2. dark set
    dark = sorted((ROOT / "det_runs/dark_set").iterdir())
    dref = _ref_logits(f"{REF}_dark*", [p.name for p in dark])
    dout = _chip(name, "dark", st.stage_images(name, "dark", dark, exact=True), shapes, a.shards)
    dl = {k: v["output"] for k, v in dout.items()}
    report["dark"] = {**dark_check(dl, {k for k in dl if k.startswith("non_bear__")}),
                      "vs_v19": compare_classifier(dl, dref)}
    print("  dark:", report["dark"], flush=True)

    # 3. detection on the eval-set TEST half, chip vs FP32 (eval_det preds_test.jsonl)
    es = json.load(open(a.eval_set))["test"]
    store = LabelStore(a.store)
    fp32 = {r["path"]: r for r in map(json.loads, open(run / "preds_test.jsonl"))}
    paths = sorted(fp32)
    staged = st.stage_images(name, "det", paths)
    sp = dict(zip(staged, paths))
    cout = _chip(name, "det", staged, shapes, a.shards)
    chip, f32, gt, neg = {}, {}, {}, set()
    by_path = {r["path_hint"].split("/")[-1]: r for r in store.rows.values()
               if r["sha1"] in set(es["bear"] + es["non_bear"])}
    for s, o in cout.items():
        p = sp[s]; m = fp32[p]
        chip[p] = decode(o["output"], o["heat"], o["box"], m["w"], m["h"], stride=stride, thr=EVAL_THR, gate=False)
        f32[p] = np.asarray(m["pred"]).reshape(-1, 5)
        row = by_path.get(Path(p).name)
        if "/non_bear/" in p:
            neg.add(p); gt[p] = np.zeros((0, 4))
        elif row is not None:
            gt[p] = np.array([b["xyxy"] for b in row["boxes"]]).reshape(-1, 4)
    report["detection"] = compare_detections(chip, f32, gt, neg)
    print("  detection:", report["detection"], flush=True)

    # 4. C decoder parity on 20 chip vectors: >= 3 multi-bear, >= 3 non_bear, >= 3 edge-touching
    def pick(pred_fn, n):
        return [s for s in cout if pred_fn(sp[s])][:n]
    multi = pick(lambda p: len(gt.get(p, [])) >= 2, 3)
    nb = pick(lambda p: p in neg, 3)
    edge = pick(lambda p: any(b[0] <= 2 or b[1] <= 2 or b[2] >= fp32[p]["w"] - 2 or b[3] >= fp32[p]["h"] - 2
                              for b in gt.get(p, [])), 3)
    chosen = list(dict.fromkeys(multi + nb + edge + list(cout)))[:20]
    vec = run / "test_vectors"
    if vec.exists():
        shutil.rmtree(vec)
    vec.mkdir()
    cases = []
    for i, s in enumerate(chosen):
        o, m = cout[s], fp32[sp[s]]
        f = f"v{i:02d}.bin"
        write_vector(vec / f, o["output"], o["heat"], o["box"], stride, m["w"], m["h"])
        cases.append((f, o["output"], o["heat"], o["box"], m["w"], m["h"], Path(sp[s]).name))
    write_expected(vec / "expected.txt", cases, stride)
    exe = run / "c_test_main"
    subprocess.check_call(["gcc", "-O2", "-std=c99", "-Wall", "-Werror", "-o", str(exe),
                           str(ROOT / "det/c/test_main.c"), str(ROOT / "det/c/bear_det_decode.c"), "-lm"])
    worst = 0.0
    blocks = (vec / "expected.txt").read_text().split("# ")[1:]
    for blk, c in zip(blocks, cases):
        want = np.array([list(map(float, l.split())) for l in blk.strip().split("\n")[1:]]).reshape(-1, 5)
        got = subprocess.check_output([str(exe), str(vec / c[0])], text=True)
        got = np.array([list(map(float, l.split())) for l in got.strip().split("\n") if l]).reshape(-1, 5)
        worst = max(worst, float(np.abs(got[:, :4] - want[:, :4]).max()) if got.shape == want.shape and len(got) else
                    (0.0 if got.shape == want.shape else float("inf")))
    report["c_parity"] = {"n": len(cases), "max_px": worst, "pass": worst <= 1.0,
                          "multi": len(multi), "non_bear": len(nb), "edge": len(edge)}
    print("  c_parity:", report["c_parity"], flush=True)

    ok = all(report[k]["pass"] for k in ("classifier", "dark", "detection", "c_parity"))
    report["pass"] = ok
    json.dump(report, open(run / "chip_report.json", "w"), indent=1, default=float)
    print("PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
