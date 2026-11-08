"""Recompute every chip-side number the delivery READMEs quote, from the simulator dumps that
verify_chip.py (a) and det/yolo/chip.py (b') left behind -- no re-simulation:
  * chip vs FP32 detection on the eval-set TEST half, for both candidates (eval threshold EVAL_THR);
  * the deployment-style gated comparison (boxes only where the V19 classifier on the chip says bear);
  * regenerated expected.txt for both test-vector sets with the current decoders, checked against C.
python -m det.compare_chip"""
import json, struct, subprocess
from pathlib import Path

import numpy as np

ROOT = Path("/data/nas07_new/PersonalData/phuocthien")
A_NAME, A_RUN = "det_v19_s16_up_slice", ROOT / "det_runs/s16_v2"
B_NAME, B_RUN = "yolo_v8n_v2_chip", ROOT / "det_runs/yolo_v8n_v2/train"
SA = {"output": (2,), "heat": (1, 23, 40), "box": (4, 23, 40)}


def _rows():
    from det.labels.store import LabelStore
    st = LabelStore(ROOT / "det_labels/labels_v1.jsonl")
    es = json.load(open(ROOT / "det_labels/eval_set.json"))["test"]
    return {r["sha1"]: r for r in (st.get(s) for s in es["bear"] + es["non_bear"]) if r}


def _gt(path, by_sha, sha_of):
    r = by_sha.get(sha_of.get(path))
    if "/non_bear/" in path:
        return np.zeros((0, 4)), r
    if r is None or r["status"] in ("masked", "excluded") or not r["boxes"]:
        return None, r
    return np.array([b["xyxy"] for b in r["boxes"]], float).reshape(-1, 4), r


def main():
    import argparse
    global A_NAME, A_RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-name", default=A_NAME); ap.add_argument("--a-run", default=str(A_RUN))
    ap.add_argument("--gated-out", default=str(ROOT / "det_runs/compare_gated_chip_test.json"))
    args = ap.parse_args()
    A_NAME, A_RUN = args.a_name, Path(args.a_run)
    from det import sim_tools as st
    from det.dataset import _sha1_cached
    from det.decode import decode as decode_a
    from det.eval_det import EVAL_THR, score
    from det.geometry import letterbox_params
    from det.verify_chip import compare_detections
    from det.yolo import chip as yc
    from det.yolo.decode import decode as decode_b
    from det.yolo.geometry import letterbox_params as lb384
    by_sha = _rows()
    fa = {r["path"]: r for r in map(json.loads, open(A_RUN / "preds_test.jsonl"))}
    fb = {r["path"]: r for r in map(json.loads, open(B_RUN / "preds_test.jsonl"))}
    sha_of = _sha1_cached(sorted(set(fa) | set(fb)), ROOT / "det_labels/sha1_cache.json")
    pa_sorted, pb_sorted = sorted(fa), sorted(fb)
    chip_a, chip_b, f32_a, f32_b, gts, neg = {}, {}, {}, {}, {}, set()
    gated = {"a": [[], [], [], [], []], "b": [[], [], [], [], []]}
    tp = fp = npos = nneg = 0
    for p in pa_sorted:
        g, r = _gt(p, by_sha, sha_of)
        if r is not None and r.get("status") == "excluded":
            continue                                  # reviewer: no bear / bear in a negative
        try:
            oa = st.read_outputs(A_NAME, "det", f"{pa_sorted.index(p):05d}__{Path(p).stem}.jpg", SA)
            ob = yc.read(B_NAME, "det", f"{pb_sorted.index(p):05d}__{Path(p).stem}.jpg") if p in fb else None
        except FileNotFoundError:
            continue
        w, h = fa[p]["w"], fa[p]["h"]
        chip_a[p] = decode_a(oa["output"], oa["heat"], oa["box"], w, h, thr=EVAL_THR, gate=False)
        f32_a[p] = np.asarray(fa[p]["pred"]).reshape(-1, 5)
        if ob is not None:
            chip_b[p] = decode_b(ob, w, h, thr=EVAL_THR)
            f32_b[p] = np.asarray(fb[p]["pred"]).reshape(-1, 5)
        is_bear = "/bear/" in p
        said = bool(oa["output"][1] > oa["output"][0])
        npos += is_bear; nneg += not is_bear; tp += is_bear and said; fp += (not is_bear) and said
        if g is None:
            continue
        gts[p] = g
        if not is_bear:
            neg.add(p)
        ga = decode_a(oa["output"], oa["heat"], oa["box"], w, h, thr=EVAL_THR, gate=True)
        gb = decode_b(ob, w, h, thr=EVAL_THR, fallback=True) if (said and ob is not None) else np.zeros((0, 5))
        for k, pr, f in (("a", ga, letterbox_params(w, h)[0] / w), ("b", gb, lb384(w, h)[0] / w)):
            if is_bear:
                gated[k][0].append(pr); gated[k][1].append(g); gated[k][2].append(r["domain"]); gated[k][4].append(f)
            else:
                gated[k][3].append(pr)
    det_a = compare_detections({p: chip_a[p] for p in gts}, {p: f32_a[p] for p in gts}, gts, neg)
    det_b = compare_detections({p: chip_b[p] for p in gts if p in chip_b}, {p: f32_b[p] for p in gts if p in f32_b}, gts, neg)
    print("(a) chip vs FP32:", {k: round(v, 4) if isinstance(v, float) else v for k, v in det_a.items()})
    print("(b') chip vs FP32:", {k: round(v, 4) if isinstance(v, float) else v for k, v in det_b.items()})
    ra = json.load(open(A_RUN / "chip_report.json")); ra["detection"] = det_a
    ra["pass"] = all(ra[k]["pass"] for k in ("classifier", "dark", "detection", "c_parity"))
    json.dump(ra, open(A_RUN / "chip_report.json", "w"), indent=1, default=float)
    rb = json.load(open(B_RUN / "chip_report.json")); rb.update(det_b)
    json.dump(rb, open(B_RUN / "chip_report.json", "w"), indent=1, default=float)
    out = {"v19_chip_image_level": {"recall": tp / npos, "precision": tp / max(tp + fp, 1), "fp": fp, "n_pos": npos, "n_neg": nneg}}
    for k in ("a", "b"):
        P, G, D, N, S = gated[k]
        out[k] = score(P, G, D, N, S)
        s = out[k]
        print(f"gated {k}: AP50 {s['ap50']:.3f} AP {s['ap']:.3f} IoU {s['iou']['median']:.3f} rec@.1 {s['rec_fppi01']:.3f} size {s['ap50_size']}")
    print("V19 on chip:", out["v19_chip_image_level"])
    json.dump(out, open(args.gated_out, "w"), indent=1, default=float)
    # regenerate expected.txt for both test-vector sets with the current decoders and check C
    for name, vec, main_c, src_c, dec in (
            ("a", A_RUN / "test_vectors", "test_main.c", "bear_det_decode.c", "a"),
            ("b", B_RUN / "test_vectors", "yolo_test_main.c", "yolo_det_decode.c", "b")):
        if not vec.exists():
            continue
        exe = ROOT / f"det_runs/_c_{name}"
        subprocess.check_call(["gcc", "-O2", "-std=c99", "-Wall", "-Werror", "-o", str(exe),
                               str(ROOT / "det/c" / main_c), str(ROOT / "det/c" / src_c), "-lm"])
        blocks, worst, nbox = [], 0.0, 0
        old = {l.split()[1]: l for l in (vec / "expected.txt").read_text().splitlines() if l.startswith("# ")}
        for f in sorted(vec.glob("v*.bin")):
            raw = open(f, "rb").read()
            if dec == "a":
                gh, gw, stride, w, h = struct.unpack("<5i", raw[:20])
                a = np.frombuffer(raw[20:], "<f4")
                lg, heat, box = a[:2], a[2:2 + gh * gw].reshape(1, gh, gw), a[2 + gh * gw:].reshape(4, gh, gw)
                pb = decode_a(lg, heat, box, w, h, stride=stride)
            else:
                w, h = struct.unpack("<2i", raw[:8])
                a = np.frombuffer(raw[8:], "<f4"); outs, o = [], 0
                for n in (64 * 48 * 80, 48 * 80, 64 * 24 * 40, 24 * 40, 64 * 12 * 20, 12 * 20):
                    outs.append(a[o:o + n]); o += n
                shp = [(64, 48, 80), (1, 48, 80), (64, 24, 40), (1, 24, 40), (64, 12, 20), (1, 12, 20)]
                pb = decode_b([x.reshape(s) for x, s in zip(outs, shp)], w, h, thr=0.3, K=20, nms=0.5, fallback=True)
            cb = np.array([list(map(float, l.split())) for l in subprocess.check_output([str(exe), str(f)], text=True).splitlines() if l.strip()]).reshape(-1, 5)
            assert cb.shape == pb.shape, (f, cb.shape, pb.shape)
            if len(cb):
                worst = max(worst, float(np.abs(cb[:, :4] - pb[:, :4]).max()))
            nbox += len(cb)
            head = old.get(f.name, f"# {f.name} {w}x{h}")
            fmt = "%.3f %.3f %.3f %.3f %.5f\n" if dec == "a" else "%.3f %.3f %.3f %.3f %.6f\n"
            blocks.append(head + "\n" + "".join(fmt % tuple(b) for b in pb))
        (vec / "expected.txt").write_text("".join(blocks))
        exe.unlink()
        print(f"test vectors {name}: {len(blocks)} vectors, {nbox} boxes, C vs Python max {worst:.4f} px")
        if dec == "a":
            ra = json.load(open(A_RUN / "chip_report.json")); ra["c_parity"].update(max_px=worst, pass_=worst <= 1.0)
            ra["c_parity"]["pass"] = worst <= 1.0; ra["c_parity"].pop("pass_", None)
            json.dump(ra, open(A_RUN / "chip_report.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
