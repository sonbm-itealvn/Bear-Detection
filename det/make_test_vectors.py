"""Deliverable test vectors for the C decoder, built from REAL simulator outputs.
Each vector: int32 gh, gw, stride, src_w, src_h; float32 logits[2], heat[gh*gw], box[4*gh*gw].
expected.txt: per vector '# <file> <src_w>x<src_h> <source image>' then one line per box
(x1 y1 x2 y2 score), from det/decode.py with the C defaults (thr 0.3, K 20, NMS 0.5, gate on).
python -m det.make_test_vectors --name det_v19_s16 --tag test --images list.txt --out DIR"""
import argparse, struct
from pathlib import Path
import numpy as np
from det.decode import decode


def write_vector(path, logits, heat, box, stride, w, h):
    gh, gw = heat.shape[-2:]
    with open(path, "wb") as f:
        f.write(struct.pack("<5i", gh, gw, stride, w, h))
        f.write(np.asarray(logits, "<f4").reshape(-1)[:2].tobytes())
        f.write(np.asarray(heat, "<f4").reshape(-1).tobytes())
        f.write(np.asarray(box, "<f4").reshape(-1).tobytes())


def write_expected(path, cases, stride):
    with open(path, "w") as f:
        for case in cases:
            name, logits, heat, box, w, h = case[:6]
            src = case[6] if len(case) > 6 else ""
            f.write(f"# {name} {w}x{h} {src}\n")
            for b in decode(logits, heat, box, w, h, stride=stride):
                f.write("%.3f %.3f %.3f %.3f %.5f\n" % tuple(b))


def main():
    from PIL import Image
    from det import sim_tools as st
    from det.geometry import GRID
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--images", required=True, help="lines: <staged name>\\t<original image path>")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gh, gw = GRID[a.stride]
    shapes = {"output": (2,), "heat": (1, gh, gw), "box": (4, gh, gw)}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cases = []
    for i, line in enumerate(l for l in open(a.images) if l.strip()):
        staged, src = line.rstrip("\n").split("\t")
        o = st.read_outputs(a.name, a.tag, staged, shapes)
        with Image.open(src) as im:
            w, h = im.size
        f = f"v{i:02d}.bin"
        write_vector(out / f, o["output"], o["heat"], o["box"], a.stride, w, h)
        cases.append((f, o["output"], o["heat"], o["box"], w, h, Path(src).name))
    write_expected(out / "expected.txt", cases, a.stride)
    print(f"wrote {len(cases)} vectors -> {out}")


if __name__ == "__main__":
    main()
