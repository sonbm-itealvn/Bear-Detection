import subprocess, struct
from pathlib import Path
import numpy as np
from det.decode import decode

C = Path(__file__).resolve().parents[1] / "c"

def _write(p, logits, heat, box, stride, w, h):
    gh, gw = heat.shape[-2:]
    with open(p, "wb") as f:
        f.write(struct.pack("<5i", gh, gw, stride, w, h))
        f.write(np.asarray(logits, "<f4").tobytes()); f.write(heat.astype("<f4").tobytes()); f.write(box.astype("<f4").tobytes())

def _run(exe, f):
    out = subprocess.check_output([str(exe), str(f)], text=True).split("\n")
    return np.array([list(map(float, l.split())) for l in out if l.strip()]).reshape(-1, 5)

def test_c_matches_python(tmp_path):
    exe = tmp_path / "t"
    subprocess.check_call(["gcc", "-O2", "-std=c99", "-Wall", "-Werror", "-o", str(exe),
                           str(C / "test_main.c"), str(C / "bear_det_decode.c"), "-lm"])
    rng = np.random.RandomState(0)
    for case in range(30):
        heat = rng.randn(1, 23, 40).astype(np.float32) * 3 - 2
        box = (rng.rand(4, 23, 40).astype(np.float32) * 2.5 - 0.5)
        logits = np.array([0.0, 1.0], np.float32)
        w, h = [(1920, 1080), (500, 800), (640, 360)][case % 3]
        f = tmp_path / f"v{case}.bin"; _write(f, logits, heat, box, 16, w, h)
        cb, pb = _run(exe, f), decode(logits, heat, box, w, h)
        assert cb.shape == pb.shape, (case, cb.shape, pb.shape)
        assert np.abs(cb[:, :4] - pb[:, :4]).max() <= 1.0
    # gate: classifier says non_bear -> no boxes
    f = tmp_path / "neg.bin"; _write(f, np.array([1.0, 0.0], np.float32), heat, box, 16, 640, 360)
    assert len(_run(exe, f)) == 0


def test_c_fallback_in_letterbox_bar(tmp_path):
    exe = tmp_path / "t"
    subprocess.check_call(["gcc", "-O2", "-std=c99", "-o", str(exe), str(C / "test_main.c"), str(C / "bear_det_decode.c"), "-lm"])
    heat = np.full((1, 23, 40), -10.0, np.float32); box = np.zeros((4, 23, 40), np.float32)
    heat[0, 11, 2] = -3.0; box[:, 11, 2] = np.log(0.5)
    heat[0, 11, 20] = -4.0; box[:, 11, 20] = np.log(3.0)
    f = tmp_path / "v.bin"; _write(f, np.array([0.0, 1.0], np.float32), heat, box, 16, 500, 800)
    cb = _run(exe, f) if "_run" in globals() else np.array([list(map(float, l.split())) for l in subprocess.check_output([str(exe), str(f)], text=True).split("\n") if l.strip()]).reshape(-1, 5)
    pb = decode(np.array([0.0, 1.0], np.float32), heat, box, 500, 800)
    assert len(cb) >= 1 and cb.shape == pb.shape and np.abs(cb[:, :4] - pb[:, :4]).max() <= 1.0
