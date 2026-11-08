import subprocess, struct
from pathlib import Path
import numpy as np
from det.yolo.decode import decode

C = Path(__file__).resolve().parents[1] / "c"
SHAPES = [(48, 80), (24, 40), (12, 20)]

def _write(p, outs, w, h):
    with open(p, "wb") as f:
        f.write(struct.pack("<2i", w, h))
        for o in outs:
            f.write(np.asarray(o, "<f4").tobytes())

def _outs(rng, hot=40):
    outs = []
    for (h, wd) in SHAPES:
        box = rng.randn(64, h, wd).astype(np.float32) * 2
        cls = np.full((1, h, wd), -8.0, np.float32)
        idx = rng.choice(h * wd, hot // 3, replace=False)
        cls.reshape(-1)[idx] = rng.randn(len(idx)).astype(np.float32) * 2 + 0.5
        outs += [box, cls]
    return outs

def _run(exe, f):
    out = subprocess.check_output([str(exe), str(f)], text=True)
    return np.array([list(map(float, l.split())) for l in out.split("\n") if l.strip()]).reshape(-1, 5)

def test_c_matches_python(tmp_path):
    exe = tmp_path / "y"
    subprocess.check_call(["gcc", "-O2", "-std=c99", "-Wall", "-Werror", "-o", str(exe),
                           str(C / "yolo_test_main.c"), str(C / "yolo_det_decode.c"), "-lm"])
    rng = np.random.RandomState(0)
    for case in range(25):
        outs = _outs(rng)
        w, h = [(1920, 1080), (500, 800), (640, 360), (1280, 720), (321, 240)][case % 5]
        f = tmp_path / f"v{case}.bin"; _write(f, outs, w, h)
        cb, pb = _run(exe, f), decode(outs, w, h, thr=0.3, K=20, nms=0.5, fallback=True)
        assert cb.shape == pb.shape, (case, cb.shape, pb.shape)
        if len(cb):
            assert np.abs(cb[:, :4] - pb[:, :4]).max() <= 1.0 and np.abs(cb[:, 4] - pb[:, 4]).max() < 1e-4

def test_fallback_returns_strongest_when_nothing_passes():
    rng = np.random.RandomState(1)
    outs = _outs(rng, hot=0)
    outs[1][0, 10, 10] = -2.0                        # best anchor, below thr 0.3
    assert len(decode(outs, 640, 360, thr=0.3, fallback=False)) == 0
    b = decode(outs, 640, 360, thr=0.3, fallback=True)
    assert len(b) == 1 and abs(b[0, 4] - 1 / (1 + np.exp(2.0))) < 1e-6


def test_c_fallback_skips_degenerate(tmp_path):
    exe = tmp_path / "y"
    subprocess.check_call(["gcc", "-O2", "-std=c99", "-Wall", "-Werror", "-o", str(exe),
                           str(C / "yolo_test_main.c"), str(C / "yolo_det_decode.c"), "-lm"])
    outs = []
    for (h, w) in SHAPES:
        outs += [np.zeros((64, h, w), np.float32), np.full((1, h, w), -9.0, np.float32)]
    for (i, j, z, d) in ((20, 2, -1.0, 0), (20, 40, -2.0, 3)):
        outs[1][0, i, j] = z
        for s in range(4):
            outs[0][s * 16:(s + 1) * 16, i, j] = -20.0; outs[0][s * 16 + d, i, j] = 20.0
    f = tmp_path / "v.bin"; _write(f, outs, 500, 800)
    cb, pb = _run(exe, f), decode(outs, 500, 800, thr=0.3, K=20, nms=0.5, fallback=True)
    assert len(cb) == 1 and cb.shape == pb.shape and np.abs(cb[:, :4] - pb[:, :4]).max() <= 1.0
