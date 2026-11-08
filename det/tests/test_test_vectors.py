import subprocess
from pathlib import Path
import numpy as np
from det.make_test_vectors import write_vector, write_expected

C = Path(__file__).resolve().parents[1] / "c"

def test_vectors_roundtrip_through_c(tmp_path):
    exe = tmp_path / "t"
    subprocess.check_call(["gcc", "-O2", "-std=c99", "-o", str(exe), str(C / "test_main.c"), str(C / "bear_det_decode.c"), "-lm"])
    rng = np.random.RandomState(1)
    cases = []
    for i, (w, h) in enumerate([(1920, 1080), (720, 1280)]):
        heat = rng.randn(1, 23, 40).astype(np.float32) * 3 - 2
        box = rng.rand(4, 23, 40).astype(np.float32) * 2 - 0.3
        logits = np.array([0.0, 2.0], np.float32)
        f = tmp_path / f"v{i:02d}.bin"
        write_vector(f, logits, heat, box, 16, w, h)
        cases.append((f.name, logits, heat, box, w, h))
    exp = tmp_path / "expected.txt"
    write_expected(exp, cases, stride=16)
    blocks = exp.read_text().split("# ")[1:]
    assert len(blocks) == 2
    for blk, (name, *_rest) in zip(blocks, cases):
        lines = blk.strip().split("\n")
        assert lines[0].startswith(name)
        want = np.array([list(map(float, l.split())) for l in lines[1:]]).reshape(-1, 5)
        got = subprocess.check_output([str(exe), str(tmp_path / name)], text=True)
        got = np.array([list(map(float, l.split())) for l in got.strip().split("\n") if l]).reshape(-1, 5)
        assert got.shape == want.shape and np.abs(got[:, :4] - want[:, :4]).max() <= 1.0
