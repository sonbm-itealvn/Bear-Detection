import numpy as np
from det.geometry import letterbox_params, orig_to_lb, lb_to_orig, GRID

def test_params_16x9_no_pad():
    assert letterbox_params(1920, 1080) == (640, 360, 0, 0)

def test_params_portrait():
    nw, nh, px, py = letterbox_params(500, 800)
    assert nh == 360 and nw == int(500 * 360 / 800) and py == 0 and px == (640 - nw) // 2

def test_roundtrip_odd_aspects():
    for w, h in [(500, 800), (321, 240), (1280, 720), (768, 527), (640, 360)]:
        b = np.array([[10.5, 20.25, w - 30.0, h - 1.0]], dtype=np.float64)
        back = lb_to_orig(orig_to_lb(b, w, h), w, h, clip=False)
        assert np.abs(back - b).max() < 1e-4, (w, h, back, b)

def test_lb_to_orig_clips():
    w, h = 500, 800                      # pillarboxed: px > 0
    nw, nh, px, py = letterbox_params(w, h)
    lb = np.array([[0.0, -5.0, 640.0, 400.0]])   # spills into padding and below
    o = lb_to_orig(lb, w, h, clip=True)
    assert (o[:, 0] >= 0).all() and (o[:, 2] <= w).all() and (o[:, 1] >= 0).all() and (o[:, 3] <= h).all()
    assert (o[:, 2] > o[:, 0]).all() and (o[:, 3] > o[:, 1]).all()

def test_grid():
    assert GRID[16] == (23, 40) and GRID[8] == (45, 80)
