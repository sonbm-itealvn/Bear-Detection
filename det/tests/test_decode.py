import numpy as np
from det.decode import decode

H, W = 23, 40

def _maps(peaks):
    heat = np.full((1, H, W), -10.0, np.float32); box = np.zeros((4, H, W), np.float32)
    for (i, j, z, d) in peaks:
        heat[0, i, j] = z; box[:, i, j] = np.log(d)       # d in stride units for l,t,r,b
    return heat, box

def test_gate_off_returns_nothing_when_not_bear():
    heat, box = _maps([(10, 20, 3.0, 2.0)])
    assert len(decode(np.array([1.0, 0.0]), heat, box, 1920, 1080, gate=True)) == 0

def test_fallback_one_box():
    heat, box = _maps([(10, 20, -3.0, 2.0)])              # below threshold
    b = decode(np.array([0.0, 1.0]), heat, box, 1920, 1080)
    assert len(b) == 1
    np.testing.assert_allclose(b[0, :4], np.array([20.5 * 16 - 32, 10.5 * 16 - 32, 20.5 * 16 + 32, 10.5 * 16 + 32]) * 3, atol=1e-3)

def test_boxes_clipped():
    heat, box = _maps([(0, 0, 3.0, 5.0)])                 # huge box at the corner
    b = decode(np.array([0.0, 1.0]), heat, box, 1920, 1080)
    assert (b[:, :4] >= 0).all() and (b[:, 2] <= 1920).all() and (b[:, 3] <= 1080).all()

def test_letterbox_mapping_portrait():
    heat, box = _maps([(11, 20, 3.0, 2.0)])               # centre (328, 184) in letterbox
    b = decode(np.array([0.0, 1.0]), heat, box, 500, 800)
    s = 360 / 800; px = (640 - int(500 * s)) // 2
    np.testing.assert_allclose(b[0, :4], [(296 - px) / s, (152) / s, (360 - px) / s, (216) / s], atol=0.05)

def test_topk_and_nms():
    peaks = [(i, j, 2.0, 0.6) for i in range(1, 22, 2) for j in range(1, 39, 2)]   # 209 isolated peaks
    heat, box = _maps(peaks)
    assert len(decode(np.array([0.0, 1.0]), heat, box, 640, 360, K=20)) == 20
    heat, box = _maps([(10, 10, 3.0, 2.0), (10, 11, 2.9, 2.0)])                   # same bear, adjacent cells
    assert len(decode(np.array([0.0, 1.0]), heat, box, 640, 360)) == 1
    heat, box = _maps([(10, 5, 3.0, 2.0), (10, 30, 3.0, 2.0)])                    # two separate bears
    assert len(decode(np.array([0.0, 1.0]), heat, box, 640, 360)) == 2


def test_fallback_survives_peak_in_letterbox_bar():
    """Review #2: 500x800 portrait -> 207-px side bars; the strongest peak sits in the bar and
    clips to nothing. The classifier said bear, so a valid box must still come back."""
    heat, box = _maps([(11, 2, 3.0, 0.5), (11, 20, 1.0, 3.0)])     # bar peak (cx=40) is strongest
    b = decode(np.array([0.0, 1.0]), heat, box, 500, 800)
    assert len(b) >= 1 and (b[:, 2] - b[:, 0] >= 1).all()
    heat, box = _maps([(11, 2, -3.0, 0.5), (11, 20, -4.0, 3.0)])   # both below threshold
    b = decode(np.array([0.0, 1.0]), heat, box, 500, 800)
    assert len(b) == 1 and b[0, 0] > 0
