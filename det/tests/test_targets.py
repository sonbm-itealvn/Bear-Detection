import numpy as np
from det.targets import encode, cell_centers

def test_peak_and_box_targets():
    b = np.array([[100., 50., 260., 210.]])            # 160x160 bear, centre (180,130)
    t = encode(b, np.zeros((0, 4)), 16, masked=False)
    i, j = np.unravel_index(t["heat"][0].argmax(), t["heat"][0].shape)
    cx, cy = cell_centers(16)
    assert abs(cx[j] - 180) <= 8 and abs(cy[i] - 130) <= 8 and t["heat"][0, i, j] == 1.0
    l, tp, r, bt = np.exp(t["box"][:, i, j]) * 16
    np.testing.assert_allclose([cx[j] - l, cy[i] - tp, cx[j] + r, cy[i] + bt], b[0], atol=1e-3)
    assert abs(t["box_w"].sum() - 1.0) < 1e-5 and t["n_pos"] == 1

def test_ignore_and_masked():
    t = encode(np.zeros((0, 4)), np.array([[0., 0., 160., 160.]]), 16, masked=False)
    assert t["heat_w"][0, :5, :5].sum() == 0 and t["heat_w"][0, 20, 39] == 1
    t2 = encode(np.array([[10., 10., 50., 50.]]), np.zeros((0, 4)), 16, masked=True)
    assert t2["heat_w"].sum() == 0 and t2["box_w"].sum() == 0

def test_two_bears_two_peaks():
    b = np.array([[20., 40., 120., 200.], [400., 60., 600., 300.]])
    t = encode(b, np.zeros((0, 4)), 16, masked=False)
    assert (t["heat"][0] == 1.0).sum() == 2 and t["n_pos"] == 2
