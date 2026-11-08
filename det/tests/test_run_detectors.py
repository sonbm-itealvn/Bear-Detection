import numpy as np
from det.labels.run_detectors import tiles, nms_rows, unflip

def test_tiles_cover_image():
    for w, h in [(1920, 1080), (640, 360), (1000, 2000)]:
        ts = tiles(w, h, 640, 0.2)
        assert ts[0][0] == 0 and ts[0][1] == 0
        assert max(t[2] for t in ts) == w and max(t[3] for t in ts) == h
        for x0, y0, x1, y1 in ts:
            assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h

def test_nms_per_class():
    rows = [[0, 0, 10, 10, .9, 21], [1, 1, 11, 11, .8, 21], [1, 1, 11, 11, .7, 16]]
    out = nms_rows(rows, 0.6)
    assert len(out) == 2 and {r[5] for r in out} == {21, 16}

def test_unflip():
    np.testing.assert_allclose(unflip([10, 5, 30, 20], 100), [70, 5, 90, 20])
