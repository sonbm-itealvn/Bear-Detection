import numpy as np
from det.labels.fusion import iou, wbf, drop_group_boxes, tier_boxes

def test_iou():
    a = np.array([[0, 0, 10, 10]]); b = np.array([[5, 5, 15, 15], [20, 20, 30, 30]])
    np.testing.assert_allclose(iou(a, b), [[25 / 175, 0]], atol=1e-6)

def test_wbf_merges_agreeing_and_keeps_separate():
    y = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float); ys = np.array([.9, .8])
    o = np.array([[1, 1, 11, 11]], float); os_ = np.array([.6])
    b, s, src = wbf([y, o], [ys, os_], iou_thr=0.55)
    assert len(b) == 2
    i = int(np.argmin(b[:, 0]))
    assert src[i] == {0, 1} and 0 <= b[i, 0] <= 1
    assert src[1 - i] == {0}

def test_drop_group_boxes():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 40, 40], [50, 50, 95, 95]], float)
    keep = drop_group_boxes(boxes, np.array([.9, .8, .8]))
    assert list(keep) == [1, 2]

def test_tiers():
    yb = np.array([[0, 0, 10, 10, .9], [100, 100, 120, 120, .6], [200, 0, 210, 10, .3]], float)
    ow = np.array([[1, 0, 10, 11, .5]], float)
    ot = np.zeros((0, 5))
    t = tier_boxes(yb, ow, ot)
    tiers = sorted((d["tier"], round(d["xyxy"][0])) for d in t)
    assert ("A", 0) in tiers and ("B", 100) in tiers and ("C", 200) in tiers


def test_optional_owl_alone_and_owl_plus_other_tiers():
    yb = np.zeros((0, 5)); ow = np.array([[0, 0, 10, 10, .45], [50, 50, 60, 60, .2]], float)
    ot = np.array([[100, 100, 120, 120, .6]], float); ow2 = np.vstack([ow, [[101, 100, 120, 121, .3]]])
    assert sorted(d["tier"] for d in tier_boxes(yb, ow, ot)) == ["C", "C", "C"]            # default unchanged
    t = tier_boxes(yb, ow, ot, owl_b=0.4)
    assert [d["tier"] for d in t if d["xyxy"][0] < 5] == ["B"] and [d["tier"] for d in t if 45 < d["xyxy"][0] < 55] == ["C"]
    t = tier_boxes(yb, ow2, ot, other_owl=True)
    assert [d["tier"] for d in t if d["xyxy"][0] > 90] == ["B"]
