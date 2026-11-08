import numpy as np
from det.metrics import ap_at, ap_range, recall_at_fppi, iou_stats

GT = [np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10]])]

def test_ap_perfect_and_half():
    perfect = [np.array([[0, 0, 10, 10, .9]]), np.array([[0, 0, 10, 10, .8]])]
    assert abs(ap_at(perfect, GT, 0.5) - 1.0) < 1e-6
    half = [np.array([[0, 0, 10, 10, .9]]), np.array([[50, 50, 60, 60, .8]])]
    assert abs(ap_at(half, GT, 0.5) - 0.5) < 1e-6

def test_duplicate_detection_is_a_false_positive():
    dup = [np.array([[0, 0, 10, 10, .9], [0, 0, 10, 10, .85]]), np.array([[0, 0, 10, 10, .8]])]
    assert ap_at(dup, GT, 0.5) < 1.0

def test_ap_range_penalises_loose_boxes():
    loose = [np.array([[0, 0, 12, 12, .9]]), np.array([[0, 0, 12, 12, .8]])]
    assert ap_at(loose, GT, 0.5) == 1.0 and ap_range(loose, GT) < 0.8

def test_recall_at_fppi():
    pos = [np.array([[0, 0, 10, 10, .9]]), np.array([[0, 0, 10, 10, .3]])]
    neg = [np.array([[0, 0, 5, 5, .5]]), np.zeros((0, 5))]
    assert recall_at_fppi(pos, GT, neg, 0.0) == 0.5          # threshold above the negative's .5
    assert recall_at_fppi(pos, GT, neg, 1.0) == 1.0

def test_iou_stats():
    s = iou_stats([0.5, 0.9, 0.95])
    assert s["median"] == 0.9


def test_size_buckets_are_coco_style():
    """Review #6: a correct box on a medium bear must not be a false positive in 'large'."""
    from det.eval_det import score
    gts = [np.array([[0, 0, 20, 20], [100, 100, 160, 160], [200, 0, 400, 200]], float)]   # small, medium, large
    preds = [np.array([[0, 0, 20, 20, .9], [100, 100, 160, 160, .8], [200, 0, 400, 200, .7]], float)]
    r = score(preds, gts, ["ir"], [], scales=[1.0])
    assert r["ap50_size"] == {"small": 1.0, "medium": 1.0, "large": 1.0}
    r = score(preds, gts, ["ir"], [], scales=[0.25])                # same image seen 4x smaller by the network
    assert set(r["ap50_size"]) == {"small", "medium"}
