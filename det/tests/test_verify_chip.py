import numpy as np
from det.verify_chip import compare_classifier, compare_detections, dark_check

def test_compare_classifier():
    a = {"x": np.array([0.5, -1.0]), "y": np.array([1.0, 2.0])}
    r = compare_classifier(a, {"x": np.array([0.5, -1.0]), "y": np.array([1.0, 2.0])})
    assert r["max_abs"] == 0 and r["argmax_changes"] == 0 and r["n"] == 2 and r["pass"]
    r = compare_classifier(a, {"x": np.array([0.5, -1.0]), "y": np.array([3.0, 2.0])})
    assert r["max_abs"] == 2.0 and r["argmax_changes"] == 1 and not r["pass"]

def test_compare_detections():
    gt = {"a": np.array([[0, 0, 10, 10]]), "b": np.array([[20, 20, 40, 40]])}
    fp = {"a": np.array([[0, 0, 10, 10, .9]]), "b": np.array([[20, 20, 40, 40, .8]]), "n": np.zeros((0, 5))}
    chip = {"a": np.array([[0, 0, 10, 10.2, .88]]), "b": np.array([[21, 20, 40, 40, .79]]), "n": np.zeros((0, 5))}
    r = compare_detections(chip, fp, gt, neg={"n"})
    assert r["median_iou_chip_vs_fp32"] > 0.94 and abs(r["ap50_drop"]) < 1e-9 and r["pass"]
    chip["b"] = np.zeros((0, 5))
    r = compare_detections(chip, fp, gt, neg={"n"})
    assert r["ap50_drop"] > 0.4 and not r["pass"]

def test_dark_check():
    logits = {"n1": np.array([1.0, -2.0]), "n2": np.array([0.5, 0.1]), "b1": np.array([-1.0, 1.0])}
    r = dark_check(logits, negatives={"n1", "n2"})
    assert r["fp"] == 0 and r["n_neg"] == 2 and abs(r["max_neg_margin"] - (-0.4)) < 1e-6 and r["pass"]


def test_iou_stat_only_uses_shown_boxes():
    """Low-score FP32 boxes (< the 0.3 display threshold) are never shown; they must not drive
    the chip-vs-FP32 IoU statistic (AP still uses the full curve)."""
    gt = {"a": np.array([[0, 0, 10, 10]])}
    fp = {"a": np.array([[0, 0, 10, 10, .9], [50, 50, 60, 60, .1]])}
    chip = {"a": np.array([[0, 0, 10, 10, .88], [70, 70, 80, 80, .12]])}
    r = compare_detections(chip, fp, gt)
    assert r["median_iou_chip_vs_fp32"] == 1.0 and r["p10_iou_chip_vs_fp32"] == 1.0
