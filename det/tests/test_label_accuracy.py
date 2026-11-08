from det.labels.label_accuracy import accuracy

def test_accuracy_counts():
    auto = {"a": {"cls": "bear", "domain": "ir", "status": "auto", "boxes": [{"xyxy": [0, 0, 10, 10], "tier": "A"}, {"xyxy": [50, 50, 60, 60], "tier": "B"}]},
            "b": {"cls": "bear", "domain": "ir", "status": "masked", "boxes": []}}
    items = [{"sha1": "a", "decision": "edited", "boxes": [[0, 0, 10, 11, "A"]]},       # B box was wrong, A kept
             {"sha1": "b", "decision": "edited", "boxes": [[5, 5, 20, 20, "H"]]}]       # missed bear drawn
    r = accuracy(items, auto)["ir"]
    assert r["auto_boxes"] == 2 and r["final_boxes"] == 2
    assert abs(r["precision"] - 0.5) < 1e-9 and abs(r["recall"] - 0.5) < 1e-9
    assert r["prec_tierA"] == 1.0 and r["prec_tierB"] == 0.0 and r["masked"] == 1 and not r["pass"]
