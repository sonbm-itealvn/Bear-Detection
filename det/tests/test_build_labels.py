from det.labels.build_labels import build_rows

def test_rows_tiers_ignore_and_truncation():
    prop = {"path": "/x/train/bear/brown_bear_brown_bear_1_a.jpg", "sha1": "s", "w": 100, "h": 80,
            "domain": "inat",
            "yolo": [[0.5, 10, 40, 60, .9, 21], [60, 10, 90, 50, .3, 21], [5, 5, 15, 15, .8, 16]],
            "owl": [[1, 11, 41, 61, .4, 0]]}
    r = build_rows(prop)
    tiers = sorted(b["tier"] for b in r["boxes"])
    assert tiers == ["A"]                              # yolo+owl agree
    assert r["boxes"][0]["truncated"] is True          # touches x=0 within 2px
    assert len(r["ignore"]) >= 1                       # low-conf yolo bear -> tier C -> ignore
    assert r["status"] == "auto" and r["cls"] == "bear"

def test_no_box_is_masked():
    prop = {"path": "/x/train/bear/a.jpg", "sha1": "t", "w": 50, "h": 50, "domain": "other",
            "yolo": [], "owl": []}
    r = build_rows(prop)
    assert r["status"] == "masked" and r["boxes"] == []

def test_non_bear_row_has_no_boxes_and_is_auto():
    prop = {"path": "/x/train/non_bear/fox.jpg", "sha1": "u", "w": 50, "h": 50, "domain": "other",
            "yolo": [[5, 5, 40, 40, .9, 21]], "owl": [[5, 5, 40, 40, .5, 0]]}
    r = build_rows(prop)
    assert r["cls"] == "non_bear" and r["boxes"] == [] and r["status"] == "auto"
