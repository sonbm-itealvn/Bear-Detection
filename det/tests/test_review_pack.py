import csv, io, zipfile
import numpy as np
from PIL import Image
from det.labels.store import LabelStore
from det.labels.review_pack import contact_sheets, import_review

def _rows(tmp_path, n=3):
    rows = []
    for i in range(n):
        p = tmp_path / f"im{i}.jpg"
        Image.fromarray((np.random.RandomState(i).rand(300, 500, 3) * 255).astype("uint8")).save(p)
        rows.append({"sha1": f"s{i}", "dhash": None, "w": 500, "h": 300, "path_hint": str(p),
                     "cls": "bear", "domain": "ir", "status": "auto",
                     "boxes": [{"xyxy": [10, 10, 100, 100], "tier": "A"}, {"xyxy": [200, 50, 300, 200], "tier": "B"}],
                     "ignore": [], "n_det": 2, "version": 1, "reviewer": None})
    return rows

def test_contact_sheets_zip(tmp_path):
    z = tmp_path / "r.zip"
    contact_sheets(_rows(tmp_path, 14), z, per_sheet=12)
    with zipfile.ZipFile(z) as f:
        names = f.namelist()
        assert {"sheet_0000.jpg", "sheet_0001.jpg", "review.csv", "README.txt"} <= set(names)
        rows = list(csv.DictReader(io.StringIO(f.read("review.csv").decode())))
    assert len(rows) == 14 and rows[0]["decision"] == "ok" and rows[0]["box_ids"] == "0 1"

def test_import_review_decisions(tmp_path):
    st = LabelStore(tmp_path / "s.jsonl")
    for r in _rows(tmp_path, 4):
        st.put(r)
    c = tmp_path / "review.csv"
    with open(c, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sheet", "idx", "sha1", "path", "box_ids", "decision", "missed", "notes"])
        for i, d in enumerate(["ok", "bad:1", "nobear", "missed"]):
            w.writerow([0, i, f"s{i}", "", "0 1", d, "", ""])
    stats = import_review(c, st)
    assert stats["ok"] == 1 and stats["bad"] == 1
    assert st.get("s0")["status"] == "reviewed"
    assert st.get("s1")["status"] == "corrected" and len(st.get("s1")["boxes"]) == 1 and st.get("s1")["ignore"] == [[200, 50, 300, 200]]
    assert st.get("s2")["status"] == "excluded"
    assert st.get("s3")["status"] == "masked"
    assert LabelStore(tmp_path / "s.jsonl").get("s1")["status"] == "corrected"     # saved
