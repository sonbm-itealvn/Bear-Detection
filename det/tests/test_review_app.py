import json, zipfile
import numpy as np
from PIL import Image
from det.labels.store import LabelStore
from det.labels.review_app import app_pack, import_app

def _rows(tmp_path):
    st = LabelStore(tmp_path / "s.jsonl"); rows = []
    spec = [("bear", (1600, 900), [[160, 90, 800, 450]], "auto"),       # will be confirmed
            ("bear", (800, 600), [[10, 10, 100, 100]], "auto"),         # will be edited
            ("bear", (500, 900), [], "masked"),                          # will be marked no-bear
            ("non_bear", (640, 360), [], "auto"),                        # will get a drawn bear
            ("non_bear", (640, 360), [], "auto"),                        # confirmed negative
            ("bear", (640, 360), [[1, 1, 50, 50]], "auto")]              # left unseen
    for i, (cls, (w, h), boxes, status) in enumerate(spec):
        p = tmp_path / f"im{i}.jpg"
        Image.fromarray((np.random.RandomState(i).rand(h, w, 3) * 255).astype("uint8")).save(p)
        r = {"sha1": f"s{i}", "dhash": None, "w": w, "h": h, "path_hint": str(p), "cls": cls, "domain": "ir",
             "status": status, "boxes": [{"xyxy": b, "tier": "A"} for b in boxes], "ignore": [[0, 0, 5, 5]],
             "n_det": len(boxes), "version": 1, "reviewer": None}
        st.put(r); rows.append(r)
    st.save()
    return st, rows

def test_pack_contents(tmp_path):
    st, rows = _rows(tmp_path)
    z = tmp_path / "p.zip"
    app_pack(rows, z, title="test pack", max_side=800)
    with zipfile.ZipFile(z) as f:
        names = set(f.namelist())
        html = f.read("review.html").decode()
        im0 = Image.open(f.open("img/0000.jpg"))
    assert {"review.html", "img/0000.jpg", "img/0005.jpg", "HUONG_DAN.txt"} <= names
    assert max(im0.size) <= 800 and "test pack" in html and '"cls": "non_bear"' in html

def test_import_decisions(tmp_path):
    st, rows = _rows(tmp_path)
    exp = [
        {"sha1": "s0", "decision": "ok", "boxes": [[160, 90, 800, 450, "A"]]},
        {"sha1": "s1", "decision": "edited", "boxes": [[20, 20, 300, 280, "H"], [400, 100, 500, 200, "H"]]},
        {"sha1": "s2", "decision": "nobear", "boxes": []},
        {"sha1": "s3", "decision": "edited", "boxes": [[10, 10, 200, 200, "H"]]},
        {"sha1": "s4", "decision": "ok", "boxes": []},
        {"sha1": "s5", "decision": None, "boxes": [[1, 1, 50, 50, "A"]]},
    ]
    (tmp_path / "e.json").write_text(json.dumps(exp))
    stats = import_app(tmp_path / "e.json", st)
    assert stats == {"ok": 2, "edited": 1, "nobear": 1, "nonbear_has_bear": 1, "unseen": 1}
    assert st.get("s0")["status"] == "reviewed" and st.get("s0")["boxes"][0]["tier"] == "A"
    s1 = st.get("s1"); assert s1["status"] == "reviewed" and len(s1["boxes"]) == 2 and s1["boxes"][0]["src"] == "human"
    assert st.get("s2")["status"] == "excluded"
    assert st.get("s3")["status"] == "excluded" and st.get("s3")["has_bear"] is True   # flagged, file not moved
    assert st.get("s4")["status"] == "reviewed"
    assert st.get("s5")["status"] == "auto" and st.get("s5")["reviewer"] is None
    assert LabelStore(tmp_path / "s.jsonl").get("s1")["status"] == "reviewed"
