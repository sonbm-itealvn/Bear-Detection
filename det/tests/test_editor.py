import json, zipfile
import numpy as np
from PIL import Image
from det.labels.store import LabelStore
from det.labels.review_pack import editor_pack, import_editor

def test_editor_roundtrip(tmp_path):
    st = LabelStore(tmp_path / "s.jsonl")
    rows = []
    for i, (w, h) in enumerate([(2560, 1440), (800, 600), (500, 900)]):
        p = tmp_path / f"im{i}.jpg"
        Image.fromarray((np.random.RandomState(i).rand(h, w, 3) * 255).astype("uint8")).save(p)
        r = {"sha1": f"s{i}", "dhash": None, "w": w, "h": h, "path_hint": str(p), "cls": "bear", "domain": "ir",
             "status": "masked", "boxes": [{"xyxy": [w * .1, h * .2, w * .6, h * .7], "tier": "B"}],
             "ignore": [], "n_det": 1, "version": 1, "reviewer": None}
        st.put(r); rows.append(r)
    z = tmp_path / "ed.zip"
    editor_pack(rows, z)
    with zipfile.ZipFile(z) as f:
        names = set(f.namelist())
        assert {"editor.html", "boxes.json", "img/0000.jpg", "img/0002.jpg"} <= names
        items = json.loads(f.read("boxes.json"))
        im0 = Image.open(f.open("img/0000.jpg"))
    assert max(im0.size) <= 1280 and abs(items[0]["scale"] - 0.5) < 1e-6
    # what the editor's Export writes: boxes divided back by scale (original pixels)
    out = [{"sha1": it["sha1"], "boxes": [[v / it["scale"] for v in b] for b in it["boxes"]]} for it in items]
    out[1]["boxes"].append([10, 10, 100, 100])                       # the user added a missed bear
    (tmp_path / "boxes_reviewed.json").write_text(json.dumps(out))
    n = import_editor(tmp_path / "boxes_reviewed.json", st)
    assert n == 3
    for r in rows:
        got = st.get(r["sha1"])
        assert got["status"] == "reviewed" and got["reviewer"] == "user"
        assert abs(np.array(got["boxes"][0]["xyxy"]) - np.array(r["boxes"][0]["xyxy"])).max() <= 1.0
        assert got["boxes"][0]["tier"] == "H" and got["boxes"][0]["src"] == "human"
    assert len(st.get("s1")["boxes"]) == 2
