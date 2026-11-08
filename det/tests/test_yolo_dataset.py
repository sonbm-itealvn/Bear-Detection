import json, os
import numpy as np
from PIL import Image
from det.labels.store import LabelStore, sha1_file

def _setup(tmp_path):
    root = tmp_path / "ds"; st = LabelStore(tmp_path / "s.jsonl")
    rows = {}
    for sp in ("train", "val"):
        for cls in ("bear", "non_bear"):
            d = root / sp / cls; d.mkdir(parents=True)
            for i in range(4):
                p = d / f"{cls}_{i}.jpg"
                Image.fromarray((np.random.RandomState(i + 7 * (cls == "bear") + 31 * (sp == "val")).rand(200, 400, 3) * 255).astype("uint8")).save(p)
                r = {"sha1": sha1_file(p), "dhash": None, "w": 400, "h": 200, "path_hint": str(p), "cls": cls, "domain": "ir",
                     "status": "auto", "boxes": [], "ignore": [], "n_det": 0, "version": 1, "reviewer": None}
                if cls == "bear":
                    r["boxes"] = [{"xyxy": [100.0, 50.0, 300.0, 150.0], "tier": "A"}]
                st.put(r); rows[(sp, cls, i)] = r
    rows[("train", "bear", 1)]["ignore"] = [[110, 60, 290, 140]]          # redundant fragment -> kept
    rows[("train", "bear", 2)]["ignore"] = [[0, 0, 50, 50]]               # residual ignore -> dropped
    rows[("train", "bear", 3)]["status"] = "masked"; rows[("train", "bear", 3)]["boxes"] = []
    st.save()
    es = {"val": {"bear": [rows[("val", "bear", i)]["sha1"] for i in range(4)],
                  "non_bear": [rows[("val", "non_bear", i)]["sha1"] for i in range(2)]},
          "test": {"bear": [], "non_bear": []}}
    (tmp_path / "es.json").write_text(json.dumps(es))
    (tmp_path / "hard.txt").write_text(str(root / "train/non_bear/non_bear_0.jpg") + "\n")
    return root, st

def test_export(tmp_path):
    from det.yolo.export_dataset import export
    root, st = _setup(tmp_path)
    out = tmp_path / "yolo"
    n = export(tmp_path / "s.jsonl", root, out, tmp_path / "es.json", tmp_path / "hard.txt", n_random_nb=2, seed=0)
    assert n["train_bear"] == 3 and n["gray_filled"] == 1 and n["dropped_masked"] == 1
    assert n["train_non_bear"] == 3 and n["val_bear"] == 4 and n["val_non_bear"] == 2
    lab = (out / "labels/train/train__bear__bear_0.txt").read_text().split()
    np.testing.assert_allclose(list(map(float, lab)), [0, 0.5, 0.5, 0.5, 0.5])
    assert (out / "labels/train/train__non_bear__non_bear_0.txt").read_text() == ""
    assert os.path.islink(out / "images/train/train__bear__bear_0.jpg")
    y = (out / "data.yaml").read_text()
    assert "names" in y and "bear" in y


def test_residual_ignore_is_gray_filled_but_boxes_kept(tmp_path):
    from det.yolo.export_dataset import export
    root, st = _setup(tmp_path)
    out = tmp_path / "yolo"
    export(tmp_path / "s.jsonl", root, out, tmp_path / "es.json", tmp_path / "hard.txt", n_random_nb=2, seed=0)
    p = out / "images/train/train__bear__bear_2.jpg"
    assert p.exists() and not os.path.islink(p)                 # materialised copy
    a = np.asarray(Image.open(p)).astype(int)
    assert np.abs(a[5:45, 5:45] - 114).max() <= 3               # ignore region [0,0,50,50] filled gray
    src = np.asarray(Image.open(root / "train/bear/bear_2.jpg")).astype(int)
    assert np.abs(a[60:140, 110:290] - src[60:140, 110:290]).mean() < 3   # the labelled bear is untouched
    assert (out / "labels/train/train__bear__bear_2.txt").read_text().strip()
