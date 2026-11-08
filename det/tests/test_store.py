import shutil
import numpy as np
from PIL import Image
from det.labels.store import LabelStore, load_rgb, sha1_file

def _img(p, mode="RGB", fmt="JPEG"):
    a = (np.random.RandomState(0).rand(40, 60, 3) * 255).astype("uint8")
    im = Image.fromarray(a).convert(mode)
    if fmt == "GIF":
        im = im.convert("P")
    im.save(p, fmt)

def test_load_rgb_handles_gif_png_gray(tmp_path):
    for mode, fmt in [("RGB", "GIF"), ("RGBA", "PNG"), ("L", "JPEG")]:
        p = tmp_path / f"x_{fmt}.jpg"; _img(p, mode, fmt)
        im = load_rgb(p)
        assert im.mode == "RGB" and im.size == (60, 40)

def test_join_after_move_and_class_change(tmp_path):
    a = tmp_path / "train/bear"; b = tmp_path / "val/bear"; c = tmp_path / "val/non_bear"
    for d in (a, b, c): d.mkdir(parents=True)
    p = a / "img1.jpg"; _img(p)
    st = LabelStore(tmp_path / "labels.jsonl")
    st.put({"sha1": sha1_file(p), "dhash": "0" * 16, "w": 60, "h": 40, "path_hint": str(p),
            "cls": "bear", "domain": "other", "status": "auto",
            "boxes": [{"xyxy": [1, 2, 30, 20], "score": .9, "src": "wbf", "tier": "A",
                       "truncated": False, "crowd": False}],
            "ignore": [], "n_det": 1, "version": 1, "reviewer": None, "ts": 0})
    st.save()
    shutil.move(str(p), b / "img1.jpg")                      # resplit move
    st2 = LabelStore(tmp_path / "labels.jsonl")
    j = st2.join([b / "img1.jpg"])
    assert j[str(b / "img1.jpg")]["boxes"][0]["xyxy"] == [1, 2, 30, 20]
    shutil.move(str(b / "img1.jpg"), c / "img1.jpg")         # relabelled to non_bear
    j = st2.join([c / "img1.jpg"])
    assert j[str(c / "img1.jpg")] is None and str(c / "img1.jpg") in st2.stale
