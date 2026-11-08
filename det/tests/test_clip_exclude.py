import numpy as np
from PIL import Image
from det.labels.clip_exclude import clip_suspect_nonbear

def test_suspects_by_basename_in_current_split(tmp_path):
    root = tmp_path / "ds"
    for sp in ("train", "val"):
        (root / sp / "non_bear").mkdir(parents=True); (root / sp / "bear").mkdir(parents=True)
    for n in ("a.jpg", "b.jpg", "c.jpg"):
        Image.new("RGB", (4, 4)).save(root / "train/non_bear" / n)
    Image.new("RGB", (4, 4)).save(root / "val/non_bear/d.jpg")
    # probe files carry OLD paths (before the 09/10 resplit); join is by basename
    files = np.array(["/old/val/non_bear/a.jpg", "/old/train/non_bear/b.jpg", "/old/train/non_bear/c.jpg",
                      "/old/train/non_bear/d.jpg", "/old/train/bear/e.jpg"])
    y = np.array([0, 0, 0, 0, 1]); oof = np.array([0.7, 0.3, 0.95, 0.8, 0.9])
    np.savez(tmp_path / "p.npz", files=files, y=y, oof=oof)
    got = clip_suspect_nonbear(tmp_path / "p.npz", root, "train", thr=0.5)
    assert sorted(p.name for p in got) == ["a.jpg", "c.jpg"]
