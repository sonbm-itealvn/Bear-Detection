import json
import numpy as np
import torch
from PIL import Image
from det.labels.store import LabelStore, sha1_file

def _make(tmp_path):
    root = tmp_path / "ds"
    st = LabelStore(tmp_path / "labels.jsonl")
    for cls, n in (("bear", 4), ("non_bear", 4)):
        d = root / "train" / cls; d.mkdir(parents=True)
        for i in range(n):
            p = d / f"{cls}_{i}.jpg"
            Image.fromarray((np.random.RandomState(i + (cls == "bear") * 10).rand(400, 600, 3) * 255).astype("uint8")).save(p)
            row = {"sha1": sha1_file(p), "dhash": None, "w": 600, "h": 400, "path_hint": str(p), "cls": cls,
                   "domain": "ir", "status": "auto", "boxes": [], "ignore": [], "n_det": 0, "version": 1, "reviewer": None}
            if cls == "bear":
                row["boxes"] = [{"xyxy": [60.0, 40.0, 300.0, 280.0], "tier": "A", "truncated": False, "crowd": False}]
                if i == 3:
                    row["status"] = "excluded"
                if i == 2:
                    row["status"] = "masked"; row["boxes"] = []
            st.put(row)
    st.save()
    return root, tmp_path / "labels.jsonl"

def test_dataset_items_and_targets(tmp_path):
    from det.dataset import DetDataset
    root, store = _make(tmp_path)
    ds = DetDataset(root, "train", 16, train=False, store_path=store, cache_path=tmp_path / "c.json")
    assert len(ds) == 7                                   # the excluded bear is dropped
    x, t, meta = ds[[m["path"] for m in ds.meta].index(str(root / "train/bear/bear_0.jpg"))]
    assert x.shape == (3, 360, 640) and x.dtype == torch.float32
    # 600x400 -> 540x360 letterbox, px=50: box (60,40,300,280) -> (104,36,320,252)
    np.testing.assert_allclose(meta["boxes_lb"][0], [104, 36, 320, 252], atol=1e-3)
    assert t["heat"].max() == 1.0 and t["heat_w"].sum() > 0
    xm, tm, _ = ds[[m["path"] for m in ds.meta].index(str(root / "train/bear/bear_2.jpg"))]
    assert tm["heat_w"].sum() == 0                        # masked bear: no heatmap loss
    xn, tn, _ = ds[[m["path"] for m in ds.meta].index(str(root / "train/non_bear/non_bear_0.jpg"))]
    assert tn["heat"].sum() == 0 and tn["heat_w"].sum() == 640 // 16 * 23
    assert json.load(open(tmp_path / "c.json"))           # sha1 cache written

def test_normalisation_is_bgr_mean(tmp_path):
    from det.dataset import to_tensor
    im = Image.new("RGB", (640, 360), (124, 116, 104))   # RGB that equals the BGR mean
    assert torch.abs(to_tensor(im)).max() < 1e-6

def test_sampler_mix(tmp_path):
    from det.dataset import DetDataset, DetSampler
    root, store = _make(tmp_path)
    ds = DetDataset(root, "train", 16, train=True, store_path=store, cache_path=tmp_path / "c.json")
    hard = {m["path"] for m in ds.meta if m["cls"] == "non_bear"}
    s = DetSampler(ds, batch=6, n_batches=5, hard=hard, seed=0)
    idx = list(s)
    assert len(idx) == 30
    nb = sum(ds.meta[i]["cls"] == "non_bear" for i in idx)
    assert nb == 10


def test_train_mode_runs_augmentations(tmp_path):
    from det.dataset import DetDataset, collate
    root, store = _make(tmp_path)
    ds = DetDataset(root, "train", 16, train=True, store_path=store, cache_path=tmp_path / "c.json")
    batch = collate([ds[i % len(ds)] for i in range(24)])
    assert batch[0].shape == (24, 3, 360, 640) and batch[1]["heat"].shape == (24, 1, 23, 40)
    assert torch.isfinite(batch[0]).all()

def test_worker_rng_differs_per_call(tmp_path):
    from det.dataset import DetDataset
    root, store = _make(tmp_path)
    ds = DetDataset(root, "train", 16, train=True, store_path=store, cache_path=tmp_path / "c.json")
    a = [ds._rng(0).rand() for _ in range(3)]
    assert len(set(a)) == 3


def test_sha1_cache_tmp_is_per_process(tmp_path, monkeypatch):
    """Concurrent trainers must not share one tmp file (two processes raced on it, 20/10)."""
    import det.dataset as D
    from PIL import Image
    p = tmp_path / "a.jpg"; Image.new("RGB", (4, 4)).save(p)
    seen = []
    real = D.os.replace
    monkeypatch.setattr(D.os, "replace", lambda a, b: (seen.append(str(a)), real(a, b)))
    D._sha1_cached([p], tmp_path / "c.json")
    assert seen and str(D.os.getpid()) in seen[0]


def test_mosaic_masks_the_quadrant_of_a_masked_tile(tmp_path):
    """Review #3: a masked (or unlabelled) bear pasted as a mosaic tile must become an ignore
    region, never background; the composite itself is not masked."""
    from det.dataset import DetDataset
    root, store = _make(tmp_path)
    ds = DetDataset(root, "train", 16, train=True, store_path=store, cache_path=tmp_path / "c.json")
    paths = [m["path"] for m in ds.meta]
    masked = paths.index(str(root / "train/bear/bear_2.jpg")); ok = paths.index(str(root / "train/bear/bear_0.jpg"))
    img, b, ig, comp_masked = ds._mosaic(ok, [masked, ok, ok], np.random.RandomState(0))
    assert comp_masked is False and len(b) == 3
    quads = [(0, 0, 320, 180), (320, 0, 640, 180), (0, 180, 320, 360), (320, 180, 640, 360)]
    assert any(all(abs(g[k] - q[k]) < 1 for k in range(4)) for g in ig for q in quads)
    img, b, ig, comp_masked = ds._mosaic(masked, [ok, ok, ok], np.random.RandomState(0))
    assert comp_masked is False and any(all(abs(g[k] - q[k]) < 1 for k in range(4)) for g in ig for q in quads)
