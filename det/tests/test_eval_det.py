import numpy as np
import torch
from det.eval_det import evaluate

class _Stub(torch.nn.Module):
    """Predicts one perfect box per bear image from the dataset's own targets."""
    def __init__(self, ds):
        super().__init__(); self.ds = ds; self.k = 0
    def forward(self, x):
        n = x.shape[0]
        heat = torch.full((n, 1, 23, 40), -9.0); box = torch.zeros((n, 4, 23, 40))
        for b in range(n):
            _, t, m = self.ds[self.k]; self.k += 1
            if m["boxes_lb"].shape[0]:
                i, j = np.unravel_index(t["heat"][0].numpy().argmax(), (23, 40))
                heat[b, 0, i, j] = 5.0; box[b, :, i, j] = t["box"][:, i, j]
        return torch.zeros(n, 2), heat, box

def test_evaluate_perfect_predictions(tmp_path):
    from det.dataset import DetDataset
    from det.tests.test_dataset import _make
    root, store = _make(tmp_path)
    ds = DetDataset(root, "train", 16, train=False, store_path=store, cache_path=tmp_path / "c.json")
    r = evaluate(_Stub(ds), ds, "cpu", stride=16, batch=3)
    assert r["n_gt"] == 2 and abs(r["ap50"] - 1.0) < 1e-6 and r["iou"]["median"] > 0.99
    assert "ir" in r["ap50_domain"] and r["n_neg"] == 4


class _TwoPeaks(torch.nn.Module):
    """One image, two bears; the second centre logit is -1.5 (p=0.18 < 0.3)."""
    def __init__(self, t):
        super().__init__(); self.t = t
    def forward(self, x):
        heat = torch.full((1, 1, 23, 40), -9.0); box = torch.zeros((1, 4, 23, 40))
        for (i, j), z in zip(self.t["peaks"], (5.0, -1.5)):
            heat[0, 0, i, j] = z; box[0, :, i, j] = torch.from_numpy(self.t["box"][:, i, j])
        return torch.tensor([[0.0, 1.0]]), heat, box

def test_evaluate_scores_the_full_pr_curve():
    """Review #1: AP must not be computed on boxes >= 0.3 only."""
    import numpy as np
    from det.targets import encode
    from det.eval_det import evaluate
    boxes = np.array([[20., 40., 120., 200.], [400., 60., 600., 300.]])
    t = encode(boxes, np.zeros((0, 4)), 16, masked=False)
    peaks = [tuple(int(v) for v in np.argwhere(t["heat"][0] == 1.0)[k]) for k in range(2)]
    class DS(torch.utils.data.Dataset):
        def __len__(self): return 1
        def __getitem__(self, i):
            return torch.zeros(3, 360, 640), {k: torch.from_numpy(v) for k, v in t.items() if k != "n_pos"}, \
                {"path": "/x/test/bear/a.jpg", "w": 640, "h": 360, "boxes_lb": boxes, "masked": False, "domain": "ir"}
    r = evaluate(_TwoPeaks({"peaks": peaks, "box": t["box"]}), DS(), "cpu", stride=16, batch=1)
    assert abs(r["ap50"] - 1.0) < 1e-6
