import json
import torch

def test_one_epoch_cpu_writes_checkpoints(tmp_path):
    from det.tests.test_dataset import _make
    from det.train_det import main
    root, store = _make(tmp_path)
    out = tmp_path / "run"
    main(["--stride", "16", "--up", "up_slice", "--gpu", "cpu", "--epochs", "1", "--out", str(out),
          "--root", str(root), "--store", str(store), "--cache", str(tmp_path / "c.json"),
          "--train-split", "train", "--val-split", "train", "--batch", "3", "--n-batches", "2",
          "--workers", "0"])
    assert (out / "ep1.pth").exists() and (out / "best.pth").exists()
    raw = torch.load(out / "ep1.pth", map_location="cpu", weights_only=False)
    assert raw["config"]["det_head"]["up"] == "up_slice" and "ap50" in raw["val"]
    log = [json.loads(l) for l in open(out / "log.jsonl")]
    assert len(log) == 1 and log[0]["loss"] == log[0]["loss"]          # finite (not NaN)

def test_backbone_unchanged_by_training(tmp_path):
    from det.tests.test_dataset import _make
    from det.train_det import main, V19
    from det.model import load_v19
    root, store = _make(tmp_path)
    out = tmp_path / "run"
    main(["--stride", "16", "--up", "up_slice", "--gpu", "cpu", "--epochs", "1", "--out", str(out),
          "--root", str(root), "--store", str(store), "--cache", str(tmp_path / "c.json"),
          "--train-split", "train", "--val-split", "train", "--batch", "3", "--n-batches", "2", "--workers", "0"])
    a = load_v19(V19).state_dict()
    b = torch.load(out / "ep1.pth", map_location="cpu", weights_only=False)["model_state_dict"]
    for k, v in a.items():
        if not k.startswith("head."):
            assert torch.equal(v, b[k]), k
