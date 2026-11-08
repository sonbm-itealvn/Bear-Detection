import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import numpy as np

def test_eval_yolo_runs_on_synthetic(tmp_path):
    from det.tests.test_yolo_dataset import _setup
    from det.probe_yolo import build
    from det.yolo.export_onnx import NovaicYolo
    from det.yolo.eval_yolo import evaluate_rows
    from det.labels.store import LabelStore
    root, st = _setup(tmp_path)
    st = LabelStore(tmp_path / "s.jsonl")
    rows = [r for r in st.rows.values() if "/val/" in r["path_hint"]]
    r = evaluate_rows(NovaicYolo(build("yolov8n")).eval(), rows, "cpu", batch=3)
    assert r["n_gt"] == 4 and r["n_neg"] == 4 and 0.0 <= r["ap50"] <= 1.0
    assert "img_level" in r and set(r["img_level"]) >= {"thr", "recall", "precision"}

def test_letterbox_tensor_shape_and_pad():
    from PIL import Image
    from det.yolo.eval_yolo import to_input
    x = to_input(Image.new("RGB", (1920, 1080), (255, 255, 255)))
    assert tuple(x.shape) == (3, 384, 640)
    # pad rows are black: (0 - mean) / 255 in BGR order
    np.testing.assert_allclose(x[:, 0, 0].numpy(), [-104 / 255, -116 / 255, -124 / 255], atol=1e-6)
