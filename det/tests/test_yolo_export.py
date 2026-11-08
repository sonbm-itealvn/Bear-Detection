import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import numpy as np
import torch

def test_novaic_input_matches_rgb01():
    from det.probe_yolo import build, RawHead
    from det.yolo.export_onnx import NovaicYolo
    m = build("yolov8n")
    raw = RawHead(m).eval()
    wrapped = NovaicYolo(m).eval()
    rgb = torch.rand(1, 3, 384, 640)                                   # what ultralytics trains on
    bgr255 = rgb.flip(1) * 255
    chip = (bgr255 - torch.tensor([104., 116., 124.]).view(1, 3, 1, 1)) / 255   # what the chip feeds
    with torch.no_grad():
        a, b = raw(rgb), wrapped(chip)
    for u, v in zip(a, b):
        assert torch.allclose(u, v, atol=1e-4)

def test_onnx_export(tmp_path):
    import onnxruntime as ort
    from det.probe_yolo import build
    from det.yolo.export_onnx import NovaicYolo, export_onnx
    m = build("yolov8n")
    dst = tmp_path / "y.onnx"
    export_onnx(NovaicYolo(m).eval(), dst)
    x = np.random.RandomState(0).randn(1, 3, 384, 640).astype(np.float32) * 0.3
    o = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"]).run(None, {"input": x})
    with torch.no_grad():
        t = NovaicYolo(m).eval()(torch.from_numpy(x))
    assert len(o) == 6 and all(np.abs(a - b.numpy()).max() < 1e-3 for a, b in zip(o, t))
