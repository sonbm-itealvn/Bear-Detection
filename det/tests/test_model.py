import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import torch, pytest
from det.model import TimmDetNet, load_v19, save_det, load_det

V19 = "/data/nas07_new/PersonalData/phuocthien/BearDetection_V19_regnetx_1009/checkpoints/checkpoint_epoch58.pth"

@pytest.mark.parametrize("stride,up,shape", [(16, "up_slice", (23, 40)), (16, "convT", (23, 40)),
                                             (8, "up_slice", (45, 80)), (8, "convT", (45, 80))])
def test_shapes(stride, up, shape):
    net = TimmDetNet(stride=stride, up=up).eval()
    with torch.no_grad():
        logits, heat, box = net(torch.zeros(1, 3, 360, 640))
    assert logits.shape == (1, 2) and heat.shape == (1, 1, *shape) and box.shape == (1, 4, *shape)

def test_classifier_identical_to_v19_wrapper():
    """The det net's classifier path must equal the V19 export wrapper exactly."""
    import timm, torch.nn as nn, torch.nn.functional as F
    raw = torch.load(V19, map_location="cpu", weights_only=False)
    net = load_v19(V19)
    ref = timm.create_model("regnetx_008", pretrained=False, num_classes=0)
    cls = nn.Sequential(nn.Dropout(0.3), nn.Linear(ref.num_features, 2))
    sd = {k.replace("features.m.", "", 1): v for k, v in raw["model_state_dict"].items() if k.startswith("features.m.")}
    ref.load_state_dict(sd)
    cls.load_state_dict({k[len("classifier."):]: v for k, v in raw["model_state_dict"].items() if k.startswith("classifier.")})
    ref.eval(); cls.eval()
    x = torch.randn(2, 3, 360, 640)
    with torch.no_grad():
        a = net(x)[0]
        b = cls(torch.flatten(F.adaptive_avg_pool2d(ref.forward_features(x), 1), 1))
    assert torch.equal(a, b)

def test_frozen_and_train_mode_keeps_backbone_eval():
    net = load_v19(V19)
    net.train()
    assert not net.m.training and not net.classifier.training and net.head.training
    assert all(not p.requires_grad for p in net.m.parameters())
    assert all(p.requires_grad for p in net.head.parameters())

def test_save_load_roundtrip(tmp_path):
    net = load_v19(V19, stride=16, up="convT")
    p = tmp_path / "d.pth"; save_det(net, p)
    net2 = load_det(p).eval(); net.eval()
    x = torch.randn(1, 3, 360, 640)
    with torch.no_grad():
        for u, v in zip(net(x), net2(x)):
            assert torch.equal(u, v)


def test_export_matches_torch(tmp_path):
    import numpy as np, onnxruntime as ort
    from det.export import export_det_onnx
    net = load_v19(V19, stride=16, up="up_slice")
    p = tmp_path / "d.pth"; save_det(net, p)
    dst = tmp_path / "d.onnx"
    export_det_onnx(torch.load(p, map_location="cpu", weights_only=False), dst)
    x = np.random.RandomState(0).randn(1, 3, 360, 640).astype(np.float32)
    o = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"]).run(None, {"input": x})
    with torch.no_grad():
        t = [v.numpy() for v in net.eval()(torch.from_numpy(x))]
    for a, b in zip(o, t):
        assert np.abs(a - b).max() < 1e-3
