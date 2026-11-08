"""ONNX export of the YOLO candidate for the Novaic compiler.
The chip's preprocessing is fixed by the V19 build: input = (BGR - [104,116,124]) / 255.
YOLO is trained on RGB / 255, so a constant 1x1 conv in front undoes the mean and swaps
channels: rgb[c] = chip[2-c] + mean_bgr[2-c] / 255. Outputs: the 6 raw head tensors."""
from pathlib import Path

import torch
import torch.nn as nn

from det.probe_yolo import RawHead

MEAN_BGR = (104.0, 116.0, 124.0)
OUT_NAMES = [f"{k}_p{l}" for l in (3, 4, 5) for k in ("box", "cls")]


class NovaicYolo(nn.Module):
    def __init__(self, detection_model):
        super().__init__()
        self.pre = nn.Conv2d(3, 3, 1, bias=True)
        w = torch.zeros(3, 3, 1, 1)
        for c in range(3):
            w[c, 2 - c] = 1.0
        self.pre.weight.data.copy_(w)
        self.pre.bias.data.copy_(torch.tensor([MEAN_BGR[2 - c] / 255.0 for c in range(3)]))
        for p in self.pre.parameters():
            p.requires_grad_(False)
        self.net = RawHead(detection_model)

    def forward(self, x):
        return self.net(self.pre(x))


def export_onnx(net, dst):
    import onnx
    from onnxsim import simplify
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, torch.zeros(1, 3, 384, 640), str(dst), opset_version=12,
                      input_names=["input"], output_names=OUT_NAMES, do_constant_folding=True)
    sm, ok = simplify(onnx.load(str(dst)))
    if ok:
        onnx.save(sm, str(dst))
    return dst


def load_trained(pt_path):
    """ultralytics best.pt -> eval DetectionModel (float32, fused-free)."""
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    return (ck.get("ema") or ck["model"]).float().eval()
