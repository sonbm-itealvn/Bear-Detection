"""V19 RegNetX-800MF (frozen) + CenterNet/TTFNet-style detection head."""
import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from det.geometry import GRID


class Up2Crop(nn.Module):
    """Nearest x2 then crop rows: the NPU's ScaleUp needs equal H/W scale."""
    def __init__(self, rows):
        super().__init__()
        self.rows = rows

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="nearest")[:, :, :self.rows, :]


def make_up(mode, ch, rows):
    if mode == "up_slice":
        return Up2Crop(rows)
    if mode == "convT":   # (H-1)*2 - 2 + 3 + op -> 12->23 (op_h=0), 20->40 (op_w=1)
        return nn.ConvTranspose2d(ch, ch, 3, stride=2, padding=1, output_padding=(0, 1))
    raise ValueError(f"unknown upsample mode {mode}")


def cbr(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DetHead(nn.Module):
    def __init__(self, stride=16, up="up_slice", ch=128, c2=128, c3=288, c4=672):
        super().__init__()
        assert stride in (8, 16)
        self.stride = stride
        self.lat4 = nn.Conv2d(c4, ch, 1)
        self.lat3 = nn.Conv2d(c3, ch, 1)
        self.up4 = make_up(up, ch, GRID[16][0])
        if stride == 8:
            self.lat2 = nn.Conv2d(c2, ch, 1)
            self.up3 = make_up(up, ch, GRID[8][0])
        self.tower = nn.Sequential(cbr(ch, ch), cbr(ch, ch))
        self.heat = nn.Conv2d(ch, 1, 1)
        self.box = nn.Conv2d(ch, 4, 1)
        nn.init.normal_(self.heat.weight, std=0.01)
        nn.init.constant_(self.heat.bias, -2.19)
        nn.init.normal_(self.box.weight, std=0.01)
        nn.init.zeros_(self.box.bias)

    def forward(self, s2, s3, s4):
        x = self.lat3(s3) + self.up4(self.lat4(s4))
        if self.stride == 8:
            x = self.lat2(s2) + self.up3(x)
        x = self.tower(x)
        return self.heat(x), self.box(x)


class TimmDetNet(nn.Module):
    def __init__(self, backbone="regnetx_008", stride=16, up="up_slice", ch=128):
        super().__init__()
        self.backbone_name = backbone
        self.head_cfg = {"stride": stride, "up": up, "ch": ch}
        self.m = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.m.num_features, 2))
        self.head = DetHead(stride, up, ch)

    def backbone_feats(self, x):
        m = self.m
        y = m.s1(m.stem(x))
        s2 = m.s2(y)
        s3 = m.s3(s2)
        s4 = m.final_conv(m.s4(s3))
        return s2, s3, s4

    def forward(self, x):
        s2, s3, s4 = self.backbone_feats(x)
        logits = self.classifier(torch.flatten(F.adaptive_avg_pool2d(s4, 1), 1))
        heat, box = self.head(s2, s3, s4)
        return logits, heat, box

    def freeze_backbone(self):
        for p in list(self.m.parameters()) + list(self.classifier.parameters()):
            p.requires_grad_(False)
        self.m.eval()
        self.classifier.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        self.m.eval()            # frozen: BN stats and dropout never change
        self.classifier.eval()
        return self


def _clean(sd):
    return {k.replace("module.", "").replace("features.m.", "m.", 1): v
            for k, v in sd.items() if not k.startswith("projection.")}


def load_v19(path, stride=16, up="up_slice", ch=128):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    net = TimmDetNet(raw["config"]["backbone"], stride, up, ch)
    missing, unexpected = net.load_state_dict(_clean(raw["model_state_dict"]), strict=False)
    bad = [k for k in missing if not k.startswith("head.")]
    if bad or unexpected:
        raise RuntimeError(f"V19 load mismatch: missing {bad[:5]} unexpected {unexpected[:5]}")
    return net.freeze_backbone()


def save_det(net, path, extra=None):
    d = {"model_state_dict": net.state_dict(),
         "config": {"backbone": net.backbone_name, "det_head": dict(net.head_cfg)}}
    d.update(extra or {})
    torch.save(d, path)


def net_from_det_raw(raw):
    c = raw["config"]
    net = TimmDetNet(c["backbone"], **c["det_head"])
    net.load_state_dict(raw["model_state_dict"], strict=True)
    return net.freeze_backbone()


def load_det(path):
    return net_from_det_raw(torch.load(path, map_location="cpu", weights_only=False))
