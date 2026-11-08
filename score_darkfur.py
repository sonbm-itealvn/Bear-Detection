"""False-positive rate on the held-out dark-fur test set, per species, per model.

Every image is a non-bear, so any "bear" answer is a false positive. FP32 is
used as the proxy for the device: measured on the simulator, non-bear margins
survive quantization almost unchanged (device ≈ 0.98·FP32 + 0.01), so the FP32
false-positive rate is a close stand-in and runs in minutes instead of hours.

V16 is scored with plain argmax, as its README says it ships -- no threshold,
no bias shift.

    python score_darkfur.py --model V16=<ckpt> --model ep61=<ckpt> [--sheet out.jpg]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import models, transforms

BASE = Path("/data/nas07_new/PersonalData/phuocthien")
ROOT = BASE / "darkfur_test_2409"


class Letterbox:
    def __init__(self, h=360, w=640):
        self.h, self.w = h, w

    def __call__(self, im):
        w, h = im.size
        k = min(self.w / w, self.h / h)
        nw, nh = max(1, int(w * k)), max(1, int(h * k))
        c = Image.new("RGB", (self.w, self.h), (0, 0, 0))
        c.paste(TF.resize(im, (nh, nw)), ((self.w - nw) // 2, (self.h - nh) // 2))
        return c


def load(path, dev):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    bb = (raw.get("config") or {}).get("backbone") if isinstance(raw, dict) else None
    if bb and bb != "torchvision_mobilenet_v2":
        return _load_timm(raw, bb, dev)
    m = models.mobilenet_v2(weights=None)

    def sw(mod):
        for n, c in mod.named_children():
            if isinstance(c, nn.ReLU6):
                setattr(mod, n, nn.ReLU(inplace=True))
            else:
                sw(c)
    sw(m)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, 2))
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in sd.get("model_state_dict", sd).items()}
    r = m.load_state_dict(sd, strict=False)
    if r.missing_keys or [k for k in r.unexpected_keys if k.endswith(".bias")]:
        raise SystemExit(f"{path}: {len(r.missing_keys)} missing / "
                         f"{len(r.unexpected_keys)} unexpected keys -- wrong checkpoint format")
    return m.eval().to(dev)


def _load_timm(raw, bb, dev):
    """timm backbones trained with OVERRIDE_BACKBONE (V19 RegNetX): same wrapper
    as the trainer -- forward_features, global average pool, Dropout+Linear."""
    import os
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    import timm
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = timm.create_model(bb, pretrained=False, num_classes=0)
            self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.m.num_features, 2))

        def forward(self, x):
            return self.classifier(torch.flatten(F.adaptive_avg_pool2d(self.m.forward_features(x), 1), 1))
    net = Net()
    sd = {k.replace("module.", "").replace("features.m.", "m.", 1): v
          for k, v in raw["model_state_dict"].items() if not k.startswith("projection.")}
    net.load_state_dict(sd, strict=True)
    return net.eval().to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="label=path")
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--sheet", help="contact sheet of the worst false positives")
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}" if a.gpu >= 0 and torch.cuda.is_available() else "cpu"

    files = sorted(ROOT.rglob("*.jpg"))
    tf = transforms.Compose([Letterbox(), transforms.ToTensor()])

    def pre(x):
        x = x[:, [2, 1, 0], :, :].clone()
        for i, m in enumerate((104, 116, 124)):
            x[:, i] -= m / 255
        return x

    margins = {}
    for spec in a.model:
        label, path = spec.split("=", 1)
        net = load(path, dev)
        out = []
        with torch.no_grad():
            for i in range(0, len(files), 32):
                X = torch.stack([tf(Image.open(p).convert("RGB")) for p in files[i:i + 32]]).to(dev)
                o = net(pre(X))
                out += (o[:, 1] - o[:, 0]).cpu().tolist()
        margins[label] = np.array(out)

    species = [p.parent.name for p in files]
    labels = list(margins)
    print(f"  {len(files)} ảnh không-gấu, ban ngày, chưa từng có trong train\n")
    print(f"  {'loài':<24}{'n':>5}" + "".join(f"{l:>12}" for l in labels))
    agg = defaultdict(lambda: defaultdict(list))
    for sp, *vals in zip(species, *[margins[l] for l in labels]):
        for l, v in zip(labels, vals):
            agg[sp][l].append(v > 0)
    order = sorted(agg, key=lambda sp: -np.mean(agg[sp][labels[0]]))
    for sp in order:
        n = len(agg[sp][labels[0]])
        print(f"  {sp:<24}{n:>5}" + "".join(
            f"{100*np.mean(agg[sp][l]):>11.1f}%" for l in labels))
    print(f"  {'TỔNG':<24}{len(files):>5}" + "".join(
        f"{100*np.mean(margins[l] > 0):>11.1f}%" for l in labels))

    (BASE / "darkfur_scores.json").write_text(json.dumps(
        {"files": [str(p.relative_to(ROOT)) for p in files],
         **{l: [round(float(x), 3) for x in margins[l]] for l in labels}}))

    if a.sheet:
        l0 = labels[0]
        idx = np.argsort(-margins[l0])[:24]
        W, H, cols = 320, 180, 6
        sheet = Image.new("RGB", (W * cols, H * 4), (12, 12, 12))
        from PIL import ImageDraw
        dr = ImageDraw.Draw(sheet)
        for j, i in enumerate(idx):
            x, y = (j % cols) * W, (j // cols) * H
            im = Image.open(files[i]).convert("RGB")
            im.thumbnail((W, H))
            sheet.paste(im, (x + (W - im.width) // 2, y + (H - im.height) // 2))
            dr.rectangle([x, y, x + W - 1, y + 13], fill=(0, 0, 0))
            dr.text((x + 2, y + 1), f"{species[i][:14]} " +
                    " ".join(f"{l}:{margins[l][i]:+.1f}" for l in labels), fill=(255, 230, 0))
        sheet.save(a.sheet, quality=88)
        print(f"\n  -> {a.sheet}")


if __name__ == "__main__":
    main()
