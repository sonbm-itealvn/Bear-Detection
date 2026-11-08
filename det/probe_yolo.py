"""M0-style probe of YOLO detectors on the Novaic toolchain (random weights: ops, ms, MB only).
SiLU -> ReLU (Novaic constraint), nc=1, input 1x3x384x640, raw head: 6 outputs
(box-DFL and class conv per level P3/P4/P5); decoding would run on the camera CPU.
python -m det.probe_yolo yolov8n|yolo11n|yolov8s"""
import re, subprocess, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/data/nas07_new/PersonalData/phuocthien")
sys.path.insert(0, str(ROOT))
import novaic_eval as ne
from det import sim_tools as st


class RawHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x):
        m = self.m
        y = []
        for layer in m.model[:-1]:
            if layer.f != -1:
                x = y[layer.f] if isinstance(layer.f, int) else [x if j == -1 else y[j] for j in layer.f]
            x = layer(x)
            y.append(x if layer.i in m.save else None)
        det = m.model[-1]
        feats = [y[j] if j != -1 else x for j in det.f]
        outs = []
        for i, f in enumerate(feats):
            outs += [det.cv2[i](f), det.cv3[i](f)]
        return tuple(outs)


def build(arch):
    import ultralytics.nn.modules.conv as C
    from ultralytics.nn.tasks import DetectionModel
    C.Conv.default_act = nn.ReLU()
    torch.manual_seed(0)
    m = DetectionModel(f"{arch}.yaml", nc=1, verbose=False).eval()
    silu = [n for n, mod in m.named_modules() if isinstance(mod, nn.SiLU)]
    assert not silu, f"SiLU left: {silu[:5]}"
    return m


def main():
    arch = sys.argv[1]
    name = f"yolo_m0_{arch}_relu"
    net = RawHead(build(arch)).eval()
    n_par = sum(p.numel() for p in net.parameters())
    md = ne.NV / "input/model/customer" / name
    md.mkdir(parents=True, exist_ok=True)
    dst = md / "deploy_float.onnx"
    x = torch.randn(1, 3, 384, 640)
    with torch.no_grad():
        outs = net(x)
    names = [f"{k}_p{l}" for l in (3, 4, 5) for k in ("box", "cls")]
    torch.onnx.export(net, x, str(dst), opset_version=12, input_names=["input"], output_names=names,
                      do_constant_folding=True)
    import onnx
    from onnxsim import simplify
    sm, ok = simplify(onnx.load(str(dst)))
    if ok:
        onnx.save(sm, str(dst))
    ops = sorted({n.op_type for n in onnx.load(str(dst)).graph.node})
    print(f"=== {name}: params {n_par / 1e6:.2f} M, onnx ops {ops}")
    print("  outputs:", [tuple(o.shape) for o in outs], flush=True)
    r = subprocess.run([sys.executable, "-u", "novaic_eval.py", "--name", name, "--n-eval", "400", "--jobs", "4",
                        "--in-hp", "1", "--calib-mode", "0", "--mean-fmt", "bgr", "--net-h", "384", "--stage", "compile"],
                       cwd=ROOT, capture_output=True, text=True)
    print(f"  [compile] rc={r.returncode} {r.stdout[-300:]}", flush=True)
    out = ne.NV / "output" / name
    legal = (out / "gentool/legal/__legal_list.txt").read_text(errors="ignore")
    prof = (out / "gentool/profile.txt").read_text(errors="ignore")
    ms = re.search(r"Simulation total time\s*:\s*([\d.]+)", prof)
    nops = sorted(set(re.findall(r"NvtAI\w+", legal)))
    binf = out / "sdk/nvt_model.bin"
    print(f"  novaic ops: {nops}")
    print(f"  CPU ops: {[o for o in nops if 'Cpu' in o]}")
    print(f"  compiler ms: {ms.group(1) if ms else '?'}   bin: {binf.stat().st_size if binf.exists() else 'MISSING'} B")
    print(f"  network outputs: {st.network_outputs(name)}", flush=True)


if __name__ == "__main__":
    main()
