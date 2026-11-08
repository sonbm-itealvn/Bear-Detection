"""ONNX export of a det checkpoint: outputs 'output' (classifier, ends in Gemm, unchanged
from V19), 'heat' [1,1,H,W] and 'box' [1,4,H,W] (both end in a plain 1x1 Conv)."""
from pathlib import Path
import torch
from det.model import net_from_det_raw


def export_det_onnx(raw, dst):
    import onnx
    from onnxsim import simplify
    net = net_from_det_raw(raw).eval()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, torch.randn(1, 3, 360, 640), str(dst), opset_version=12,
                      export_params=True, do_constant_folding=True,
                      input_names=["input"], output_names=["output", "heat", "box"])
    sm, ok = simplify(onnx.load(str(dst)))
    if ok:
        onnx.save(sm, str(dst))
    outs = [o.name for o in onnx.load(str(dst)).graph.output]
    assert outs == ["output", "heat", "box"], outs
    print(f"  exported {dst} (det head {raw['config']['det_head']})")
