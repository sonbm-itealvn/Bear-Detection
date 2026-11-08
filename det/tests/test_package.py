import json, zipfile, hashlib
from pathlib import Path
import torch
from det.package import package

def test_package_layout(tmp_path):
    out = tmp_path / "out"; build = tmp_path / "build"; run = tmp_path / "run"
    for d in (build / "sdk", build / "gentool", run / "test_vectors"):
        d.mkdir(parents=True)
    (build / "sdk/nvt_model.bin").write_bytes(b"\x01" * 1000)
    (build / "gentool/quan_info_bitinfo.json").write_text(json.dumps({
        "Gemm_output_Y": {"is_network_out": True, "ori_graph_name": "/classifier/classifier.1/Gemm", "output0_frac_bit_num": 10, "output0_bitdepth": 16},
        "Conv_/head/heat/Conv_output_0_Y": {"is_network_out": True, "ori_graph_name": "/head/heat/Conv", "output0_frac_bit_num": 9, "output0_bitdepth": 16},
        "Conv_/head/box/Conv_output_0_Y": {"is_network_out": True, "ori_graph_name": "/head/box/Conv", "output0_frac_bit_num": 11, "output0_bitdepth": 16}}))
    (build / "gentool/profile.txt").write_text("|-- Simulation total time : 33.5(ms)\n")
    model = tmp_path / "model"; model.mkdir()
    for f in ("deploy_float.onnx", "mean_data.txt"):
        (model / f).write_text(f)
    cfg = tmp_path / "cfg"; cfg.mkdir(); (cfg / "gen_config.txt").write_text("gen")
    torch.save({"model_state_dict": {"w": torch.zeros(1)}, "config": {"backbone": "regnetx_008", "det_head": {"stride": 16, "up": "up_slice", "ch": 128}}, "val": {}}, run / "best.pth")
    (run / "test_vectors/v00.bin").write_bytes(b"0"); (run / "test_vectors/expected.txt").write_text("# v00.bin\n")
    (run / "chip_report.json").write_text(json.dumps({"pass": True, "classifier": {"n": 400, "max_abs": 0.0, "argmax_changes": 0, "pass": True},
        "dark": {"n_neg": 48, "fp": 0, "max_neg_margin": -2.1, "pass": True}, "detection": {"median_iou_chip_vs_fp32": 0.97, "ap50_fp32": 0.9, "ap50_chip": 0.89, "ap50_drop": 0.01, "pass": True},
        "c_parity": {"n": 20, "max_px": 0.01, "pass": True}}))
    (run / "eval_test.json").write_text(json.dumps({"ap50": 0.9, "ap": 0.6, "rec_fppi01": 0.85, "iou": {"median": 0.8}, "ap50_domain": {"ir": 0.88}, "ap50_size": {"small": 0.5, "large": 0.95}, "n_gt": 700}))
    d = package(run / "best.pth", build, model, out, tag="V19det_test", config_dir=cfg)
    names = {p.name for p in d.rglob("*")}
    assert {"nvt_model.bin", "README.md", "bear_det_decode.c", "bear_det_decode.h", "test_main.c", "decode_ref.py",
            "geometry.py", "checkpoint_det.pth", "expected.txt", "quan_info_bitinfo.json", "gen_config.txt"} <= names
    assert "from geometry import" in (d / "decode_ref.py").read_text() and "from det." not in (d / "decode_ref.py").read_text()
    import subprocess, sys
    subprocess.check_call([sys.executable, "-c", "import decode_ref"], cwd=d)        # standalone import works
    readme = (d / "README.md").read_text()
    assert hashlib.md5(b"\x01" * 1000).hexdigest() in readme and "frac" in readme and "33.5" in readme
    ck = torch.load(d / "checkpoint_det.pth", weights_only=False)
    assert set(ck) == {"model_state_dict", "config"}
    assert zipfile.is_zipfile(out / "V19det_test.zip") and zipfile.is_zipfile(out / "V19det_test_small.zip")
