import json, subprocess, sys
from pathlib import Path
from det.yolo.package_yolo import package_yolo

def test_layout_and_standalone_decoder(tmp_path):
    build = tmp_path / "b"; (build / "sdk").mkdir(parents=True); (build / "gentool").mkdir()
    (build / "sdk/nvt_model.bin").write_bytes(b"\x02" * 500)
    (build / "gentool/quan_info_bitinfo.json").write_text(json.dumps({"Conv_box_p3_Y": {"is_network_out": True, "output0_frac_bit_num": 11, "output0_bitdepth": 16}}))
    (build / "gentool/profile.txt").write_text("|-- Simulation total time : 16.36(ms)\n")
    model = tmp_path / "m"; model.mkdir(); (model / "deploy_float.onnx").write_text("x"); (model / "mean_data.txt").write_text("104,116,124")
    cfg = tmp_path / "c"; cfg.mkdir(); (cfg / "gen_config.txt").write_text("g")
    run = tmp_path / "run"; (run / "weights").mkdir(parents=True); (run / "weights/best.pt").write_bytes(b"pt")
    (run / "test_vectors").mkdir(); (run / "test_vectors/v00.bin").write_bytes(b"0"); (run / "test_vectors/expected.txt").write_text("# v00\n")
    (run / "chip_report.json").write_text(json.dumps({"n": 781, "n_neg": 194, "median_iou_chip_vs_fp32": .98, "ap50_fp32": .82, "ap50_chip": .81, "ap_fp32": .73, "ap_chip": .71, "pass": True}))
    gated = tmp_path / "g.json"; gated.write_text(json.dumps({"a": {"ap50": .77, "ap": .46, "iou": {"median": .81}, "rec_fppi01": .78, "ap50_size": {"small": .4}},
                                                             "b": {"ap50": .785, "ap": .686, "iou": {"median": .94}, "rec_fppi01": .81, "ap50_size": {"small": .5}}}))
    d = package_yolo(run, build, model, cfg, gated, tmp_path / "out", "T")
    names = {p.name for p in d.rglob("*")}
    assert {"nvt_model.bin", "README.md", "yolo_det_decode.c", "yolo_det_decode.h", "bear_det_decode.h", "yolo_test_main.c",
            "decode_ref.py", "geometry.py", "gen_config.txt", "expected.txt"} <= names
    assert "16.36" in (d / "README.md").read_text() and "640x384" in (d / "README.md").read_text()
    subprocess.check_call([sys.executable, "-c", "import decode_ref"], cwd=d)
