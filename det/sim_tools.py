"""Stage letterboxed images, run the Novaic simulator in shards, read all network outputs.
Per image the simulator writes debug/sim_layer_<img>/float/<key>.bin: the FIXED result
dequantized, dense NCHW float32 (verified 13/10 on the Gemm dump). Dump file names are
the bitinfo key with '/' replaced by '_'."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

import novaic_eval as ne

NV = ne.NV


def dump_name(key):
    return key.replace("/", "_") + ".bin"


def _match_outputs(bitinfo):
    """Map ONNX output names to compiler dump keys via the producing node's name."""
    out = {}
    for k, v in bitinfo.items():
        if not v.get("is_network_out"):
            continue
        g = v.get("ori_graph_name", "")
        if "/head/heat/" in g:
            out["heat"] = k
        elif "/head/box/" in g:
            out["box"] = k
        elif k.startswith("Gemm"):
            out["output"] = k
    return out


def _bitinfo(name, fname="quan_info_bitinfo.json"):
    return json.load(open(NV / "output" / name / "gentool" / fname))


def onnx_output_map(name):
    return _match_outputs(_bitinfo(name))


def network_outputs(name):
    info = _bitinfo(name)
    return [{"key": k, "frac": v.get("output0_frac_bit_num"), "bits": v.get("output0_bitdepth")}
            for k, v in info.items() if v.get("is_network_out")]


def stage_images(name, tag, paths, exact=False):
    """exact=False: letterbox + re-encode (camera-like input from an original image).
    exact=True: byte-for-byte copy under the same name -- for images that are already
    letterboxed simulator inputs (re-encoding would change pixels and break a
    bit-exact comparison with an earlier build)."""
    d = NV / "input/data" / f"{name}_{tag}"
    img_dir = d / "test_images"
    if d.exists():
        shutil.rmtree(d)
    img_dir.mkdir(parents=True)
    staged = []
    for i, p in enumerate(paths):
        if exact:
            n = Path(p).name
            shutil.copy2(p, img_dir / n)
        else:
            n = f"{i:05d}__{Path(p).stem}.jpg"
            ne.copy_as_rgb_jpeg(Path(p), img_dir / n)
        staged.append(n)
    (d / "test_img_list.txt").write_text("\n".join(staged) + "\n")
    (d / "test_ground_truth.txt").write_text("\n".join("0" for _ in staged) + "\n")
    (d / "label.txt").write_text("non_bear\nbear\n")
    return staged


def run_sim(name, tag, staged, shards=4, keep_bins=False):
    model_dir = NV / "input/model/customer" / name
    img_dir = NV / "input/data" / f"{name}_{tag}" / "test_images"
    procs = []
    for k in range(shards):
        sh = staged[k::shards]
        if not sh:
            continue
        sd = NV / "input/data" / f"{name}_{tag}_s{k}"
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "test_img_list.txt").write_text("\n".join(sh) + "\n")
        (sd / "test_ground_truth.txt").write_text("\n".join("0" for _ in sh) + "\n")
        (sd / "label.txt").write_text("non_bear\nbear\n")
        out = NV / "output" / f"{name}_{tag}_s{k}"
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(NV / "output" / name, out)
        pat = f"{name}_{tag}_s{k}"
        ne.write_cfg(NV / "config" / pat / "cnn25/sim_config.txt",
                     ne.SIM_CONFIG.format(net_h=ne.NET_H, data_dir=sd, model_dir=model_dir, img_dir=img_dir,
                                          name=pat, out_dir=out))
        shutil.copy2(NV / "config" / name / "cnn25/gen_config.txt",
                     NV / "config" / pat / "cnn25/gen_config.txt")
        log = ne.BASE / "novaic_eval_logs" / f"sim_{pat}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        procs.append(subprocess.Popen(
            [str(ne.BIN / "simulator.nvtai"), "--config-dir", str(NV), "--pattern-name", pat,
             "--chip", ne.CHIP, "--emu-onlyRstLayer", "1", "--delete-bin", "0" if keep_bins else "1",
             "--openmp-en", "4", "--verbose", "3"],
            cwd=NV, env=ne.env(), stdout=open(log, "w"), stderr=subprocess.STDOUT))
    return [p.wait() for p in procs]


def _find_img_dir(name, tag, img):
    for d in sorted((NV / "output").glob(f"{name}_{tag}_s*")):
        p = d / "debug" / f"sim_layer_{img}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no simulator dump for {img}")


def read_outputs(name, tag, img, shapes):
    m = onnx_output_map(name)
    base = _find_img_dir(name, tag, img)
    res = {}
    for oname, key in m.items():
        f = base / "float" / dump_name(key)
        if not f.exists():
            raise FileNotFoundError(f"{f} missing")
        a = np.fromfile(f, np.float32)
        n = int(np.prod(shapes[oname]))
        res[oname] = a[:n].reshape(shapes[oname])
    return res
