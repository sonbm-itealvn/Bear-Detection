"""M0 probe: random-weight det heads (up_slice and convT, stride 16 and 8) ->
export -> compile -> simulate 8 images that V19's letterboxed build already simulated.
Reports: NPU vs CPU placement of the upsample, the three output dumps, classifier
logits vs the V19 build, backbone frac bits vs V19, compiler ms, .bin size.
Usage: python -m det.probe_m0 [stride up [--report-only]]   (no args = all four variants)"""
import re, subprocess, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path("/data/nas07_new/PersonalData/phuocthien")
sys.path.insert(0, str(ROOT))
import novaic_eval as ne
from det.model import load_v19, save_det
from det import sim_tools as st

V19 = ROOT / "BearDetection_V19_regnetx_1009/checkpoints/checkpoint_epoch58.pth"
REF = "v19_ep58_lb"
SHAPES = {16: {"output": (2,), "heat": (1, 23, 40), "box": (4, 23, 40)},
          8: {"output": (2,), "heat": (1, 45, 80), "box": (4, 45, 80)}}


def compile_variant(stride, up):
    name = f"det_m0_s{stride}_{up}"
    torch.manual_seed(0)
    net = load_v19(V19, stride=stride, up=up)
    ck = ROOT / "det_runs" / "m0" / f"{name}.pth"
    ck.parent.mkdir(parents=True, exist_ok=True)
    save_det(net, ck)
    for stage in ("export", "compile"):
        r = subprocess.run([sys.executable, "-u", "novaic_eval.py", "--checkpoint", str(ck), "--name", name,
                            "--n-eval", "400", "--jobs", "4", "--in-hp", "1", "--calib-mode", "0",
                            "--mean-fmt", "bgr", "--stage", stage], cwd=ROOT, capture_output=True, text=True)
        print(f"[{stage}] rc={r.returncode}\n{r.stdout[-600:]}{r.stderr[-600:]}", flush=True)
    return name


def ref_images(n=8):
    """First n letterboxed V19 test images that have a V19 simulator dump."""
    out = []
    for p in sorted((ne.NV / "input/data" / f"{REF}_test/test_images").iterdir()):
        hit = list((ne.NV / "output").glob(f"{REF}_s*/debug/sim_layer_{p.name}/float/Gemm_output_Y.bin"))
        if hit:
            out.append((p, hit[0]))
        if len(out) == n:
            break
    return out


def report(name, stride):
    out = ne.NV / "output" / name
    legal = (out / "gentool/legal/__legal_list.txt").read_text(errors="ignore")
    prof = (out / "gentool/profile.txt").read_text(errors="ignore")
    ms = re.search(r"Simulation total time\s*:\s*([\d.]+)", prof)
    binf = out / "sdk/nvt_model.bin"
    ops = sorted(set(re.findall(r"NvtAI\w+", legal)))
    cpu_ops = [o for o in ops if "Cpu" in o]
    print(f"\n=== {name}")
    print(f"  ops: {ops}")
    print(f"  ScaleUp on NPU: {any('Scaleup' in o or 'ScaleUp' in o for o in ops if 'Cpu' not in o)}   CPU ops: {cpu_ops}")
    print(f"  compiler ms: {ms.group(1) if ms else '?'}   bin: {binf.stat().st_size if binf.exists() else 'MISSING'} B")
    print(f"  network outputs: {st.network_outputs(name)}")
    a = st._bitinfo(name); b = st._bitinfo(REF)
    keys = [k for k in b if k in a]
    diff = [k for k in keys if a[k].get("output0_frac_bit_num") != b[k].get("output0_frac_bit_num")
            or a[k].get("weight_frac_bit_num") != b[k].get("weight_frac_bit_num")]
    print(f"  tensors shared with V19: {len(keys)}/{len(b)}   frac-bit differences: {len(diff)} {diff[:10]}")
    refs = ref_images(8)
    staged = st.stage_images(name, "m0", [p for p, _ in refs], exact=True)   # same bytes as V19's run
    print(f"  sim rcs: {st.run_sim(name, 'm0', staged, shards=4)}", flush=True)
    n_eq = 0
    for s, (src, refbin) in zip(staged, refs):
        try:
            o = st.read_outputs(name, "m0", s, SHAPES[stride])
        except FileNotFoundError as e:
            print(f"  {s}: {e}")
            continue
        refv = np.fromfile(refbin, np.float32)[:2]
        eq = np.array_equal(o["output"], refv)
        n_eq += eq
        print(f"  {s}: cls {o['output']} vs V19 {refv} equal={eq}  heat {o['heat'].shape} "
              f"[{o['heat'].min():.3f},{o['heat'].max():.3f}] box {o['box'].shape}")
    print(f"  classifier equal to V19 on {n_eq}/{len(refs)} images", flush=True)


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    if len(args) == 2:
        variants = [(int(args[0]), args[1])]
    else:
        variants = [(16, "up_slice"), (16, "convT"), (8, "up_slice"), (8, "convT")]
    for stride, up in variants:
        name = f"det_m0_s{stride}_{up}" if "--report-only" in sys.argv else compile_variant(stride, up)
        report(name, stride)
