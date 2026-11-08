"""Train the detection head on frozen V19 (spec §3).
python -m det.train_det --stride 16 --up up_slice --gpu 2 --out det_runs/s16
python -m det.train_det --build-hard-nb --gpu 2      (once: V19's top-20k train non_bear)"""
import argparse, json, math, os, sys, time
from pathlib import Path

ROOT = Path("/data/nas07_new/PersonalData/phuocthien")
V19 = ROOT / "BearDetection_V19_regnetx_1009/checkpoints/checkpoint_epoch58.pth"
HARD_NB = ROOT / "det_runs/hard_nb.txt"


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True, help="one GPU index, or 'cpu' (tests)")
    ap.add_argument("--stride", type=int, default=16, choices=[8, 16])
    ap.add_argument("--up", default="up_slice", choices=["up_slice", "convT"])
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-batches", type=int, default=0, help="batches per epoch (0 = one pass over bears)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "det_runs/s16"))
    ap.add_argument("--root", default=str(ROOT / "bear_dataset_v_final_final"))
    ap.add_argument("--store", default=str(ROOT / "det_labels/labels_v1.jsonl"))
    ap.add_argument("--cache", default=str(ROOT / "det_labels/sha1_cache.json"))
    ap.add_argument("--eval-set", default=str(ROOT / "det_labels/eval_set.json"))
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--build-hard-nb", action="store_true")
    a = ap.parse_args(argv)
    if a.gpu in ("0", "7") and os.environ.get("ALLOW_ALL_GPUS", "0") != "1" and os.path.exists("/data/nas07_new/PersonalData/phuocthien"):
        sys.exit("GPUs 0 and 7 belong to the user -- refusing")
    return a


def build_hard_nb(a, device, n=20000):
    import numpy as np, torch
    from det.dataset import letterbox, to_tensor
    from det.labels.store import load_rgb
    from det.model import load_v19
    net = load_v19(V19).to(device).eval()
    d = Path(a.root) / "train/non_bear"
    files = sorted(d.iterdir())

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(files)

        def __getitem__(self, i):
            return to_tensor(letterbox(load_rgb(files[i])))

    dl = torch.utils.data.DataLoader(_DS(), batch_size=64, num_workers=a.workers)
    ps = []
    with torch.no_grad():
        for x in dl:
            ps.append(torch.softmax(net(x.to(device))[0].float(), 1)[:, 1].cpu())
    p = torch.cat(ps).numpy()
    top = np.argsort(-p)[:n]
    HARD_NB.parent.mkdir(parents=True, exist_ok=True)
    HARD_NB.write_text("\n".join(str(files[i]) for i in top) + "\n")
    print(f"hard_nb: {len(top)} of {len(files)}, p(bear) at cut {p[top[-1]]:.4f} -> {HARD_NB}")


def main(argv=None):
    a = parse(argv if argv is not None else sys.argv[1:])
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if a.gpu == "cpu" else a.gpu
    import numpy as np, torch
    from det.dataset import DetDataset, DetSampler, collate
    from det.eval_det import evaluate
    from det.losses import total_loss
    from det.model import load_v19, save_det
    device = "cpu" if a.gpu == "cpu" else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available -- refusing to fall back to CPU")
    if a.build_hard_nb:
        return build_hard_nb(a, device)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    es = json.load(open(a.eval_set)) if Path(a.eval_set).exists() else None
    eval_sha = set(sum((es[s]["bear"] + es[s]["non_bear"] for s in es), [])) if es else set()
    tr = DetDataset(a.root, a.train_split, a.stride, train=True, store_path=a.store, cache_path=a.cache,
                    exclude_sha1=eval_sha, limit=a.limit_train, epochs=a.epochs)
    only = set(es["val"]["bear"] + es["val"]["non_bear"]) if es and a.val_split == "val" else None
    va = DetDataset(a.root, a.val_split, a.stride, train=False, store_path=a.store, cache_path=a.cache,
                    only_sha1=only, limit=a.limit_val)
    hard = set(HARD_NB.read_text().split("\n")) if HARD_NB.exists() else set()
    n_bear_batch = int(round(a.batch * 2 / 3))
    nb = a.n_batches or max(1, sum(m["cls"] == "bear" for m in tr.meta) // n_bear_batch)
    sampler = DetSampler(tr, batch=a.batch, n_batches=nb, hard=hard)
    dl = torch.utils.data.DataLoader(tr, batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                                     collate_fn=collate, pin_memory=device == "cuda",
                                     persistent_workers=a.workers > 0)
    print(f"train {len(tr)} (stale {tr.n_stale}, unlabelled {tr.n_unlabelled}), val {len(va)}, "
          f"hard_nb {len(hard)}, {nb} batches/epoch x {a.epochs}", flush=True)

    net = load_v19(V19, stride=a.stride, up=a.up, ch=a.ch).to(device)
    opt = torch.optim.AdamW(net.head.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * nb
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    best = (-1.0, -1.0)
    for ep in range(1, a.epochs + 1):
        tr.set_epoch(ep - 1); sampler.set_epoch(ep - 1)
        net.train()
        t0, losses, skipped = time.time(), [], 0
        for x, t, _ in dl:
            x = x.to(device, non_blocking=True)
            t = {k: v.to(device, non_blocking=True) for k, v in t.items()}
            bn = {k: v.clone() for k, v in net.head.state_dict().items() if "running" in k or "num_batches" in k}
            with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                feats = net.backbone_feats(x)
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                heat, box = net.head(*[f.float() for f in feats])
            loss, parts = total_loss((heat.float(), box.float()), t, a.stride)
            if not torch.isfinite(loss):
                net.head.load_state_dict({**net.head.state_dict(), **bn})   # forward already touched BN stats
                opt.zero_grad(set_to_none=True); skipped += 1
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(loss.item())
        r = evaluate(net, va, device, a.stride, batch=a.batch, workers=a.workers)
        rec = {"epoch": ep, "loss": float(np.mean(losses)) if losses else float("nan"), "skipped": skipped,
               "lr": opt.param_groups[0]["lr"], "min": round((time.time() - t0) / 60, 1),
               **{k: r[k] for k in ("ap50", "ap", "rec_fppi01", "iou", "ap50_domain", "ap50_size", "n_gt")}}
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"epoch {ep} loss {rec['loss']:.4f} skipped {skipped} | val AP50 {r['ap50']:.4f} "
              f"AP {r['ap']:.4f} rec@0.1FPPI {r['rec_fppi01']:.4f} medIoU {r['iou']['median']:.3f} "
              f"| {rec['min']} min", flush=True)
        save_det(net.cpu(), out / f"ep{ep}.pth", extra={"epoch": ep, "val": r})
        if (r["ap50"], r["ap"]) > best:
            best = (r["ap50"], r["ap"])
            save_det(net, out / "best.pth", extra={"epoch": ep, "val": r})
        net.to(device)


if __name__ == "__main__":
    main()
