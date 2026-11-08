"""FP32 evaluation of a det checkpoint on the (reviewed) eval set. Decoding uses the
reference decoder with gate=False, so box quality is measured independently of the
classifier. Masked bear images (no trusted boxes) are skipped; non_bear images count
as negatives (every box on them is a false positive)."""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/nas07_new/PersonalData/phuocthien")
EVAL_THR = 0.05     # AP needs the whole PR curve; 0.3 is only the deployment default
SIZE_BUCKETS = (("small", 0, 32), ("medium", 32, 96), ("large", 96, 1e9))


def _size(b):
    return float(np.sqrt(max(b[2] - b[0], 0) * max(b[3] - b[1], 0)))


def score(preds, gts, doms, negp, scales=None):
    """Metrics from per-image predictions [K,5] / ground truth [N,4] (original pixels),
    their domains, and predictions on negative images. scales: per positive image, the
    letterbox scale (network px per original px); size buckets are in network pixels."""
    from det.labels.fusion import iou
    from det.metrics import ap_at, ap_range, recall_at_fppi, iou_stats
    ious = []
    for p, g in zip(preds, gts):
        if len(p) and len(g):
            ious += list(iou(g, p[:, :4]).max(1))
    r = {"n_img": len(preds), "n_gt": int(sum(len(g) for g in gts)), "n_neg": len(negp),
         "ap50": ap_at(preds + negp, gts + [np.zeros((0, 4))] * len(negp), 0.5),
         "ap": ap_range(preds + negp, gts + [np.zeros((0, 4))] * len(negp)),
         "rec_fppi01": recall_at_fppi(preds, gts, negp, 0.1),
         "iou": iou_stats(ious), "ap50_domain": {}, "ap50_size": {}}
    for d in sorted(set(doms)):
        k = [i for i, x in enumerate(doms) if x == d]
        r["ap50_domain"][d] = ap_at([preds[i] for i in k], [gts[i] for i in k], 0.5)
    sc = scales if scales is not None else [1.0] * len(gts)
    for name, lo, hi in SIZE_BUCKETS:              # COCO-style: other sizes are ignored, not dropped
        gi = [np.array([not (lo <= _size(b) * f < hi) for b in g], bool) for g, f in zip(gts, sc)]
        if sum(int((~x).sum()) for x in gi):
            pi = [np.array([not (lo <= _size(q) * f < hi) for q in p], bool) for p, f in zip(preds, sc)]
            r["ap50_size"][name] = ap_at(preds, gts, 0.5, gt_ignore=gi, pred_ignore=pi)
    return r


def evaluate(net, ds, device, stride, batch=32, workers=0, keep_preds=False):
    import torch
    from det.dataset import collate
    from det.decode import decode
    from det.geometry import lb_to_orig, letterbox_params
    from det.labels.fusion import iou
    from det.metrics import ap_at, ap_range, recall_at_fppi, iou_stats
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=collate)
    preds, gts, doms, negp, rows, scales = [], [], [], [], [], []
    net.eval()
    with torch.no_grad():
        for x, _, metas in dl:
            logits, heat, box = net(x.to(device))
            for b, m in enumerate(metas):
                p = decode(logits[b].float().cpu().numpy(), heat[b].float().cpu().numpy(),
                           box[b].float().cpu().numpy(), m["w"], m["h"], stride=stride, thr=EVAL_THR, gate=False)
                if keep_preds:
                    rows.append({"path": m["path"], "w": m["w"], "h": m["h"], "pred": p.tolist()})
                if m["path"].split("/")[-2] == "non_bear":
                    negp.append(p)
                elif not m["masked"]:
                    preds.append(p)
                    gts.append(lb_to_orig(m["boxes_lb"], m["w"], m["h"], clip=True))
                    doms.append(m["domain"])
                    scales.append(letterbox_params(m["w"], m["h"])[0] / m["w"])
    r = score(preds, gts, doms, negp, scales)
    if keep_preds:
        r["preds"] = rows
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--eval-set", default=str(ROOT / "det_labels/eval_set.json"))
    ap.add_argument("--store", default=str(ROOT / "det_labels/labels_v1.jsonl"))
    a = ap.parse_args()
    if a.gpu in ("0", "7"):
        sys.exit("GPUs 0 and 7 belong to the user -- refusing")
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    import torch
    from det.dataset import DetDataset
    from det.model import load_det
    es = json.load(open(a.eval_set))[a.split]
    ds = DetDataset(ROOT / "bear_dataset_v_final_final", a.split, 0, train=False, store_path=a.store,
                    cache_path=ROOT / "det_labels/sha1_cache.json", only_sha1=set(es["bear"] + es["non_bear"]))
    net = load_det(a.ckpt)
    ds.stride = net.head_cfg["stride"]
    dev = "cuda" if torch.cuda.is_available() else sys.exit("no CUDA device visible")
    r = evaluate(net.to(dev), ds, dev, ds.stride, workers=8, keep_preds=True)
    out = Path(a.ckpt).parent
    with open(out / f"preds_{a.split}.jsonl", "w") as f:
        for row in r.pop("preds"):
            f.write(json.dumps(row) + "\n")
    json.dump(r, open(out / f"eval_{a.split}.json", "w"), indent=1)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
