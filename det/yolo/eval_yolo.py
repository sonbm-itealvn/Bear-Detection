"""FP32 evaluation of the YOLO candidate on the eval set, scored exactly like candidate (a)
(det.eval_det.score). Input = what the chip feeds: 640x384 black letterbox, (BGR-mean)/255,
through NovaicYolo. Also an image-level bear decision (any box >= thr).
python -m det.yolo.eval_yolo --pt det_runs/yolo_v8n/train/weights/best.pt --split val --gpu 2"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from det.dataset import to_tensor
from det.labels.store import load_rgb
from det.yolo.geometry import IN_W, IN_H, letterbox_params

ROOT = Path("/data/nas07_new/PersonalData/phuocthien")


def to_input(img):
    nw, nh, px, py = letterbox_params(*img.size)
    canvas = Image.new("RGB", (IN_W, IN_H), (0, 0, 0))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    return to_tensor(canvas)


class _Rows(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = load_rgb(self.rows[i]["path_hint"])
        return to_input(img), i, img.width, img.height


def image_level(preds_pos, preds_neg, thr=0.3):
    hit = lambda p: bool(len(p)) and float(np.max(p[:, 4])) >= thr
    tp = sum(hit(p) for p in preds_pos); fp = sum(hit(p) for p in preds_neg)
    return {"thr": thr, "recall": tp / max(len(preds_pos), 1), "precision": tp / max(tp + fp, 1),
            "fp": fp, "n_pos": len(preds_pos), "n_neg": len(preds_neg)}


def evaluate_rows(net, rows, device, batch=16, workers=0, thr_eval=0.05, keep_preds=False):
    from det.eval_det import score
    from det.yolo.decode import decode
    dl = torch.utils.data.DataLoader(_Rows(rows), batch_size=batch, num_workers=workers)
    preds, gts, doms, negp, posp_all, out_rows, scales = [], [], [], [], [], [], []
    net.eval()
    with torch.no_grad():
        for x, idx, ws, hs in dl:
            outs = [o.float().cpu().numpy() for o in net(x.to(device))]
            for b, i in enumerate(idx.tolist()):
                r = rows[i]
                p = decode([o[b] for o in outs], int(ws[b]), int(hs[b]), thr=thr_eval)
                if keep_preds:
                    out_rows.append({"path": r["path_hint"], "w": int(ws[b]), "h": int(hs[b]), "pred": p.tolist()})
                if r["cls"] != "bear":
                    negp.append(p)
                else:
                    posp_all.append(p)
                    if r["status"] not in ("masked", "excluded") and r["boxes"]:
                        preds.append(p)
                        gts.append(np.array([bx["xyxy"] for bx in r["boxes"]], float).reshape(-1, 4))
                        doms.append(r["domain"])
                        scales.append(letterbox_params(int(ws[b]), int(hs[b]))[0] / int(ws[b]))
    res = score(preds, gts, doms, negp, scales)
    res["img_level"] = image_level(posp_all, negp)
    if keep_preds:
        res["preds"] = out_rows
    return res


def main():
    from det.labels.store import LabelStore
    from det.yolo.export_onnx import NovaicYolo, load_trained
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--store", default=str(ROOT / "det_labels/labels_v1.jsonl"))
    a = ap.parse_args()
    if a.gpu in ("0", "7"):
        sys.exit("GPUs 0 and 7 belong to the user -- refusing")
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible")
    es = json.load(open(ROOT / "det_labels/eval_set.json"))[a.split]
    st = LabelStore(a.store)
    rows = [st.get(s) for s in es["bear"] + es["non_bear"] if st.get(s)]
    rows = [r for r in rows if r.get("status") != "excluded"]
    net = NovaicYolo(load_trained(a.pt)).to("cuda")
    r = evaluate_rows(net, rows, "cuda", workers=8, keep_preds=True)
    out = Path(a.pt).parent.parent
    with open(out / f"preds_{a.split}.jsonl", "w") as f:
        for row in r.pop("preds"):
            f.write(json.dumps(row) + "\n")
    json.dump(r, open(out / f"eval_{a.split}.json", "w"), indent=1)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
