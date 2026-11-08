"""Detection losses (spec §3): penalty-reduced focal (a=2, b=4), GIoU x5 + L1 x0.5 on the
log-ltrb box map, and a logit hinge that keeps |heat logit| <= 8 for 16-bit fixed point."""
import torch
import torch.nn.functional as F


def focal_loss(logit, tgt, w, a=2.0, b=4.0):
    p = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)
    pos = (tgt >= 1.0).float() * w
    neg = (tgt < 1.0).float() * w
    lp = -((1 - p) ** a) * torch.log(p) * pos
    ln = -((1 - tgt) ** b) * (p ** a) * torch.log(1 - p) * neg
    return (lp.sum() + ln.sum()) / pos.sum().clamp(min=1.0)


def ltrb_to_xyxy(ltrb_log, stride):
    _, _, H, W = ltrb_log.shape
    ys = (torch.arange(H, device=ltrb_log.device) + 0.5) * stride
    xs = (torch.arange(W, device=ltrb_log.device) + 0.5) * stride
    cy, cx = torch.meshgrid(ys, xs, indexing="ij")
    d = torch.exp(ltrb_log.clamp(-6, 6)) * stride
    return torch.stack([cx - d[:, 0], cy - d[:, 1], cx + d[:, 2], cy + d[:, 3]], 1)


def giou_loss(pred, tgt, wt, stride):
    p = ltrb_to_xyxy(pred, stride); t = ltrb_to_xyxy(tgt, stride)
    ix1 = torch.max(p[:, 0], t[:, 0]); iy1 = torch.max(p[:, 1], t[:, 1])
    ix2 = torch.min(p[:, 2], t[:, 2]); iy2 = torch.min(p[:, 3], t[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    ap = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1]); at = (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
    union = ap + at - inter
    c = (torch.max(p[:, 2], t[:, 2]) - torch.min(p[:, 0], t[:, 0])) * \
        (torch.max(p[:, 3], t[:, 3]) - torch.min(p[:, 1], t[:, 1]))
    giou = inter / union.clamp(min=1e-6) - (c - union) / c.clamp(min=1e-6)
    return ((1 - giou) * wt[:, 0]).sum() / max(float(wt.shape[0]), 1.0)


def l1_loss(pred, tgt, wt):
    return ((pred - tgt).abs().sum(1) * wt[:, 0]).sum() / max(float(wt.shape[0]), 1.0)


def hinge(logit, z0=8.0):
    return (F.relu(logit.abs() - z0) ** 2).mean() * 0.01


def total_loss(outs, tgts, stride):
    heat, box = outs
    lf = focal_loss(heat, tgts["heat"], tgts["heat_w"])
    lg = giou_loss(box, tgts["box"], tgts["box_w"], stride)
    l1 = l1_loss(box, tgts["box"], tgts["box_w"])
    lh = hinge(heat)
    loss = lf + 5.0 * lg + 0.5 * l1 + lh
    return loss, {"focal": lf.item(), "giou": lg.item(), "l1": l1.item(), "hinge": lh.item()}
