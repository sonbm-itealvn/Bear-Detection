import torch
from det.losses import focal_loss, giou_loss, hinge

def test_focal_lower_when_correct():
    tgt = torch.zeros(1, 1, 4, 4); tgt[0, 0, 1, 1] = 1.0; w = torch.ones_like(tgt)
    good = torch.full_like(tgt, -6.0); good[0, 0, 1, 1] = 6.0
    bad = torch.full_like(tgt, 6.0); bad[0, 0, 1, 1] = -6.0
    assert focal_loss(good, tgt, w) < focal_loss(bad, tgt, w)

def test_focal_ignores_zero_weight():
    tgt = torch.zeros(1, 1, 4, 4); w = torch.zeros_like(tgt)
    assert focal_loss(torch.full_like(tgt, 6.0), tgt, w).item() == 0.0

def test_giou_zero_for_exact():
    tb = torch.zeros(1, 4, 2, 2); wt = torch.zeros(1, 1, 2, 2); wt[0, 0, 0, 0] = 1
    assert giou_loss(tb.clone(), tb, wt, 16).item() < 1e-6

def test_hinge():
    assert hinge(torch.tensor([7.0, -7.0])).item() == 0
    assert hinge(torch.tensor([10.0])).item() > 0
