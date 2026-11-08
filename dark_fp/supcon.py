import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=1280, hidden=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=1)

def supcon_loss(emb, labels, temperature=0.1):
    device = emb.device
    B = emb.shape[0]
    sim = (emb @ emb.t()) / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()   # stability
    labels = labels.view(-1, 1)
    self_mask = torch.eye(B, device=device)
    pos_mask = (labels == labels.t()).float() - self_mask       # same-class, exclude self
    exp_sim = torch.exp(sim) * (1 - self_mask)                  # exclude self from denom
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if valid.sum() == 0:
        return (emb.sum() * 0.0)
    mlpp = (pos_mask * log_prob).sum(dim=1)[valid] / pos_count[valid]
    return -mlpp.mean()

class BearNet(nn.Module):
    """Wraps a torchvision mobilenet_v2 (2-class classifier). Adds an optional
    projection head for SupCon. Keys stay features.*/classifier.*(/projection.*)
    so a plain mobilenet_v2 loads the weights strict=False for export."""
    def __init__(self, backbone, projection=None):
        super().__init__()
        self.features = backbone.features
        self.classifier = backbone.classifier
        self.projection = projection
    def forward(self, x):
        f = self.features(x)
        p = torch.flatten(F.adaptive_avg_pool2d(f, 1), 1)
        logits = self.classifier(p)
        if self.projection is None:
            return logits
        return logits, self.projection(p)
