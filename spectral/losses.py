"""Per-sample identity losses used for SGM analysis."""
import torch
import torch.nn.functional as F


def cross_entropy_per_sample(logits, labels):
    return F.cross_entropy(logits, labels, reduction="none")


def margin_per_sample(logits, labels):
    """L_margin = -(s_y - max_{j != y} s_j) = max_{j != y} s_j - s_y."""
    if logits.dim() != 2:
        raise ValueError("logits must be [B, C]")
    b, c = logits.shape
    if c < 2:
        raise ValueError("margin loss needs at least two classes")
    rows = torch.arange(b, device=logits.device)
    s_y = logits[rows, labels]
    masked = logits.clone()
    masked[rows, labels] = -torch.inf
    max_other = masked.max(dim=1).values
    return max_other - s_y


LOSSES = {
    "ce": cross_entropy_per_sample,
    "margin": margin_per_sample,
}
