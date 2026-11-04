"""Shared attack utilities: L-infinity/box projection and identity losses."""
import torch

from spectral.losses import cross_entropy_per_sample, margin_per_sample


def project_linf_box(delta, x, eps):
    """Project delta onto {||delta||inf <= eps, 0 <= x+delta <= 1}."""
    lower = torch.maximum(torch.full_like(x, -float(eps)), -x)
    upper = torch.minimum(torch.full_like(x, float(eps)), 1.0 - x)
    return torch.clamp(delta, lower, upper)


def gallery_logits(adapter, classifier, x, scale):
    features = adapter.extract_features(x)
    return float(scale) * classifier.scores(features)


def loss_per_sample(name, logits, labels):
    if name == "margin":
        return margin_per_sample(logits, labels)
    if name == "ce":
        return cross_entropy_per_sample(logits, labels)
    raise ValueError(f"unknown loss {name!r}")


def targeted_margin_per_sample(logits, targets):
    """Objective to **maximize** for a targeted attack: s_target - max_other."""
    b = logits.shape[0]
    rows = torch.arange(b, device=logits.device)
    s_target = logits[rows, targets]
    masked = logits.clone()
    masked[rows, targets] = -torch.inf
    return s_target - masked.max(dim=1).values


def untargeted_success(clean_pred, adv_pred, labels):
    return (clean_pred == labels) & (adv_pred != labels)


def targeted_success(clean_pred, adv_pred, labels, targets):
    return (clean_pred == labels) & (adv_pred == targets)
