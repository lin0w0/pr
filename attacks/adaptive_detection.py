"""Adaptive PGD against spectral-consistency detection.

Maximize  L_margin(x_adv) - lambda * D(x_adv)
where lower D = more adversarial for the detector, so subtracting lambda*D
pushes the score upward (evasion), while L_margin pushes attack success.
"""
import torch

from attacks.base import project_linf_box
from defenses.detection import detector_scores
from spectral.losses import margin_per_sample


def adaptive_pgd(adapter, classifier, x, labels, init_adv, sigmas, method,
                 lam, eps, alpha, steps, scale=10.0):
    x = x.to(adapter.device)
    labels = labels.to(adapter.device)
    delta = (init_adv.to(adapter.device) - x).clamp(-eps, eps)
    delta = project_linf_box(delta, x, eps)
    for _ in range(int(steps)):
        xa = (x + delta).detach().requires_grad_(True)
        feats = torch.nn.functional.normalize(adapter.extract_features(xa), dim=1)
        logits = float(scale) * (feats @ classifier.prototypes.T)
        margin = margin_per_sample(logits, labels)
        det = detector_scores(adapter, xa, sigmas, method, differentiable=True)
        loss = (margin - float(lam) * det).sum()
        grad, = torch.autograd.grad(loss, xa)
        delta = project_linf_box(delta + float(alpha) * grad.sign(), x, eps)
    return (x + delta).detach()
