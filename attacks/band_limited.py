"""Band-limited PGD with Dykstra projection onto band ∩ box."""
import torch

from attacks.base import (
    gallery_logits,
    loss_per_sample,
    targeted_margin_per_sample,
)
from spectral.band_projection import (
    constraint_metrics,
    dykstra_project,
    dykstra_project_band_l2_box,
    project_l2,
)


def _init_delta(x, eps, random_start, seed):
    if random_start:
        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(seed))
        delta = (torch.rand(x.shape, generator=gen, device=x.device, dtype=x.dtype) * 2 - 1) * eps
    else:
        delta = torch.zeros_like(x)
    return delta


def band_limited_feature_shift(adapter, x, band_mask, eps, alpha, steps,
                               random_start=True, seed=0, dykstra_iters=20,
                               dykstra_tol=1e-6):
    """Maximize 1 - cos(f(x), f(x+delta)) under the band/box constraints."""
    x = x.to(adapter.device)
    delta = _init_delta(x, eps, random_start, seed)
    delta, _ = dykstra_project(delta, x, eps, band_mask, n_iters=dykstra_iters,
                               tol=dykstra_tol, return_metrics=False)
    with torch.no_grad():
        base = adapter.extract_features(x)
    for _ in range(int(steps)):
        x_adv = (x + delta).detach().requires_grad_(True)
        feat = adapter.extract_features(x_adv)
        loss = (1.0 - (feat * base).sum(dim=1)).sum()
        grad, = torch.autograd.grad(loss, x_adv)
        delta = delta + float(alpha) * grad.sign()
        delta, _ = dykstra_project(delta, x, eps, band_mask, n_iters=dykstra_iters,
                                   tol=dykstra_tol, return_metrics=False)
    delta, metrics = dykstra_project(delta, x, eps, band_mask,
                                     n_iters=dykstra_iters, tol=dykstra_tol,
                                     return_metrics=True)
    x_adv = (x + delta).detach()
    return x_adv, delta.detach(), metrics


def band_limited_margin_l2(adapter, classifier, x, labels, band_mask, radius,
                           alpha, steps, objective="untargeted", targets=None,
                           scale=10.0, random_start=False, seed=0,
                           dykstra_iters=30, dykstra_tol=1e-6):
    """Band-limited attack under a per-sample L2 budget (no L-infinity cap).

    Projection is Dykstra onto band ∩ {||delta||2 <= radius} ∩ pixel box.
    """
    x = x.to(adapter.device)
    labels = labels.to(adapter.device)
    if objective == "targeted":
        if targets is None:
            raise ValueError("targeted attack requires targets")
        targets = targets.to(adapter.device)
    if random_start:
        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(seed))
        delta = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
        delta = project_l2(delta, radius)
    else:
        delta = torch.zeros_like(x)
    delta, _ = dykstra_project_band_l2_box(
        delta, x, radius, band_mask, n_iters=dykstra_iters,
        tol=dykstra_tol, return_metrics=False,
    )
    for _ in range(int(steps)):
        x_adv = (x + delta).detach().requires_grad_(True)
        logits = gallery_logits(adapter, classifier, x_adv, scale)
        if objective == "untargeted":
            loss = loss_per_sample("margin", logits, labels).sum()
        elif objective == "targeted":
            loss = targeted_margin_per_sample(logits, targets).sum()
        else:
            raise ValueError(f"unknown objective {objective!r}")
        grad, = torch.autograd.grad(loss, x_adv)
        delta = delta + float(alpha) * grad.sign()
        delta, _ = dykstra_project_band_l2_box(
            delta, x, radius, band_mask, n_iters=dykstra_iters,
            tol=dykstra_tol, return_metrics=False,
        )
    delta, metrics = dykstra_project_band_l2_box(
        delta, x, radius, band_mask, n_iters=dykstra_iters,
        tol=dykstra_tol, return_metrics=True,
    )
    x_adv = (x + delta).detach()
    return x_adv, delta.detach(), metrics


def band_limited_margin(adapter, classifier, x, labels, band_mask, eps, alpha,
                        steps, objective="untargeted", targets=None,
                        scale=10.0, random_start=True, seed=0,
                        dykstra_iters=20, dykstra_tol=1e-6):
    """Band-limited untargeted or targeted margin attack.

    ``objective="untargeted"`` maximizes max_{j!=y} s_j - s_y.
    ``objective="targeted"`` maximizes s_target - max_{j!=target} s_j.
    """
    x = x.to(adapter.device)
    labels = labels.to(adapter.device)
    if objective == "targeted":
        if targets is None:
            raise ValueError("targeted attack requires targets")
        targets = targets.to(adapter.device)
    delta = _init_delta(x, eps, random_start, seed)
    delta, _ = dykstra_project(delta, x, eps, band_mask, n_iters=dykstra_iters,
                               tol=dykstra_tol, return_metrics=False)
    for _ in range(int(steps)):
        x_adv = (x + delta).detach().requires_grad_(True)
        logits = gallery_logits(adapter, classifier, x_adv, scale)
        if objective == "untargeted":
            loss = loss_per_sample("margin", logits, labels).sum()
        elif objective == "targeted":
            loss = targeted_margin_per_sample(logits, targets).sum()
        else:
            raise ValueError(f"unknown objective {objective!r}")
        grad, = torch.autograd.grad(loss, x_adv)
        delta = delta + float(alpha) * grad.sign()
        delta, _ = dykstra_project(delta, x, eps, band_mask, n_iters=dykstra_iters,
                                   tol=dykstra_tol, return_metrics=False)
    delta, metrics = dykstra_project(delta, x, eps, band_mask,
                                     n_iters=dykstra_iters, tol=dykstra_tol,
                                     return_metrics=True)
    x_adv = (x + delta).detach()
    return x_adv, delta.detach(), metrics
