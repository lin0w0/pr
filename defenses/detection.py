"""Spectral-consistency detection scores used by Phase 3 final validation."""
import torch
import torch.nn.functional as F

from spectral.frequency import radial_frequency
from spectral.gaussian_filter import gaussian_lowpass


def consistency_matrix(adapter, x, sigmas, differentiable=False):
    """Return ``[N, S]`` consistency scores C_sigma(x)."""
    device = adapter.device
    x = x.to(device)
    rho = radial_frequency(x.shape[-2], x.shape[-1], device=device, dtype=x.dtype)
    if differentiable:
        f0 = F.normalize(adapter.extract_features(x), dim=1)
        out = []
        for sigma in sigmas:
            low = gaussian_lowpass(x, sigma, rho=rho)
            fl = F.normalize(adapter.extract_features(low), dim=1)
            out.append((f0 * fl).sum(dim=1))
        return torch.stack(out, dim=1)
    with torch.no_grad():
        f0 = F.normalize(adapter.extract_features(x), dim=1)
        out = []
        for sigma in sigmas:
            low = gaussian_lowpass(x, sigma, rho=rho)
            fl = F.normalize(adapter.extract_features(low), dim=1)
            out.append((f0 * fl).sum(dim=1))
        return torch.stack(out, dim=1)


def combine_scores(consistency, method="mean"):
    arr = consistency
    if method == "mean":
        return arr.mean(dim=1)
    if method == "min":
        return arr.min(dim=1).values
    raise ValueError(f"unknown multi-scale method {method!r}")


def detector_scores(adapter, x, sigmas, method="mean", differentiable=False):
    """Single-scale (one sigma) or multi-scale score; lower = more adversarial."""
    mat = consistency_matrix(adapter, x, sigmas, differentiable=differentiable)
    return combine_scores(mat, method)
