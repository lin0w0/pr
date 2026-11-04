"""Feature spectral consistency C(x, sigma) = cos(f(x), f(LPF_sigma(x)))."""
import torch
import torch.nn.functional as F

from spectral.gaussian_filter import gaussian_lowpass
from spectral.frequency import radial_frequency


@torch.no_grad()
def spectral_consistency(adapter, x, sigmas, rho=None, batch_size=None):
    """Per-sample consistency for each sigma.

    Args:
        adapter: frozen palmprint adapter.
        x: raw images ``[B, 1, H, W]`` in ``[0, 1]``.
        sigmas: iterable of Gaussian sigmas.
        rho: optional precomputed radial grid.

    Returns:
        ``[B, len(sigmas)]`` cosine similarities.
    """
    x = x.to(adapter.device)
    if rho is None:
        rho = radial_frequency(x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype)
    base = F.normalize(adapter.extract_features(x), dim=1)
    out = []
    for sigma in sigmas:
        low = gaussian_lowpass(x, sigma, rho=rho)
        feat = F.normalize(adapter.extract_features(low), dim=1)
        out.append((base * feat).sum(dim=1))
    return torch.stack(out, dim=1)
