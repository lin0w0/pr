"""Spectral Gradient Magnitude (SGM) with phase fixed.

The amplitude spectrum is made a differentiable variable while the phase is
detached. The reconstructed image is fed through the full official
preprocessing + frozen backbone, so the gradient includes the per-image
z-score.

    S(u,v) = |d L / d A(u,v)|
"""
import torch

from spectral.frequency import band_masks


def reconstruct_from_amplitude(amplitude, phase, size):
    spectrum = amplitude * torch.exp(1j * phase)
    return torch.fft.irfft2(spectrum, s=size, norm="ortho")


def amplitude_gradient(adapter, classifier, x, labels, loss_fn, logit_scale=30.0,
                       returns="grad", min_batch=4):
    """Compute per-sample gradients of an identity loss w.r.t. spectral amplitude.

    Small batches are padded to ``min_batch`` with duplicates of the first
    sample before the forward pass. CompNet/CCNet contain no cross-sample
    operations (no BatchNorm, per-image preprocessing), so padding does not
    change the gradient of any real sample; it only forces cuDNN to use the
    same batched kernel path as larger batches. Without this, batch=1 uses a
    single-image kernel and differs from batch>=4 by ~0.1% in FP32 (and by
    ~5.8% when TF32 is enabled).

    Args:
        adapter: frozen palmprint adapter (preprocessing included in the graph).
        classifier: object with ``scores(features) -> cosine scores``.
        x: raw images ``[B, 1, H, W]`` in ``[0, 1]``.
        labels: identity column indices ``[B]``.
        loss_fn: one of ``spectral.losses.LOSSES``.
        logit_scale: multiply cosine scores before the loss (1, 10, 30, ...).
        returns: "grad" or "all" (grad, per-sample losses, logits).
        min_batch: minimum internal batch used to stabilise the cuDNN kernel.

    Returns:
        grad ``[B, 1, H, W//2+1]`` (detached) and optionally losses/logits.
        Only the original ``B`` samples are returned.
    """
    x = x.to(adapter.device)
    labels = labels.to(adapter.device)
    if x.shape[0] == 0:
        raise ValueError("empty batch")
    n = x.shape[0]
    if n < min_batch:
        repeat = min_batch - n
        x = torch.cat([x, x[:1].repeat(repeat, 1, 1, 1)], dim=0)
        labels = torch.cat([labels, labels[:1].repeat(repeat)], dim=0)

    spectrum = torch.fft.rfft2(x, norm="ortho")
    amplitude = spectrum.abs()
    phase = spectrum.angle()

    amp_var = amplitude.detach().requires_grad_(True)
    x_recon = reconstruct_from_amplitude(amp_var, phase.detach(), x.shape[-2:])

    features = adapter.extract_features(x_recon)
    scores = classifier.scores(features)
    logits = float(logit_scale) * scores
    losses = loss_fn(logits, labels)
    total = losses.sum()
    grad, = torch.autograd.grad(total, amp_var)
    grad = grad.detach()[:n]
    if returns == "all":
        return grad, losses.detach()[:n], logits.detach()[:n]
    return grad


def gradient_metrics(grad, rho, bands=None):
    """Per-sample and aggregate gradient statistics.

    Args:
        grad: ``[B, 1, H, Wf]``
        rho: ``[H, Wf]`` radial frequency
        bands: dict name -> (lo, hi) or None for defaults

    Returns:
        dict of aggregate scalars plus ``per_sample`` arrays.
    """
    if grad.dim() != 4 or grad.shape[1] != 1:
        raise ValueError(f"grad must be [B,1,H,Wf], got {tuple(grad.shape)}")
    masks = band_masks(rho, bands=bands, device=grad.device)
    abs_grad = grad.abs()
    b = abs_grad.shape[0]

    per_sample_l1 = abs_grad.sum(dim=(1, 2, 3))
    per_sample_l2 = torch.sqrt((abs_grad ** 2).sum(dim=(1, 2, 3)))
    per_sample_max = abs_grad.amax(dim=(1, 2, 3))

    band_energy = {}
    for name, mask in masks.items():
        # mask [H, Wf] broadcasts over [B,1,H,Wf]
        band_energy[name] = (abs_grad * mask).sum(dim=(1, 2, 3))
    total_energy = sum(band_energy.values())
    total_safe = total_energy.clamp_min(1e-30)
    band_ratios = {name: (e / total_safe) for name, e in band_energy.items()}

    agg_total = total_energy.sum()
    aggregate_ratios = {
        name: float((e.sum() / agg_total.clamp_min(1e-30)).item())
        for name, e in band_energy.items()
    }
    near_zero_1e12 = float((per_sample_l1 < 1e-12).float().mean().item())
    near_zero_1e10 = float((per_sample_l1 < 1e-10).float().mean().item())

    return {
        "n_samples": int(b),
        "grad_l1_mean": float(per_sample_l1.mean().item()),
        "grad_l1_sum": float(per_sample_l1.sum().item()),
        "grad_l1_std": float(per_sample_l1.std(unbiased=False).item()) if b > 1 else 0.0,
        "grad_l2_mean": float(per_sample_l2.mean().item()),
        "grad_max": float(per_sample_max.max().item()),
        "grad_max_mean": float(per_sample_max.mean().item()),
        "finite_fraction": float(torch.isfinite(grad).float().mean().item()),
        "near_zero_l1_lt_1e-12_fraction": near_zero_1e12,
        "near_zero_l1_lt_1e-10_fraction": near_zero_1e10,
        "band_ratio_aggregate": aggregate_ratios,
        "band_ratio_mean": {name: float(r.mean().item()) for name, r in band_ratios.items()},
        "band_ratio_std": {name: float(r.std(unbiased=False).item()) if b > 1 else 0.0
                           for name, r in band_ratios.items()},
        "per_sample": {
            "l1": per_sample_l1.detach().cpu(),
            "l2": per_sample_l2.detach().cpu(),
            "max": per_sample_max.detach().cpu(),
            "band_ratio": {name: r.detach().cpu() for name, r in band_ratios.items()},
        },
    }
