"""Band-limited perturbations and Dykstra projection onto the constraint set.

Constraint set for a band ``k``:

    A = { delta : FFT(delta) * (1 - M_k) = 0 }          (band-limited)
    B = { delta : ||delta||_inf <= eps, 0 <= x + delta <= 1 }  (box)

Both are convex. Dykstra's alternating projection returns a point in (or very
close to) A ∩ B. Projections are exact; the final residuals are reported so a
run is never silently claimed to be strictly band-limited.
"""
import torch

from spectral.fft_utils import irfft2, rfft2


def band_mask(rho, band):
    """Boolean mask for an inclusive/half-open radial band.

    ``band = (lo, hi)``; ``hi`` is inclusive when it equals 1.0 (the last band).
    """
    lo, hi = float(band[0]), float(band[1])
    if hi >= 1.0:
        return (rho >= lo) & (rho <= hi)
    return (rho >= lo) & (rho < hi)


def project_band(delta, mask, norm="ortho"):
    spec = rfft2(delta, norm=norm)
    return irfft2(spec * mask.unsqueeze(0).unsqueeze(0), delta.shape[-2:], norm=norm)


def project_box(delta, x, eps):
    # delta >= max(-eps, -x) and delta <= min(eps, 1 - x)
    lower = torch.maximum(torch.full_like(x, -float(eps)), -x)
    upper = torch.minimum(torch.full_like(x, float(eps)), 1.0 - x)
    return torch.clamp(delta, lower, upper)


def project_pixel_box(delta, x):
    """Projection onto {delta : 0 <= x + delta <= 1} (no L-infinity bound)."""
    return torch.clamp(delta, -x, 1.0 - x)


def project_l2(delta, radius):
    """Projection onto the per-sample L2 ball ||delta||_2 <= radius."""
    norms = delta.reshape(delta.shape[0], -1).norm(dim=1)
    scale = torch.clamp(float(radius) / (norms + 1e-30), max=1.0)
    return delta * scale.view(-1, 1, 1, 1)


def constraint_metrics_l2(delta, x, radius, mask, norm="ortho"):
    spec = rfft2(delta, norm=norm)
    total = spec.abs().pow(2).sum(dim=(-2, -1))
    outside = (spec * (~mask).unsqueeze(0).unsqueeze(1)).abs().pow(2).sum(dim=(-2, -1))
    band_outside = outside / (total + 1e-30)
    x_adv = x + delta
    box_violation = torch.clamp(x_adv - 1.0, min=0).amax(dim=(1, 2, 3)) + \
        torch.clamp(-x_adv, min=0).amax(dim=(1, 2, 3))
    band_residual = (delta - project_band(delta, mask, norm=norm)).abs().amax(dim=(1, 2, 3))
    l2 = delta.reshape(delta.shape[0], -1).norm(dim=1)
    l2_excess = (l2 - float(radius)).clamp_min(0.0)
    linf = delta.abs().amax(dim=(1, 2, 3))
    return {
        "band_outside_energy_ratio": band_outside,
        "l2": l2,
        "l2_excess": l2_excess,
        "linf": linf,
        "box_violation": box_violation,
        "band_residual_linf": band_residual,
    }


def dykstra_project_band_l2_box(delta, x, radius, mask, n_iters=30, tol=1e-6,
                                norm="ortho", return_metrics=True):
    """Dykstra projection onto band ∩ L2 ball ∩ pixel box (three convex sets)."""
    if delta.shape != x.shape:
        raise ValueError("delta and x must have the same shape")
    p = torch.zeros_like(delta)
    q = torch.zeros_like(delta)
    r = torch.zeros_like(delta)
    current = delta
    converged = torch.zeros(delta.shape[0], dtype=torch.bool, device=delta.device)
    iters_used = torch.zeros(delta.shape[0], dtype=torch.int64, device=delta.device)
    for i in range(int(n_iters)):
        y = project_band(current + p, mask, norm=norm)
        p = current + p - y
        z = project_l2(y + q, radius)
        q = y + q - z
        new = project_pixel_box(z + r, x)
        r = z + r - new
        change = (new - current).abs().amax(dim=(1, 2, 3))
        newly = (~converged) & (change <= tol)
        converged = converged | newly
        iters_used = torch.where(newly, torch.full_like(iters_used, i + 1), iters_used)
        current = new
        if bool(converged.all()):
            break
    iters_used = torch.where(converged, iters_used,
                             torch.full_like(iters_used, int(n_iters)))
    if not return_metrics:
        return current, None
    metrics = constraint_metrics_l2(current, x, radius, mask, norm=norm)
    metrics["iterations"] = iters_used
    metrics["converged"] = converged
    return current, metrics


def constraint_metrics(delta, x, eps, mask, norm="ortho"):
    spec = rfft2(delta, norm=norm)
    total = spec.abs().pow(2).sum(dim=(-2, -1))
    outside = (spec * (~mask).unsqueeze(0).unsqueeze(0)).abs().pow(2).sum(dim=(-2, -1))
    band_outside = outside / (total + 1e-30)
    x_adv = x + delta
    box_violation = torch.clamp(x_adv - 1.0, min=0).amax(dim=(1, 2, 3)) + \
        torch.clamp(-x_adv, min=0).amax(dim=(1, 2, 3))
    band_residual = (delta - project_band(delta, mask, norm=norm)).abs().amax(dim=(1, 2, 3))
    linf = delta.abs().amax(dim=(1, 2, 3))
    return {
        "band_outside_energy_ratio": band_outside,
        "linf": linf,
        "linf_excess": (linf - eps).clamp_min(0.0),
        "box_violation": box_violation,
        "band_residual_linf": band_residual,
    }


def dykstra_project(delta, x, eps, mask, n_iters=20, tol=1e-6, norm="ortho",
                    return_metrics=True):
    """Dykstra projection onto (band-limited ∩ box).

    Returns the projected delta and per-sample diagnostics. The final iterate
    ends with the box projection, so box constraints are exact; the band
    residual is reported and may be non-zero if ``n_iters`` is too small.
    """
    if delta.shape != x.shape:
        raise ValueError("delta and x must have the same shape")
    p = torch.zeros_like(delta)
    q = torch.zeros_like(delta)
    current = delta
    converged = torch.zeros(delta.shape[0], dtype=torch.bool, device=delta.device)
    iters_used = torch.zeros(delta.shape[0], dtype=torch.int64, device=delta.device)
    for i in range(int(n_iters)):
        y = project_band(current + p, mask, norm=norm)
        p = current + p - y
        new = project_box(y + q, x, eps)
        q = y + q - new
        change = (new - current).abs().amax(dim=(1, 2, 3))
        newly = (~converged) & (change <= tol)
        converged = converged | newly
        iters_used = torch.where(
            newly, torch.full_like(iters_used, i + 1), iters_used
        )
        current = new
        if bool(converged.all()):
            break
    iters_used = torch.where(
        converged, iters_used, torch.full_like(iters_used, int(n_iters))
    )
    if not return_metrics:
        return current, None
    metrics = constraint_metrics(current, x, eps, mask, norm=norm)
    metrics["iterations"] = iters_used
    metrics["converged"] = converged
    return current, metrics
