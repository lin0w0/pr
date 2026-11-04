"""Radial Gaussian low-pass filtering in the rfft2 domain.

    G_sigma(rho) = exp(-rho^2 / (2 sigma^2))
    x_low = IFFT(FFT(x) * G_sigma)

Smaller sigma => narrower passband => stronger low-pass. The filter is applied
to the raw 128x128 ROI in [0, 1]; official preprocessing is applied afterwards
by the adapter.
"""
import math

import numpy as np
import torch

from spectral.fft_utils import apply_spectral_response
from spectral.frequency import radial_frequency


def gaussian_response(rho, sigma):
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return torch.exp(-(rho ** 2) / (2.0 * float(sigma) ** 2))


def gaussian_lowpass(x, sigma, rho=None, norm="ortho"):
    if rho is None:
        rho = radial_frequency(x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype)
    return apply_spectral_response(x, gaussian_response(rho, sigma), norm=norm)


def rho_at_gain(sigma, gain):
    """Radial frequency where G_sigma(rho) = gain (0 < gain < 1)."""
    if not (0.0 < gain < 1.0):
        raise ValueError("gain must be in (0, 1)")
    return float(sigma) * math.sqrt(-2.0 * math.log(gain))


def filter_response_table(sigmas, rho_points, device="cpu"):
    """Row-wise G_sigma(rho) table with correctly labelled cut-offs.

    - half-gain (G=0.5) is -6.0206 dB in amplitude, not -3 dB;
    - -3 dB amplitude corresponds to G=1/sqrt(2)=0.7071.

    ``cutoff_rho_half_gain`` is kept for backward compatibility with the
    earlier Phase 2 CSV; ``rho_minus3db`` is the true -3 dB cut-off.
    """
    rho = torch.tensor(rho_points, dtype=torch.float32, device=device)
    rows = []
    for sigma in sigmas:
        values = gaussian_response(rho, sigma)
        gain_db = [float(20.0 * math.log10(max(float(v), 1e-30))) for v in values]
        rows.append({
            "sigma": float(sigma),
            "rho_values": [float(v) for v in rho_points],
            "response": [float(v) for v in values],
            "gain_db": gain_db,
            "dc_gain": float(values[rho_points.index(0.0)]) if 0.0 in rho_points else 1.0,
            "rho_half_gain": rho_at_gain(sigma, 0.5),
            "rho_minus3db": rho_at_gain(sigma, 1.0 / math.sqrt(2.0)),
            "cutoff_rho_half_gain": rho_at_gain(sigma, 0.5),
        })
    return rows


def sigma_grid(values, device="cpu"):
    return [float(v) for v in values]


def response_curve(sigma, rho_max=1.0, n=201, device="cpu"):
    rho = torch.linspace(0.0, rho_max, n, device=device)
    return rho, gaussian_response(rho, sigma)
