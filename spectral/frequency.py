"""Radial frequency coordinates for rfft2 spectra.

For a real H x W image, ``rfft2`` returns ``[H, W//2 + 1]``. The normalized
radial coordinate is

    rho = sqrt(fy^2 + fx^2) / sqrt(0.5^2 + 0.5^2)

with ``fy = fftfreq(H)`` and ``fx = rfftfreq(W)`` in cycles/pixel, so
``rho in [0, 1]``.
"""
import math

import torch

DEFAULT_BANDS = {
    "low": (0.0, 0.2),
    "middle": (0.2, 0.5),
    "high": (0.5, 1.0),
}


def radial_frequency(height, width, device=None, dtype=torch.float32):
    fy = torch.fft.fftfreq(height, d=1.0, device=device, dtype=dtype)
    fx = torch.fft.rfftfreq(width, d=1.0, device=device, dtype=dtype)
    rho = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    rho = rho / math.sqrt(0.5 ** 2 + 0.5 ** 2)
    return rho


def band_masks(rho, bands=None, device=None):
    """Build disjoint masks covering [0, 1].

    Bands are interpreted as ``[lo, hi)`` except the last band, whose ``hi`` is
    inclusive, so the mask union covers the entire spectrum.
    """
    bands = DEFAULT_BANDS if bands is None else bands
    names = list(bands)
    masks = {}
    for i, name in enumerate(names):
        lo, hi = bands[name]
        if i == len(names) - 1:
            mask = (rho >= lo) & (rho <= hi)
        else:
            mask = (rho >= lo) & (rho < hi)
        if device is not None:
            mask = mask.to(device)
        masks[name] = mask
    union = torch.zeros_like(rho, dtype=torch.bool)
    for mask in masks.values():
        if (union & mask).any():
            raise ValueError("frequency bands overlap")
        union |= mask
    if not bool(union.all()):
        raise ValueError("frequency bands do not cover the spectrum")
    return masks
