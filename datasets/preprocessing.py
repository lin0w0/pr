"""Differentiable replica of the official CompNet/CCNet inference preprocessing.

Official chain (both repositories, ``MyDataset(train=False)``):

    Image.open(path).convert("L")
    -> torchvision.transforms.Resize(128)      # no-op: Tongji ROI is 128x128
    -> torchvision.transforms.ToTensor()       # uint8 / 255 -> [0, 1]
    -> NormSingleROI                           # per-image z-score over pixels > 0

``Resize(int)`` resizes the *shorter* edge and preserves aspect ratio. The Tongji
ROI images are already 128x128, so it is an identity. We hard-fail on any other
shape rather than silently applying a different geometric transform.
"""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from utils.paths import resolve_path


def load_raw_image(path, input_size=128):
    """Load a ROI image as a raw ``[1, H, W]`` float32 tensor in ``[0, 1]``.

    No normalization is applied here: this is the pixel space in which attack
    budgets are defined.
    """
    abs_path = resolve_path(path)
    with Image.open(abs_path) as im:
        im = im.convert("L")
        if im.size != (input_size, input_size):
            raise ValueError(
                f"{abs_path} has size {im.size}, expected "
                f"({input_size}, {input_size}). The official pipeline uses "
                f"Resize(int) which preserves aspect ratio; only square inputs "
                f"of the expected size are supported."
            )
        arr = np.array(im, dtype=np.uint8, copy=True)
    return torch.from_numpy(arr).float().div_(255.0).unsqueeze(0)


def official_normalize(x, eps=1e-6):
    """Differentiable ``NormSingleROI`` for a batch.

    Args:
        x: ``[B, 1, H, W]`` raw image tensor in ``[0, 1]``.
        eps: same epsilon as the official implementation.

    Returns:
        ``[B, 1, H, W]`` tensor. Pixels equal to zero are left unchanged, exactly
        like ``NormSingleROI`` (``tensor[idx] = ...`` only touches the mask).
    """
    if x.dim() != 4:
        raise ValueError(f"expected [B, C, H, W], got {tuple(x.shape)}")
    if x.shape[1] != 1:
        raise ValueError(f"only single-channel ROI is supported, got C={x.shape[1]}")
    if not torch.isfinite(x).all():
        raise ValueError("input contains NaN/Inf")

    B, C, H, W = x.shape
    flat = x.reshape(B, C, H * W)
    idx = flat > 0
    count = idx.sum(dim=(1, 2))
    if (count < 2).any():
        raise ValueError(
            "at least two non-zero pixels per sample are required for the "
            "official unbiased std; found a sample with fewer"
        )

    mean = (flat * idx).sum(dim=(1, 2)) / count
    centered = flat - mean[:, None, None]
    var = ((centered ** 2) * idx).sum(dim=(1, 2)) / (count - 1)
    std = torch.sqrt(var.clamp_min(0.0))

    normalized = centered / (std[:, None, None] + eps)
    out = torch.where(idx, normalized, flat)
    return out.reshape(B, C, H, W)


def fixed_mask_normalize(x, eps=1e-6):
    """Diagnostic variant: z-score over the **fixed full 128x128 ROI**.

    The official ``NormSingleROI`` uses a dynamic ``pixel > 0`` mask. On clean
    Tongji ROI every pixel is positive, so both are identical; they can differ
    only after filtering/clipping creates non-positive values. This function is
    for the mask-sensitivity audit and is NOT used by the main pipeline.
    """
    if x.dim() != 4 or x.shape[1] != 1:
        raise ValueError("expected [B, 1, H, W]")
    B, C, H, W = x.shape
    flat = x.reshape(B, C, H * W)
    mean = flat.mean(dim=(1, 2))
    var = flat.var(dim=(1, 2), unbiased=True)
    out = (flat - mean[:, None, None]) / (torch.sqrt(var.clamp_min(0.0))[:, None, None] + eps)
    return out.reshape(B, C, H, W)


class OfficialROIPreprocess(nn.Module):
    """Module form of :func:`official_normalize` with an input-size check."""

    def __init__(self, input_size=128, eps=1e-6):
        super().__init__()
        self.input_size = int(input_size)
        self.eps = float(eps)

    def forward(self, x):
        if x.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(
                f"expected {self.input_size}x{self.input_size} input, got "
                f"{tuple(x.shape[-2:])}"
            )
        return official_normalize(x, eps=self.eps)

    def extra_repr(self):
        return f"input_size={self.input_size}, eps={self.eps}"
