"""Real-valued 2D FFT helpers used by all Phase 2 experiments."""
import torch


def rfft2(x, norm="ortho"):
    """Real FFT. Returns ``[B, C, H, W//2+1]`` complex."""
    return torch.fft.rfft2(x, norm=norm)


def irfft2(spectrum, size, norm="ortho"):
    """Inverse real FFT back to ``[B, C, H, W]``."""
    return torch.fft.irfft2(spectrum, s=size, norm=norm)


def amplitude_phase(x, norm="ortho"):
    spectrum = rfft2(x, norm=norm)
    return spectrum.abs(), spectrum.angle()


def amplitude_spectrum(x, norm="ortho"):
    return rfft2(x, norm=norm).abs()


def apply_spectral_response(x, response, norm="ortho"):
    """Multiply the rfft2 spectrum by a real response ``[H, W//2+1]``."""
    if response.dim() == 2:
        response = response.unsqueeze(0).unsqueeze(0)
    spec = rfft2(x, norm=norm)
    return irfft2(spec * response, x.shape[-2:], norm=norm)


def spectral_energy(spectrum):
    """Parseval energy of an rfft2 spectrum (orthonormal convention)."""
    return spectrum.abs().pow(2).sum(dim=(-2, -1))


def band_energy_ratio(delta, mask, norm="ortho", eps=1e-30):
    """Fraction of ``delta`` energy outside a radial band mask."""
    spec = rfft2(delta, norm=norm)
    total = spectral_energy(spec)
    outside = spectral_energy(spec * (~mask))
    return outside / (total + eps)


def one_sided_weights(height, width, device=None, dtype=torch.float32):
    """Multiplicity of each rfft2 bin for a real image.

    Interior frequency columns represent a conjugate pair (weight 2);
    the fx=0 and fx=Nyquist columns are self-conjugate (weight 1).
    Sum of weights equals H*W, the number of real pixels.
    """
    w = torch.full((height, width // 2 + 1), 2.0, device=device, dtype=dtype)
    w[:, 0] = 1.0
    if width % 2 == 0:
        w[:, -1] = 1.0
    return w


def weighted_spectral_energy(spectrum, weights):
    """Parseval energy summed over the one-sided spectrum with multiplicities."""
    return (spectrum.abs().pow(2) * weights).sum(dim=(-2, -1))
