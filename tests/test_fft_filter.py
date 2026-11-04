import math
import unittest

import torch

from spectral.fft_utils import rfft2, irfft2
from spectral.frequency import radial_frequency
from spectral.gaussian_filter import (
    filter_response_table,
    gaussian_lowpass,
    gaussian_response,
    response_curve,
)


class TestFFT(unittest.TestCase):
    def test_roundtrip(self):
        x = torch.rand(2, 1, 32, 32)
        rec = irfft2(rfft2(x, norm="ortho"), x.shape[-2:], norm="ortho")
        self.assertLess(float((rec - x).abs().max()), 1e-5)

    def test_spectrum_shape(self):
        x = torch.rand(2, 1, 128, 128)
        self.assertEqual(tuple(rfft2(x).shape), (2, 1, 128, 65))

    def test_filter_gradient(self):
        x = (torch.rand(1, 1, 32, 32) * 0.8 + 0.1).requires_grad_(True)
        y = gaussian_lowpass(x, 0.2)
        loss = y.pow(2).sum()
        g, = torch.autograd.grad(loss, x)
        self.assertTrue(torch.isfinite(g).all())
        self.assertGreater(float(g.abs().max()), 0.0)


class TestGaussianFilter(unittest.TestCase):
    def setUp(self):
        self.rho = radial_frequency(128, 128)

    def test_dc_gain_is_one(self):
        for sigma in (0.02, 0.1, 1.0):
            self.assertAlmostEqual(float(gaussian_response(torch.tensor([0.0]), sigma)), 1.0, places=6)

    def test_smaller_sigma_is_stronger(self):
        r = torch.tensor([0.05, 0.2, 0.5, 1.0])
        strong = gaussian_response(r, 0.1)
        weak = gaussian_response(r, 0.5)
        self.assertTrue(bool((strong <= weak + 1e-7).all()))
        self.assertLess(float(strong[1]), float(weak[1]))

    def test_response_is_monotone_decreasing(self):
        r, g = response_curve(0.2, n=101)
        self.assertTrue(bool((g[1:] <= g[:-1] + 1e-7).all()))

    def test_lowpass_attenuates_high_frequency(self):
        # 2D cosine at 0.4 cycles/pixel (high band).
        t = torch.arange(32, dtype=torch.float32)
        xx, yy = torch.meshgrid(t, t, indexing="ij")
        x = 0.5 + 0.4 * torch.cos(2 * math.pi * 0.4 * xx) * torch.cos(2 * math.pi * 0.4 * yy)
        x = x.view(1, 1, 32, 32)
        low = gaussian_lowpass(x, 0.05)
        self.assertLess(float(low.std()), 0.25 * float(x.std()))

    def test_near_identity_for_very_large_sigma(self):
        x = torch.rand(1, 1, 32, 32)
        low = gaussian_lowpass(x, 1e4)  # G ~= 1 - rho^2 / (2e8)
        self.assertLess(float((low - x).abs().max()), 1e-6)

    def test_response_table_cutoff(self):
        rows = filter_response_table([0.1, 0.5], [0.0, 0.5, 1.0])
        self.assertEqual(len(rows), 2)
        for row in rows:
            expected = row["sigma"] * math.sqrt(2 * math.log(2))
            self.assertAlmostEqual(row["cutoff_rho_half_gain"], expected, places=6)

    def test_shape_and_real(self):
        x = torch.rand(3, 1, 128, 128)
        low = gaussian_lowpass(x, 0.3)
        self.assertEqual(low.shape, x.shape)
        self.assertFalse(torch.is_complex(low))


if __name__ == "__main__":
    unittest.main()
