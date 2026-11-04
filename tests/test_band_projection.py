import unittest

import torch

from spectral.band_projection import (
    band_mask,
    constraint_metrics,
    dykstra_project,
    project_band,
    project_box,
)
from spectral.fft_utils import band_energy_ratio
from spectral.frequency import radial_frequency


class TestBandMask(unittest.TestCase):
    def test_disjoint_and_cover(self):
        rho = radial_frequency(128, 128)
        low = band_mask(rho, (0.0, 0.2))
        mid = band_mask(rho, (0.2, 0.5))
        high = band_mask(rho, (0.5, 1.0))
        self.assertFalse(bool((low & mid).any()))
        self.assertFalse(bool((mid & high).any()))
        self.assertFalse(bool((low & high).any()))
        self.assertTrue(bool((low | mid | high).all()))


class TestProjections(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.rho = radial_frequency(128, 128)
        self.mask = band_mask(self.rho, (0.0, 0.2))
        self.x = torch.rand(2, 1, 128, 128) * 0.6 + 0.2
        self.delta = torch.randn(2, 1, 128, 128) * 0.05

    def test_project_band_removes_outside_energy(self):
        projected = project_band(self.delta, self.mask)
        ratio = band_energy_ratio(projected, self.mask)
        self.assertLess(float(ratio.max()), 1e-6)

    def test_project_box_satisfies_bounds(self):
        eps = 0.02
        projected = project_box(self.delta, self.x, eps)
        x_adv = self.x + projected
        self.assertLessEqual(float((projected.abs() - eps).clamp_min(0).max()), 1e-7)
        self.assertGreaterEqual(float(x_adv.min()), 0.0)
        self.assertLessEqual(float(x_adv.max()), 1.0)

    def test_dykstra_intersection(self):
        eps = 0.02
        projected, metrics = dykstra_project(
            self.delta, self.x, eps, self.mask, n_iters=50, tol=1e-7
        )
        x_adv = self.x + projected
        self.assertLessEqual(float((projected.abs() - eps).clamp_min(0).max()), 1e-6)
        self.assertGreaterEqual(float(x_adv.min()), -1e-7)
        self.assertLessEqual(float(x_adv.max()), 1 + 1e-7)
        self.assertLess(float(metrics["band_outside_energy_ratio"].max()), 1e-3)
        self.assertEqual(metrics["linf"].shape, (2,))
        self.assertEqual(metrics["iterations"].shape, (2,))

    def test_constraint_metrics_on_exact_band_limited(self):
        delta = project_band(self.delta, self.mask)
        m = constraint_metrics(delta, self.x, 0.05, self.mask)
        self.assertLess(float(m["band_outside_energy_ratio"].max()), 1e-6)
        self.assertLess(float(m["box_violation"].max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
