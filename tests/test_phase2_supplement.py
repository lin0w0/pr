import math
import unittest

import numpy as np
import torch

from datasets.preprocessing import fixed_mask_normalize, official_normalize
from evaluation.detection_metrics import (
    auc_lower_is_adv,
    multiscale_score,
    rates_at_threshold,
    threshold_at_fpr,
)
from spectral.band_projection import (
    band_mask,
    dykstra_project_band_l2_box,
    project_l2,
)
from spectral.fft_utils import one_sided_weights, rfft2
from spectral.frequency import radial_frequency
from spectral.gaussian_filter import filter_response_table, gaussian_response


class TestOneSidedWeights(unittest.TestCase):
    def test_weights_and_parseval(self):
        w = one_sided_weights(128, 128)
        self.assertEqual(tuple(w.shape), (128, 65))
        self.assertAlmostEqual(float(w.sum()), 128 * 128)
        self.assertEqual(float(w[0, 0]), 1.0)
        self.assertEqual(float(w[0, -1]), 1.0)
        self.assertEqual(float(w[0, 1]), 2.0)

        torch.manual_seed(0)
        x = torch.randn(2, 1, 128, 128)
        spec = rfft2(x, norm="ortho")
        weighted = (spec.abs().pow(2) * w).sum(dim=(-2, -1))
        real = x.pow(2).sum(dim=(-2, -1))
        self.assertLess(float((weighted - real).abs().max()), 0.01)


class TestFilterCutoffs(unittest.TestCase):
    def test_half_gain_and_minus3db(self):
        sigma = 0.1
        row = filter_response_table([sigma], [0.0, 0.5, 1.0])[0]
        rho_half = row["rho_half_gain"]
        rho_3db = row["rho_minus3db"]
        gain_half = float(gaussian_response(torch.tensor([rho_half]), sigma))
        gain_3db = float(gaussian_response(torch.tensor([rho_3db]), sigma))
        self.assertAlmostEqual(gain_half, 0.5, places=5)
        self.assertAlmostEqual(gain_3db, 1.0 / math.sqrt(2.0), places=5)
        self.assertLess(rho_3db, rho_half)


class TestFixedMask(unittest.TestCase):
    def test_identical_when_no_nonpositive(self):
        x = torch.rand(2, 1, 16, 16) * 0.8 + 0.1
        self.assertTrue(torch.allclose(official_normalize(x), fixed_mask_normalize(x), atol=1e-6))

    def test_differs_with_zeros(self):
        x = torch.rand(1, 1, 16, 16) * 0.8 + 0.1
        x[0, 0, 0, 0] = 0.0
        dyn = official_normalize(x)
        fix = fixed_mask_normalize(x)
        self.assertEqual(float(dyn[0, 0, 0, 0]), 0.0)
        self.assertNotAlmostEqual(float(fix[0, 0, 0, 0]), float(dyn[0, 0, 0, 0]))


class TestL2Projection(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.rand(2, 1, 16, 16) * 0.6 + 0.2
        self.rho = radial_frequency(16, 16)
        self.mask = band_mask(self.rho, (0.0, 0.2))

    def test_project_l2(self):
        d = torch.randn(2, 1, 16, 16) * 2.0
        p = project_l2(d, 0.5)
        norms = p.flatten(1).norm(dim=1)
        self.assertTrue(bool((norms <= 0.5 + 1e-6).all()))

    def test_dykstra_band_l2_box(self):
        d = torch.randn(2, 1, 16, 16) * 0.2
        out, metrics = dykstra_project_band_l2_box(
            d, self.x, 0.2, self.mask, n_iters=50, tol=1e-8
        )
        self.assertLessEqual(float(metrics["l2"].max()), 0.2 + 1e-5)
        self.assertLess(float(metrics["band_outside_energy_ratio"].max()), 1e-3)
        self.assertLess(float(metrics["box_violation"].max()), 1e-6)
        xa = self.x + out
        self.assertGreaterEqual(float(xa.min()), -1e-6)
        self.assertLessEqual(float(xa.max()), 1 + 1e-6)


class TestDetectionMetrics(unittest.TestCase):
    def test_threshold_and_rates(self):
        clean = np.array([0.9, 0.92, 0.94, 0.96, 0.98, 1.0])
        adv = np.array([0.2, 0.4, 0.8])
        tau = threshold_at_fpr(clean, 1.0 / 6.0)
        r = rates_at_threshold(clean, adv, tau)
        self.assertLessEqual(r["FPR"], 1.0 / 6.0 + 1e-12)
        self.assertAlmostEqual(r["TPR"], 1.0)

    def test_auc_orientation(self):
        clean = np.array([0.9, 0.95, 0.99])
        adv = np.array([0.1, 0.2, 0.3])
        self.assertAlmostEqual(auc_lower_is_adv(clean, adv), 1.0)

    def test_multiscale(self):
        arr = np.array([[0.9, 0.8], [0.4, 0.6]])
        self.assertTrue(np.allclose(multiscale_score(arr, "mean"), [0.85, 0.5]))
        self.assertTrue(np.allclose(multiscale_score(arr, "min"), [0.8, 0.4]))


if __name__ == "__main__":
    unittest.main()
