import unittest

import torch
import torch.nn.functional as F

from attacks.band_limited import band_limited_feature_shift, band_limited_margin
from attacks.pgd import pgd_linf
from spectral.band_projection import band_mask
from spectral.frequency import radial_frequency
from spectral.losses import margin_per_sample


class DummyAdapter:
    device = "cpu"

    def extract_features(self, x):
        v = x.mean(dim=(2, 3))
        return F.normalize(torch.cat([v, -v, torch.full_like(v, 0.1)], dim=1), dim=1)


class DummyClassifier:
    def scores(self, features):
        return features


class TestPGD(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.rand(2, 1, 16, 16) * 0.8 + 0.1
        self.y = torch.tensor([0, 1])
        self.ad = DummyAdapter()
        self.clf = DummyClassifier()

    def test_budget_and_box(self):
        eps = 0.05
        x_adv = pgd_linf(self.ad, self.clf, self.x, self.y, eps=eps, alpha=0.01,
                         steps=5, random_start=True, seed=0, loss="margin", scale=10)
        delta = x_adv - self.x
        self.assertLessEqual(float((delta.abs() - eps).clamp_min(0).max()), 1e-6)
        self.assertGreaterEqual(float(x_adv.min()), 0.0)
        self.assertLessEqual(float(x_adv.max()), 1.0)

    def test_loss_increases(self):
        eps = 0.05
        with torch.no_grad():
            before = margin_per_sample(10 * self.clf.scores(self.ad.extract_features(self.x)), self.y).mean()
        x_adv = pgd_linf(self.ad, self.clf, self.x, self.y, eps=eps, alpha=0.01,
                         steps=5, random_start=False, seed=0, loss="margin", scale=10)
        with torch.no_grad():
            after = margin_per_sample(10 * self.clf.scores(self.ad.extract_features(x_adv)), self.y).mean()
        self.assertGreater(float(after), float(before))


class TestBandLimited(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.rand(2, 1, 16, 16) * 0.6 + 0.2
        self.y = torch.tensor([0, 1])
        self.rho = radial_frequency(16, 16)
        self.mask = band_mask(self.rho, (0.0, 0.2))
        self.ad = DummyAdapter()
        self.clf = DummyClassifier()

    def test_feature_shift_constraints(self):
        eps = 0.05
        xa, delta, metrics = band_limited_feature_shift(
            self.ad, self.x, self.mask, eps=eps, alpha=0.01, steps=5,
            random_start=True, seed=0, dykstra_iters=20,
        )
        self.assertLess(float(metrics["band_outside_energy_ratio"].max()), 1e-3)
        self.assertLessEqual(float((delta.abs() - eps).clamp_min(0).max()), 1e-6)
        self.assertGreaterEqual(float(xa.min()), -1e-7)
        self.assertLessEqual(float(xa.max()), 1 + 1e-7)

    def test_targeted_margin_constraints(self):
        eps = 0.05
        targets = torch.tensor([2, 2])
        xa, delta, metrics = band_limited_margin(
            self.ad, self.clf, self.x, self.y, self.mask, eps=eps, alpha=0.01,
            steps=5, objective="targeted", targets=targets, scale=10,
            random_start=True, seed=0, dykstra_iters=20,
        )
        self.assertLess(float(metrics["band_outside_energy_ratio"].max()), 1e-3)
        self.assertLessEqual(float((delta.abs() - eps).clamp_min(0).max()), 1e-6)
        self.assertEqual(tuple(xa.shape), tuple(self.x.shape))


if __name__ == "__main__":
    unittest.main()
