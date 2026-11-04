import unittest

import torch
import torch.nn.functional as F

from spectral.frequency import band_masks, radial_frequency
from spectral.losses import cross_entropy_per_sample, margin_per_sample
from spectral.sgm import amplitude_gradient, gradient_metrics, reconstruct_from_amplitude


class TestFrequency(unittest.TestCase):
    def test_radial_frequency_range_and_shape(self):
        rho = radial_frequency(128, 128)
        self.assertEqual(tuple(rho.shape), (128, 65))
        self.assertGreaterEqual(float(rho.min()), 0.0)
        self.assertLessEqual(float(rho.max()), 1.0 + 1e-6)

    def test_band_masks_disjoint_and_cover(self):
        rho = radial_frequency(128, 128)
        masks = band_masks(rho)
        self.assertEqual(set(masks), {"low", "middle", "high"})
        self.assertFalse(bool((masks["low"] & masks["middle"]).any()))
        self.assertFalse(bool((masks["mid" + "dle"] & masks["high"]).any()))
        union = masks["low"] | masks["middle"] | masks["high"]
        self.assertTrue(bool(union.all()))


class TestLosses(unittest.TestCase):
    def test_margin_value(self):
        logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]])
        labels = torch.tensor([1, 0])
        # sample 0: max_other=3, s_y=2 -> 1 ; sample 1: max_other=1, s_y=3 -> -2
        out = margin_per_sample(logits, labels)
        self.assertTrue(torch.allclose(out, torch.tensor([1.0, -2.0])))

    def test_ce_positive(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        out = cross_entropy_per_sample(logits, torch.tensor([1]))
        self.assertGreater(float(out), 0.0)


class TestAmplitudeGradient(unittest.TestCase):
    def test_roundtrip(self):
        x = torch.rand(2, 1, 16, 16) * 0.8 + 0.1
        spec = torch.fft.rfft2(x, norm="ortho")
        rec = reconstruct_from_amplitude(spec.abs(), spec.angle(), (16, 16))
        self.assertLess(float((rec - x).abs().max()), 1e-5)

    def test_gradient_through_dummy_model(self):
        class DummyAdapter:
            device = "cpu"

            def extract_features(self, x):
                v = x.mean(dim=(2, 3))
                return F.normalize(torch.cat([v, -v, torch.full_like(v, 0.1)], dim=1), dim=1)

        class DummyClassifier:
            def scores(self, f):
                return f

        x = torch.rand(2, 1, 16, 16) * 0.8 + 0.1
        labels = torch.tensor([0, 1])
        grad, losses, logits = amplitude_gradient(
            DummyAdapter(), DummyClassifier(), x, labels,
            cross_entropy_per_sample, logit_scale=10.0, returns="all",
        )
        self.assertEqual(tuple(grad.shape), (2, 1, 16, 9))
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.0)
        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_min_batch_padding_preserves_sample_gradient(self):
        class DummyAdapter:
            device = "cpu"

            def extract_features(self, x):
                v = x.mean(dim=(2, 3))
                return F.normalize(torch.cat([v, -v, torch.full_like(v, 0.1)], dim=1), dim=1)

        class DummyClassifier:
            def scores(self, f):
                return f

        x = torch.rand(4, 1, 16, 16) * 0.8 + 0.1
        labels = torch.tensor([0, 1, 0, 1])
        g1 = amplitude_gradient(
            DummyAdapter(), DummyClassifier(), x[:1], labels[:1],
            cross_entropy_per_sample, logit_scale=10.0, min_batch=4,
        )
        g4 = amplitude_gradient(
            DummyAdapter(), DummyClassifier(), x, labels,
            cross_entropy_per_sample, logit_scale=10.0, min_batch=4,
        )[:1]
        self.assertEqual(tuple(g1.shape), (1, 1, 16, 9))
        self.assertTrue(torch.allclose(g1, g4, atol=1e-6))


class TestGradientMetrics(unittest.TestCase):
    def test_low_band_only(self):
        rho = radial_frequency(8, 8)
        grad = torch.zeros(1, 1, 8, 5)
        grad[:, :, rho < 0.2] = 1.0
        m = gradient_metrics(grad, rho)
        self.assertAlmostEqual(m["band_ratio_aggregate"]["low"], 1.0, places=6)
        self.assertAlmostEqual(m["band_ratio_aggregate"]["middle"], 0.0, places=6)
        self.assertAlmostEqual(m["band_ratio_aggregate"]["high"], 0.0, places=6)
        self.assertAlmostEqual(m["near_zero_l1_lt_1e-12_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
