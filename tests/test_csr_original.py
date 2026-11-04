import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from defenses.csr_original import OriginalCSR


class DummyAdapter:
    """Differentiable toy feature extractor with frozen parameters."""
    device = "cpu"

    def __init__(self):
        self.net = nn.Linear(1, 1, bias=False)
        self.net.weight.requires_grad_(False)

    def extract_features(self, x):
        # Per-image mean and a high-frequency component.
        mean = x.mean(dim=(2, 3))
        hf = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=(2, 3))
        v = torch.cat([mean, hf], dim=1)
        return F.normalize(v, dim=1)


class TestOriginalCSR(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.ad = DummyAdapter()
        self.csr = OriginalCSR(self.ad, radius=3.0, lam=1.0, steps=3,
                               eps=0.05, alpha=0.02)
        self.x = torch.rand(3, 1, 16, 16) * 0.8 + 0.1

    def test_shape_range_budget(self):
        out, records = self.csr.rectify(self.x, return_records=True)
        self.assertEqual(tuple(out.shape), tuple(self.x.shape))
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)
        delta = out - self.x
        self.assertLessEqual(float(delta.abs().max()), 0.05 + 1e-6)
        self.assertTrue(all(0 <= r["step"] < 3 for r in records))

    def test_model_params_have_no_grad(self):
        _ = self.csr.rectify(self.x)
        for p in self.ad.net.parameters():
            self.assertIsNone(p.grad)
            self.assertFalse(p.requires_grad)

    def test_gradient_propagates_to_input(self):
        out, records = self.csr.rectify(self.x, return_records=True)
        self.assertFalse(torch.allclose(out, self.x))
        self.assertTrue(all(torch.isfinite(torch.tensor(r["lrec"])) for r in records))

    def test_greedy_selection_is_max(self):
        out, records = self.csr.rectify(self.x, return_records=True)
        for j in range(3):
            recs = [r for r in records if r["local_index"] == j]
            best = max(r["lrec"] for r in recs)
            selected = [r for r in recs if r["selected"]]
            self.assertEqual(len(selected), 1)
            self.assertAlmostEqual(selected[0]["lrec"], best, places=6)

    def test_same_input_reproducible(self):
        out1 = self.csr.rectify(self.x)
        out2 = self.csr.rectify(self.x)
        self.assertTrue(torch.equal(out1, out2))

    def test_detection_score_range(self):
        s = self.csr.score(self.x)
        self.assertTrue(bool((s >= -1.0001).all() and (s <= 1.0001).all()))

    def test_process_only_flagged(self):
        # tau=-2 flags nothing; tau=2 flags everything.
        out_none, mask_none, rec_none, _ = self.csr.process(self.x, tau=-2.0)
        self.assertFalse(bool(mask_none.any()))
        self.assertTrue(torch.equal(out_none, self.x))
        self.assertEqual(rec_none, [])
        out_all, mask_all, rec_all, _ = self.csr.process(self.x, tau=2.0)
        self.assertTrue(bool(mask_all.all()))
        self.assertGreater(len(rec_all), 0)


if __name__ == "__main__":
    unittest.main()
