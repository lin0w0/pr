import unittest

import torch
import torch.nn.functional as F

from models import build_adapter
from utils.config import load_model_config

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class TestAdapters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapters = {}
        for name in ("compnet", "ccnet"):
            cfg = load_model_config(name)
            cls.adapters[name] = build_adapter(name, device=DEVICE, config=cfg)

    def test_interface_shapes_and_norms(self):
        for name, adapter in self.adapters.items():
            report = adapter.validate()
            self.assertEqual(report["feature_dim"], adapter.expected_feature_dim)
            self.assertEqual(report["num_classes"], 600)
            self.assertLess(report["feature_norm_max_err"], 1e-5)
            x = torch.rand(2, 1, 128, 128, device=DEVICE) * 0.8 + 0.1
            with torch.no_grad():
                f = adapter.extract_features(x)
                self.assertEqual(tuple(f.shape), (2, adapter.feature_dim))
                self.assertEqual(tuple(adapter.get_model_logits(x).shape), (2, 600))

    def test_model_logits_are_30_cosine_of_features(self):
        for name, adapter in self.adapters.items():
            x = torch.rand(3, 1, 128, 128, device=DEVICE) * 0.8 + 0.1
            with torch.no_grad():
                z = adapter.preprocess_raw(x)
                f = adapter.net.getFeatureCode(z)
                logits = adapter.get_model_logits(x)
                weight = adapter.net.arclayer.weight if hasattr(adapter.net, "arclayer") \
                    else adapter.net.arclayer_.weight
                cos = F.linear(F.normalize(f), F.normalize(weight))
            max_err = float((logits - 30.0 * cos).abs().max())
            self.assertLess(max_err, 1e-4, msg=f"{name}: {max_err}")

    def test_ccnet_ignored_keys_documented(self):
        cc = self.adapters["ccnet"]
        self.assertEqual(cc.checkpoint_info["n_ignored_tensors"], 36)
        self.assertEqual(cc.checkpoint_info["n_active_tensors"], 59)
        comp = self.adapters["compnet"]
        self.assertEqual(comp.checkpoint_info["n_ignored_tensors"], 0)

    def test_input_gradient_flows(self):
        for name, adapter in self.adapters.items():
            x = (torch.rand(2, 1, 128, 128, device=DEVICE) * 0.8 + 0.1).requires_grad_(True)
            f = adapter.extract_features(x)
            loss = f[:, 0].sum() + f[:, -1].sum()
            g, = torch.autograd.grad(loss, x)
            self.assertTrue(torch.isfinite(g).all(), name)
            self.assertGreater(float(g.abs().max()), 0.0, name)

    def test_frozen_parameters_unchanged(self):
        for name, adapter in self.adapters.items():
            params = list(adapter.net.parameters())
            self.assertTrue(all(not p.requires_grad for p in params), name)
            before = {k: v.detach().clone() for k, v in adapter.net.state_dict().items()}
            x = (torch.rand(2, 1, 128, 128, device=DEVICE) * 0.8 + 0.1).requires_grad_(True)
            loss = adapter.extract_features(x).sum()
            loss.backward()
            for p in params:
                self.assertIsNone(p.grad, f"{name}: parameter received a gradient")
            after = adapter.net.state_dict()
            for k, v in before.items():
                self.assertTrue(torch.equal(v, after[k].detach()), f"{name}: {k} changed")

    def test_similarity(self):
        for name, adapter in self.adapters.items():
            f = F.normalize(torch.randn(4, adapter.feature_dim), dim=1)
            sim = adapter.get_similarity(f, f)
            self.assertEqual(tuple(sim.shape), (4, 4))
            self.assertTrue(torch.allclose(sim.diagonal(), torch.ones(4), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
