import importlib.util
import unittest
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

from datasets.preprocessing import OfficialROIPreprocess, load_raw_image, official_normalize
from utils.paths import PROJECT_ROOT

OFFICIAL_DATASET = PROJECT_ROOT / "official_project/compnet/CompNet/models/dataset.py"


def _load_official():
    spec = importlib.util.spec_from_file_location("_official_dataset_for_test", OFFICIAL_DATASET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reference_normalize(t):
    """Exact per-sample copy of the official NormSingleROI logic."""
    c, h, w = t.shape
    out = t.clone().view(c, h * w)
    idx = out > 0
    vals = out[idx]
    m = vals.mean()
    s = vals.std()
    vals = vals.sub_(m).div_(s + 1e-6)
    out[idx] = vals
    return out.view(c, h, w)


class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        official = _load_official()
        cls.official = official
        cls.paths = [f"data/TongjI/ROI/session1/{i:05d}.bmp" for i in range(1, 11)]
        cls.raw = torch.stack([load_raw_image(p) for p in cls.paths])
        tf = T.Compose([T.Resize(128), T.ToTensor(), official.NormSingleROI(outchannels=1)])
        cls.ref = torch.stack([
            tf(Image.open(PROJECT_ROOT / p).convert("L")) for p in cls.paths
        ])

    def test_matches_official_pipeline(self):
        got = official_normalize(self.raw)
        self.assertEqual(tuple(got.shape), tuple(self.ref.shape))
        self.assertTrue(torch.allclose(got, self.ref, atol=1e-6),
                        msg=f"max diff={(got - self.ref).abs().max().item()}")

    def test_zero_pixels_preserved(self):
        x = torch.rand(1, 1, 8, 8) * 0.5 + 0.2
        x[0, 0, 0, 0] = 0.0
        x[0, 0, 3, 4] = 0.0
        got = official_normalize(x)
        ref = _reference_normalize(x[0]).unsqueeze(0)
        self.assertTrue(torch.allclose(got, ref, atol=1e-6))
        self.assertEqual(got[0, 0, 0, 0].item(), 0.0)

    def test_gradient_flows(self):
        x = self.raw.clone().requires_grad_(True)
        y = official_normalize(x)
        loss = y.pow(2).sum()
        g, = torch.autograd.grad(loss, x)
        self.assertTrue(torch.isfinite(g).all())
        self.assertGreater(float(g.abs().max()), 0.0)

    def test_rejects_non_square(self):
        module = OfficialROIPreprocess(128)
        with self.assertRaises(ValueError):
            module(torch.rand(1, 1, 64, 64) + 0.1)


if __name__ == "__main__":
    unittest.main()
