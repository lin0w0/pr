import unittest

import torch

from evaluation.official_matching import (
    nearest_gallery_identity,
    rank_accuracy_image_gallery,
)


class TestOfficialMatching(unittest.TestCase):
    def setUp(self):
        # Two identities, two registration images each. The second image of an
        # identity is deliberately dissimilar so top-k can split across IDs.
        self.gallery = torch.tensor([
            [1.0, 0.0],       # id 0
            [-1.0, 0.0],      # id 0
            [0.0, 1.0],       # id 1
            [0.0, -1.0],      # id 1
        ])
        self.gallery_labels = [0, 0, 1, 1]

    def test_rank1_uses_nearest_image_not_mean(self):
        probes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        r = rank_accuracy_image_gallery(
            probes, [0, 1], self.gallery, self.gallery_labels, topk=(1, 2)
        )
        self.assertAlmostEqual(r["rank1"], 100.0)
        self.assertAlmostEqual(r["rank2"], 100.0)

    def test_topk_accepts_any_registration_image(self):
        # Nearest is id 0, second nearest is id 1 (query label 1) -> rank2 hit.
        probes = torch.tensor([[1.0, 0.1]])
        r = rank_accuracy_image_gallery(
            probes, [1], self.gallery, self.gallery_labels, topk=(1, 2)
        )
        self.assertAlmostEqual(r["rank1"], 0.0)
        self.assertAlmostEqual(r["rank2"], 100.0)

    def test_rank5_uses_identity_max_not_top5_images(self):
        # Five id-0 images are all closer than the only id-1 image. Taking the
        # top-5 gallery *images* would return five id-0 images and miss id 1;
        # identity-level max scores must rank id 1 second.
        gallery = torch.tensor([
            [1.0, 0.00], [1.0, 0.01], [1.0, 0.02], [1.0, 0.03], [1.0, 0.04],
            [0.0, 1.0],
        ])
        labels = [0, 0, 0, 0, 0, 1]
        probes = torch.tensor([[1.0, 0.05]])
        r = rank_accuracy_image_gallery(
            probes, [1], gallery, labels, topk=(1, 5)
        )
        self.assertAlmostEqual(r["rank1"], 0.0)
        self.assertAlmostEqual(r["rank5"], 100.0)

    def test_nearest_identity(self):
        probes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        pred = nearest_gallery_identity(probes, self.gallery, self.gallery_labels)
        self.assertEqual(pred.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
