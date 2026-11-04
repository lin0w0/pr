import unittest

import numpy as np
import torch
import torch.nn.functional as F

from datasets.gallery import GalleryClassifier, prototypes_from_features
from evaluation.identification_metrics import rank_accuracy
from evaluation.verification_metrics import at_threshold, pairwise_scores, verification_metrics


class TestGallery(unittest.TestCase):
    def test_prototypes_are_mean_then_normalized(self):
        f = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
        labels = torch.tensor([0, 0, 1, 1])
        protos, unique = prototypes_from_features(f, labels)
        expected0 = F.normalize(F.normalize(f[:2], dim=1).mean(0), dim=0)
        expected1 = F.normalize(F.normalize(f[2:], dim=1).mean(0), dim=0)
        self.assertTrue(torch.allclose(protos[0], expected0, atol=1e-6))
        self.assertTrue(torch.allclose(protos[1], expected1, atol=1e-6))
        self.assertEqual(unique.tolist(), [0, 1])
        self.assertTrue(torch.allclose(protos.norm(dim=1), torch.ones(2), atol=1e-6))

    def test_gallery_classifier_logits(self):
        protos = F.normalize(torch.randn(5, 8), dim=1)
        clf = GalleryClassifier(protos, temperature_scale=30.0)
        q = F.normalize(torch.randn(3, 8), dim=1)
        expected = 30.0 * (q @ protos.T)
        self.assertTrue(torch.allclose(clf.logits(q), expected, atol=1e-5))
        self.assertEqual(clf.predict(q).tolist(), clf.scores(q).argmax(1).tolist())


class TestIdentificationMetrics(unittest.TestCase):
    def test_perfect_and_imperfect(self):
        scores = torch.tensor([[0.9, 0.1, 0.0],
                               [0.1, 0.8, 0.1],
                               [0.2, 0.1, 0.7]])
        labels = torch.tensor([0, 1, 2])
        r = rank_accuracy(scores, labels, topk=(1, 2))
        self.assertAlmostEqual(r["rank1"], 100.0)
        self.assertAlmostEqual(r["rank2"], 100.0)

        scores[2] = torch.tensor([0.6, 0.5, 0.1])
        r = rank_accuracy(scores, labels, topk=(1,))
        self.assertAlmostEqual(r["rank1"], 200.0 / 3.0)

    def test_gallery_label_mapping(self):
        scores = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
        # columns correspond to identities 10 and 20
        r = rank_accuracy(scores, torch.tensor([20, 10]), gallery_labels=[10, 20])
        self.assertAlmostEqual(r["rank1"], 100.0)


class TestVerificationMetrics(unittest.TestCase):
    def test_eer_perfect_separation(self):
        genuine = np.array([0.9, 0.8, 0.7])
        impostor = np.array([0.1, 0.2, 0.3])
        m = verification_metrics(genuine, impostor)
        self.assertAlmostEqual(m["eer_pct"], 0.0)
        self.assertAlmostEqual(m["tar_at_far_1pct"], 100.0)

    def test_eer_known_overlap(self):
        genuine = np.array([0.5, 0.6, 0.7, 0.8])
        impostor = np.array([0.5, 0.6, 0.7, 0.8])
        m = verification_metrics(genuine, impostor)
        self.assertAlmostEqual(m["eer_pct"], 50.0)

    def test_at_threshold(self):
        r = at_threshold(np.array([0.9, 0.4]), np.array([0.3, 0.6]), 0.5)
        self.assertAlmostEqual(r["TAR"], 0.5)
        self.assertAlmostEqual(r["FAR"], 0.5)

    def test_pairwise_scores_counts(self):
        a = F.normalize(torch.randn(4, 8), dim=1).numpy()
        b = F.normalize(torch.randn(6, 8), dim=1).numpy()
        la = np.array([0, 0, 1, 1])
        lb = np.array([0, 0, 0, 1, 1, 1])
        gen, imp = pairwise_scores(a, b, la, lb, chunk_size=2)
        self.assertEqual(gen.size + imp.size, 24)
        self.assertEqual(gen.size, 2 * 3 + 2 * 3)
        self.assertEqual(imp.size, 24 - gen.size)


if __name__ == "__main__":
    unittest.main()
