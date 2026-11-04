import unittest

import numpy as np

from evaluation.verification_metrics import (
    at_threshold,
    equal_error_rate,
    evaluate_at_threshold,
    roc_curve,
    threshold_at_far,
    verification_metrics,
)


def brute_force_eer(genuine, impostor, tol=1e-12):
    """Independent reference: enumerate every distinct score as a threshold."""
    g = np.asarray(genuine, dtype=np.float64)
    i = np.asarray(impostor, dtype=np.float64)
    ths = sorted(set(g.tolist()) | set(i.tolist()), reverse=True)
    fpr = [0.0] + [float((i >= t).mean()) for t in ths] + [1.0]
    tpr = [0.0] + [float((g >= t).mean()) for t in ths] + [1.0]
    fnr = [1.0 - x for x in tpr]
    d = [a - b for a, b in zip(fpr, fnr)]
    interior = list(range(1, len(d) - 1))
    for j in interior:
        if abs(d[j]) <= tol:
            return fpr[j] * 100.0, "exact"
    for j in interior:
        if d[j - 1] < -tol and d[j] > tol:
            u = (fnr[j - 1] - fpr[j - 1]) / ((fpr[j] - fpr[j - 1]) - (fnr[j] - fnr[j - 1]))
            x = fpr[j - 1] + u * (fpr[j] - fpr[j - 1])
            return x * 100.0, "interpolated"
    best = min(interior, key=lambda j: abs(d[j]))
    return (fpr[best] + fnr[best]) / 2 * 100.0, "closest"


class TestEER(unittest.TestCase):
    def test_perfect_separation(self):
        m = equal_error_rate([0.9, 0.8, 0.7], [0.1, 0.2, 0.3])
        self.assertAlmostEqual(m["eer_pct"], 0.0, places=8)
        self.assertEqual(m["eer_mode"], "exact")

    def test_identical_distributions(self):
        m = equal_error_rate([0.5, 0.6, 0.7, 0.8], [0.5, 0.6, 0.7, 0.8])
        self.assertAlmostEqual(m["eer_pct"], 50.0, places=6)

    def test_impostor_only_threshold_is_required(self):
        """The true crossing is at 0.75, an impostor-only score.

        Sampling only genuine thresholds (0.8, 0.7, 0.6) would pick 0.8 or 0.7
        and report ~50%; the complete ROC gives 2/3.
        """
        genuine = [0.6, 0.7, 0.8]
        impostor = [0.5, 0.75, 0.9]
        m = equal_error_rate(genuine, impostor)
        self.assertAlmostEqual(m["eer_pct"], 100.0 * 2 / 3, places=5)
        self.assertEqual(m["eer_mode"], "exact")
        self.assertAlmostEqual(m["eer_threshold"], 0.75, places=6)

    def test_ties_count_as_accepted(self):
        # At threshold 0.5 all ties on both sides are accepted:
        # (FAR, FRR) = (2/3, 0); interpolating to the next operating point
        # (FAR, FRR) = (0, 2/3) gives a 1/3 equal-error crossing.
        m = equal_error_rate([0.5, 0.5, 0.9], [0.5, 0.5, 0.1])
        self.assertAlmostEqual(m["eer_pct"], 100.0 / 3.0, places=5)
        self.assertEqual(m["eer_mode"], "interpolated")

    def test_roc_contains_all_score_thresholds(self):
        _, _, th = roc_curve([0.6, 0.7, 0.8], [0.5, 0.75, 0.9])
        th = th[np.isfinite(th)]
        for value in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9):
            self.assertTrue(np.any(np.isclose(th, value)), value)

    def test_matches_brute_force_on_random_data(self):
        rng = np.random.default_rng(0)
        for seed in range(20):
            # The production scores are float32 cosine values; the EER code
            # sorts float32, so compare against the same quantized inputs.
            gen = rng.normal(0.7, 0.1, size=50).astype(np.float32)
            imp = rng.normal(0.3, 0.1, size=200).astype(np.float32)
            got = equal_error_rate(gen, imp)
            expected, mode = brute_force_eer(gen, imp)
            self.assertAlmostEqual(got["eer_pct"], expected, places=6, msg=f"seed {seed}")
            self.assertEqual(got["eer_mode"], mode, msg=f"seed {seed}")

    def test_verification_metrics_marks_oracle(self):
        m = verification_metrics([0.9, 0.8], [0.1, 0.2])
        self.assertTrue(m["tar_at_far_1pct_is_test_oracle"])
        self.assertTrue(m["tar_at_far_0.1pct_is_test_oracle"])


class TestThresholdCalibration(unittest.TestCase):
    def test_target_far(self):
        impostor = np.arange(100, dtype=np.float64)  # 0..99
        genuine = np.arange(50, 150, dtype=np.float64)
        cal = threshold_at_far(genuine, impostor, 0.10)
        self.assertAlmostEqual(cal["threshold"], 90.0)
        self.assertAlmostEqual(cal["calibration_far"], 0.10)
        self.assertLessEqual(cal["calibration_far"], 0.10)

    def test_zero_far_goes_above_max(self):
        impostor = np.array([0.1, 0.2, 0.3])
        cal = threshold_at_far([0.9, 0.8], impostor, 0.0)
        self.assertGreater(cal["threshold"], 0.3)
        self.assertAlmostEqual(cal["calibration_far"], 0.0)

    def test_far_one_accepts_all(self):
        cal = threshold_at_far([0.9], [0.1, 0.2], 1.0)
        self.assertTrue(np.isinf(cal["threshold"]) and cal["threshold"] < 0)
        self.assertAlmostEqual(cal["calibration_far"], 1.0)

    def test_apply_to_held_out_test(self):
        val_gen, val_imp = [0.9, 0.8], [0.1, 0.2, 0.3]
        cal = threshold_at_far(val_gen, val_imp, 0.0)
        test_gen, test_imp = [0.95, 0.2], [0.2, 0.1]
        res = evaluate_at_threshold(test_gen, test_imp, cal["threshold"])
        self.assertAlmostEqual(res["TAR"], 0.5)
        self.assertAlmostEqual(res["FAR"], 0.0)
        self.assertAlmostEqual(res["FRR"], 0.5)

    def test_at_threshold_ties(self):
        r = at_threshold(np.array([0.9, 0.4]), np.array([0.3, 0.6]), 0.5)
        self.assertAlmostEqual(r["TAR"], 0.5)
        self.assertAlmostEqual(r["FAR"], 0.5)


if __name__ == "__main__":
    unittest.main()
