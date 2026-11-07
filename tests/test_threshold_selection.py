import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baseline'))
from reconstruct_10354 import best_cutoff, scores


class ThresholdSelectionTests(unittest.TestCase):
    def test_exact_search_matches_brute_force_at_group_boundaries(self):
        cases = [
            ([0.1, 0.1, 0.2, 0.3, 0.3, 0.9], [0, 1, 1, 0, 1, 1]),
            ([1, 1, 0, 0], [1, 0, 1, 0]),
            ([0.5, np.nextafter(np.float32(0.5), np.float32(1))], [0, 1]),
            ([0.2, 0.3], [1, 1]),
            ([0.2, 0.8, 0.8, 0.8], [1, 0, 1, 1]),
        ]
        for probability, truth in cases:
            with self.subTest(probability=probability):
                probability = np.asarray(probability, dtype=np.float32)
                truth = np.asarray(truth)
                cutoff, measured = best_cutoff(truth, probability)
                candidates = [-1.0] + list(np.unique(probability.astype(float)))
                optimum = max(scores(truth, probability, value)['customer_f0_5'] for value in candidates)
                self.assertAlmostEqual(measured['customer_f0_5'], optimum, places=12)
                self.assertAlmostEqual(scores(truth, probability, cutoff)['customer_f0_5'], optimum, places=12)

    def test_equal_probabilities_are_kept_as_a_group(self):
        probability = np.array([0.5, 0.5, 0.5, 0.5])
        truth = np.array([1, 1, 0, 0])
        cutoff, result = best_cutoff(truth, probability)
        self.assertEqual(result['tp'], 2)
        self.assertEqual(result['fp'], 2)
        self.assertLess(cutoff, 0.5)


if __name__ == '__main__':
    unittest.main()
