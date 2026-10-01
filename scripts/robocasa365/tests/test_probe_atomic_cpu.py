import unittest
import numpy as np
from scripts.robocasa365.probe_atomic_cpu import binary_metrics


class BinaryMetricTest(unittest.TestCase):
    def test_majority_accuracy_does_not_hide_missing_positive_mode(self):
        truth = np.array([0.] * 9 + [1.])
        metrics = binary_metrics(np.zeros(10), truth, np.ones(10, dtype=bool))
        self.assertEqual(metrics['accuracy'], .9)
        self.assertEqual(metrics['positive_recall'], 0.)
        self.assertEqual(metrics['balanced_accuracy'], .5)

    def test_padding_and_exact_half_threshold(self):
        result = binary_metrics(np.array([.5, .49, 1.]), np.array([1., 0., 0.]), np.array([True, True, False]))
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['accuracy'], 1.)

    def test_absent_class_is_not_claimed_as_balanced(self):
        result = binary_metrics(np.ones(4), np.ones(4), np.ones(4, dtype=bool))
        self.assertIsNone(result['negative_recall'])
        self.assertIsNone(result['balanced_accuracy'])


if __name__ == '__main__':
    unittest.main()
