"""
Tests for Fidelity & Explanation Validation
"""

import unittest
from glassbox_molecular.fidelity import evaluate_explanation_fidelity

class TestFidelity(unittest.TestCase):
    def test_fidelity_metrics(self):
        attributions = [0.1, -0.3, -0.3, -0.3, 0.85, 0.90]
        fid = evaluate_explanation_fidelity(attributions)
        self.assertGreater(fid.fidelity_plus, 0.50)
        self.assertGreater(fid.fidelity_minus, 0.50)
        self.assertGreater(fid.sparsity, 0.0)
        self.assertLess(fid.sparsity, 1.0)
        self.assertTrue(fid.is_valid_explanation)

    def test_empty_attributions(self):
        fid = evaluate_explanation_fidelity([])
        self.assertFalse(fid.is_valid_explanation)

if __name__ == "__main__":
    unittest.main()
