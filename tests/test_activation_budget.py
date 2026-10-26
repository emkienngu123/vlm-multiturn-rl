"""Memory budget decisions are CPU-testable without PyTorch."""
import unittest
from vlm_multiturn_rl.browser_policy import ActivationBudget


class ActivationBudgetTests(unittest.TestCase):
    def test_default_offloads_and_never_retains(self):
        b = ActivationBudget()
        self.assertFalse(b.keep(100, 10000, 1000))
        self.assertEqual((b.retained_bytes, b.offloaded_bytes), (0, 100))

    def test_cap_and_free_memory_headroom_both_apply(self):
        b = ActivationBudget(200)
        self.assertTrue(b.keep(100, 2000, 1000))
        self.assertFalse(b.keep(50, 1049, 1000))
        self.assertTrue(b.keep(100, 2000, 1000))
        self.assertFalse(b.keep(1, 2000, 1000))
        self.assertEqual((b.retained_bytes, b.offloaded_bytes), (200, 51))

    def test_invalid_budgets_rejected(self):
        for value in (-1, 1.5, True):
            with self.assertRaises(ValueError): ActivationBudget(value)
