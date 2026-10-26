"""Synthetic arithmetic and data invariants; no local PyTorch dependency."""

from copy import deepcopy
import math
import unittest

from vlm_multiturn_rl.grpo import reference_loss, relative_advantages, synthetic_checks, validate_group
from vlm_multiturn_rl.stage5 import Stage5Config


class GRPOTests(unittest.TestCase):
    def test_population_standard_deviation_and_episode_returns(self):
        result = relative_advantages([0.0, 0.0, 0.0, 1.0])
        self.assertEqual(result["mean"], 0.25)
        self.assertAlmostEqual(result["std"], math.sqrt(3)/4)
        self.assertAlmostEqual(sum(result["advantages"]), 0.0)
        self.assertAlmostEqual(result["advantages"][3], math.sqrt(3), places=5)
        self.assertFalse(result["zero_variance"])

    def test_all_failure_and_all_success_groups_have_finite_zero_advantages(self):
        for reward in (-0.08, 0.98, 0.0):
            result = relative_advantages([reward]*4)
            self.assertTrue(result["zero_variance"])
            self.assertEqual(result["advantages"], [0.0]*4)
            self.assertEqual(result["std"], 0.0)
        self.assertTrue(all(math.isfinite(a) for a in relative_advantages([0, 0, 0, 1e-10])["advantages"]))

    def test_rejects_nonfinite_returns_and_invalid_epsilon(self):
        for returns in ([], [1], [0, float("nan")], [0, float("inf")]):
            with self.assertRaises(ValueError):
                relative_advantages(returns)
        for epsilon in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                relative_advantages([0, 1], epsilon)

    def test_clipping_is_sign_sensitive_and_does_not_discard_bad_ratios(self):
        def objective(ratio, advantage):
            return reference_loss([math.log(ratio)], [0], [True], advantage)

        self.assertAlmostEqual(objective(1.5, 1), -1.2)  # Positive advantage: cap improvement.
        self.assertAlmostEqual(objective(0.5, 1), -0.5)  # Bad change remains penalized.
        self.assertAlmostEqual(objective(0.5, -1), 0.8)  # Negative advantage: cap improvement.
        self.assertAlmostEqual(objective(1.5, -1), 1.5)  # Bad change remains penalized.

    def test_observation_values_do_not_affect_loss(self):
        mask = [False, False, True, True, False]
        expected = reference_loss([0, 0], [0, 0], [True, True], 1)
        self.assertEqual(reference_loss([float("nan"), 1000, 0, 0, -1000],
                                        [float("nan"), -1000, 0, 0, 1000], mask, 1), expected)
        with self.assertRaises(ValueError):
            reference_loss([float("nan")], [0], [True], 1)
        with self.assertRaises(ValueError):
            reference_loss([0], [0], [False], 1)

    def test_equal_episode_weight_differs_from_global_token_weight(self):
        # A short positive trajectory and a long negative trajectory cancel at r=1.
        first = reference_loss([0], [0], [True], 1)
        second = reference_loss([0]*9, [0]*9, [True]*9, -1)
        self.assertEqual((first+second)/2, 0)
        self.assertEqual((first+9*second)/10, 0.8)

    def test_zero_initial_loss_can_have_nonzero_policy_gradient(self):
        advantages = [1, -1]
        def loss(parameter):
            return sum(reference_loss([sign*parameter], [0], [True], advantage)
                       for sign, advantage in zip([1, -1], advantages))/2

        self.assertEqual(loss(0), 0)
        finite_difference = (loss(1e-6)-loss(-1e-6))/(2e-6)
        self.assertAlmostEqual(finite_difference, -1, places=6)
        clipped_derivative = (reference_loss([math.log(1.5)+1e-6], [0], [True], 1)
                              - reference_loss([math.log(1.5)-1e-6], [0], [True], 1))/(2e-6)
        self.assertEqual(clipped_derivative, 0)

    def test_groups_require_same_board_initial_state_cap_and_unique_seeds(self):
        group = [{"layout_id": "same", "initial_position": [1, 2], "max_steps": 8, "seed": i} for i in range(4)]
        validate_group(group)
        for key, value in (("layout_id", "different"), ("initial_position", [1, 3]), ("max_steps", 7), ("seed", 1)):
            changed = deepcopy(group)
            changed[0][key] = value
            with self.assertRaises(ValueError):
                validate_group(changed)
        with self.assertRaises(ValueError):
            validate_group(group[:3])

    def test_smoke_configuration_is_bounded(self):
        self.assertEqual(Stage5Config().updates, 2)
        for kwargs in ({"updates": 0}, {"updates": 21}, {"group_size": 2}, {"max_new_tokens": 16},
                       {"learning_rate": float("nan")}, {"sft_run_id": "../escape"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Stage5Config(**kwargs)
        self.assertTrue(all(synthetic_checks()["checks"].values()))


if __name__ == "__main__":
    unittest.main()
