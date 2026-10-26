"""CPU tests for observation controls and exact generation boundaries."""

import unittest

from vlm_multiturn_rl.vision import (
    Board, Stage2Config, control_boards, expected_action, generated_suffix, messages, parse_action,
)


class VisionTests(unittest.TestCase):
    def test_control_changes_goal_without_changing_agent_or_prompt(self):
        boards = control_boards()
        left, right = boards["goal-left"], boards["goal-right"]
        self.assertEqual(left.agent, right.agent)
        self.assertEqual(left.size, right.size)
        self.assertNotEqual(left.goal, right.goal)
        self.assertEqual(expected_action(left), "left")
        self.assertEqual(expected_action(right), "right")
        # Image is a placeholder; fixture coordinates/labels are never in the prompt.
        self.assertEqual(messages()[0]["content"][0], {"type": "image"})
        self.assertNotIn("goal-left", str(messages()))
        self.assertNotIn("goal-right", str(messages()))

    def test_coordinate_convention_and_adjacent_goal_requirement(self):
        self.assertEqual(expected_action(Board((2, 2), (1, 2))), "up")
        self.assertEqual(expected_action(Board((2, 2), (3, 2))), "down")
        with self.assertRaisesRegex(ValueError, "adjacent"):
            expected_action(Board((0, 0), (3, 3)))
        for agent, goal in [((0, 0), (0, 0)), ((-1, 0), (1, 0)), ((0, 0), (4, 0))]:
            with self.subTest(agent=agent, goal=goal), self.assertRaises(ValueError):
                Board(agent, goal)

    def test_parser_rejects_explanations_and_multiple_actions(self):
        self.assertEqual(parse_action(" Right\n"), "right")
        for text in ["", "left or right", "move(left)", "left.", "I choose right"]:
            with self.subTest(text=text):
                self.assertIsNone(parse_action(text))

    def test_suffix_preserves_exact_ids_and_does_not_search_for_eos(self):
        # A token may appear both in the prompt and the completion. Slice by the
        # full prefix length, not by matching token text or the first EOS marker.
        prompt = [11, 99, 12, 99]
        self.assertEqual(generated_suffix(prompt, prompt + [99, 13]), [99, 13])
        self.assertEqual(generated_suffix(prompt, prompt), [])
        with self.assertRaisesRegex(ValueError, "input prefix"):
            generated_suffix(prompt, [11, 12, 99, 13])

    def test_limits_are_validated_before_remote_work(self):
        for kwargs in [{"max_pixels": 224**2 - 1}, {"max_pixels": 448**2 + 1},
                       {"max_new_tokens": 0}, {"max_new_tokens": 65}, {"seed": -1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Stage2Config(**kwargs)


if __name__ == "__main__":
    unittest.main()
