"""Check the environment contract without importing Torch, TRL or Pillow."""

import itertools
import json
import unittest

from vlm_multiturn_rl.gridworld import make_layouts, render_svg, shortest_path
from vlm_multiturn_rl.stage7 import Stage7Config, dataset_row
from vlm_multiturn_rl.trajectories import TOOL_PROMPT, move_call
from vlm_multiturn_rl.trl_environment import GridToolEnvironment, dispatch_response


class TRLEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.assets, self.episodes, self.owners = {}, [], {}
        self.counter = itertools.count()
        self.layout = make_layouts("train", 2)[1]

    def env(self):
        return GridToolEnvironment(renderer=render_svg, assets=self.assets, episodes=self.episodes,
                                   image_owners=self.owners, next_key=lambda: f"test-{next(self.counter)}")

    def act(self, env, text):
        env._set_decision(text, {"generated_ids": None, "token_source": "solver_text"})
        return env.move(**dispatch_response(text)["tool_calls"][0]["function"]["arguments"])

    def test_reset_is_private_and_instances_are_independent(self):
        a, b = self.env(), self.env()
        first = a.reset(**dataset_row(self.layout))
        b.reset(**dataset_row(self.layout))
        self.assertEqual([p["type"] for p in first], ["image", "text"])
        self.assertNotIn(self.layout.layout_id, json.dumps(a._history))
        self.assertEqual(a._history[0]["content"], TOOL_PROMPT)
        self.act(a, move_call(shortest_path(self.layout)[0]))
        self.assertEqual(b._env.steps, 0)
        self.assertNotEqual(a._observations[0]["image"], b._observations[0]["image"])

    def test_invalid_response_consumes_turn_until_truncation(self):
        env = self.env()
        env.reset(**dataset_row(self.layout))
        for _ in range(8):
            feedback = self.act(env, "right")
            self.assertFalse(json.loads(feedback[0]["text"])["valid_action"])
        self.assertAlmostEqual(env.get_reward(), -.08)
        self.assertTrue(env._env.truncated)
        self.assertEqual(len(env._images), 9)
        self.assertEqual(env._history[2]["content"], "right")
        env.get_reward()
        self.assertEqual(len(self.episodes), 1)
        with self.assertRaises(RuntimeError):
            self.act(env, move_call("up"))

    def test_solver_replay_and_reset_clear_all_rollout_state(self):
        env = self.env()
        env.reset(**dataset_row(self.layout))
        original = env._images[0]
        for direction in shortest_path(self.layout): self.act(env, move_call(direction))
        self.assertGreater(env.get_reward(), 0)
        self.assertTrue(self.episodes[0]["success"])
        env.reset(**dataset_row(self.layout))
        self.assertEqual(env._env.steps, 0)
        self.assertEqual(len(env._observations), 1)
        self.assertNotIn(id(original), self.owners)
        self.assertEqual(env._transitions, [])
        with self.assertRaises(RuntimeError): env.get_reward()

    def test_dispatch_never_repairs_malformed_or_multiple_calls(self):
        for text in ("left", move_call("up")+move_call("right"), "explain "+move_call("up")):
            parsed = dispatch_response(text)
            self.assertEqual(parsed["content"], text)
            self.assertEqual(parsed["tool_calls"][0]["function"]["arguments"]["direction"], "__invalid__")
        self.assertEqual(dispatch_response(move_call("left"))["tool_calls"][0]["function"]["arguments"]["direction"], "left")

    def test_integration_config_is_bounded(self):
        self.assertEqual(Stage7Config().updates, 1)
        with self.assertRaises(ValueError): Stage7Config(updates=20)
        with self.assertRaises(ValueError): Stage7Config(train_layouts=1)
