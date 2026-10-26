"""CPU checks for dynamics, task isolation, policy boundaries, and replay."""

from copy import deepcopy
import random
import unittest

from vlm_multiturn_rl.gridworld import (
    GridWorld, Layout, make_layouts, replay_episode, run_episode, shortest_path, summarize,
)
from vlm_multiturn_rl.stage3 import Stage3Config, replay_html


DETOUR = Layout((0, 0), (0, 2), ((0, 1), (1, 1), (3, 3)))


class GridWorldTests(unittest.TestCase):
    def test_bfs_and_rewards_on_a_known_detour(self):
        actions = ["down", "down", "right", "right", "up", "up"]
        self.assertEqual(shortest_path(DETOUR), actions)
        env = GridWorld(DETOUR)
        rewards = []
        for action in actions:
            outcome = env.step(action)
            rewards.append(outcome.reward)
        self.assertEqual(env.position, DETOUR.goal)
        self.assertTrue(outcome.terminated)
        self.assertFalse(outcome.truncated)
        self.assertAlmostEqual(sum(rewards), 0.94)
        self.assertEqual(rewards[:-1], [-0.01] * 5)
        self.assertEqual(rewards[-1], 0.99)
        self.assertEqual(shortest_path(DETOUR, DETOUR.goal), [])

    def test_wall_boundary_and_malformed_moves_consume_budget(self):
        env = GridWorld(DETOUR)
        for index, (action, valid, blocked) in enumerate([
            ("up", True, True), ("right", True, True), (None, False, False),
            ("left or down", False, False),
        ], start=1):
            outcome = env.step(action)
            self.assertEqual(env.position, DETOUR.start)
            self.assertEqual(env.steps, index)
            self.assertEqual(outcome.reward, -0.01)
            self.assertEqual(outcome.valid_action, valid)
            self.assertEqual(outcome.blocked, blocked)

    def test_truncation_goal_on_last_step_and_reset(self):
        layout = Layout((0, 0), (0, 1))
        env = GridWorld(layout, max_steps=1)
        limit = env.step("down")
        self.assertFalse(limit.terminated)
        self.assertTrue(limit.truncated)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            env.step("right")
        env.reset()
        self.assertEqual(env.position, layout.start)
        self.assertEqual(env.steps, 0)
        goal = env.step("right")
        self.assertTrue(goal.terminated)
        self.assertFalse(goal.truncated)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            env.step("down")

    def test_unreachable_and_invalid_layouts_are_rejected(self):
        unreachable = Layout((0, 0), (3, 3), ((0, 1), (1, 0)))
        self.assertIsNone(shortest_path(unreachable))
        with self.assertRaisesRegex(ValueError, "unreachable"):
            GridWorld(unreachable)
        for start, goal, walls in [((0, 0), (0, 0), ()), ((-1, 0), (1, 1), ()),
                                   ((0, 0), (1, 1), ((0, 0),)),
                                   ((0, 0), (1, 1), ((0, 1), (0, 1)))]:
            with self.subTest(start=start, goal=goal, walls=walls), self.assertRaises(ValueError):
                Layout(start, goal, walls)

    def test_generation_is_deterministic_and_does_not_use_global_rng(self):
        state = random.getstate()
        first = make_layouts("eval", 4)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(first, make_layouts("eval", 4))
        self.assertEqual(first, make_layouts("eval", 8)[:4])
        self.assertEqual(Layout.from_dict(first[0].to_dict()), first[0])
        self.assertEqual(DETOUR.layout_id, Layout(DETOUR.start, DETOUR.goal, tuple(reversed(DETOUR.walls))).layout_id)

    def test_maze_splits_are_disjoint_across_dataset_seeds(self):
        train = [*make_layouts("train", 32, 2026), *make_layouts("train", 32, 7)]
        evaluation = [*make_layouts("eval", 32, 2026), *make_layouts("eval", 32, 7)]
        self.assertFalse({x.maze_id for x in train} & {x.maze_id for x in evaluation})
        for layout in (*train, *evaluation):
            self.assertLessEqual(len(shortest_path(layout)), 6)
            self.assertGreaterEqual(len(shortest_path(layout)), 2)

    def test_oracle_solves_every_task_and_random_is_reproducible(self):
        tasks = make_layouts("eval", 16)
        rows = [run_episode(x, "oracle") for x in tasks]
        for layout, episode in zip(tasks, rows):
            replay_episode(layout, episode)
            self.assertEqual(episode["steps"], len(shortest_path(layout)))
        metrics = summarize(rows)
        self.assertEqual(metrics["success_rate"], 1.0)
        self.assertEqual(metrics["mean_path_efficiency"], 1.0)
        self.assertEqual(metrics["blocked_move_rate"], 0.0)
        self.assertEqual(run_episode(tasks[0], "random", seed=7), run_episode(tasks[0], "random", seed=7))

    def test_replay_rejects_changed_actions_rewards_states_and_summaries(self):
        episode = run_episode(DETOUR, "oracle")
        edits = [lambda e: e["transitions"][0].update(text="up"),
                 lambda e: e["transitions"][0].update(reward=1.0),
                 lambda e: e["transitions"][0].update(after=[3, 3]),
                 lambda e: e.update(success=False),
                 lambda e: e["transitions"].pop()]
        for edit in edits:
            changed = deepcopy(episode)
            edit(changed)
            with self.assertRaises(ValueError):
                replay_episode(DETOUR, changed)

    def test_vlm_actor_receives_only_the_renderer_output(self):
        actions = iter(shortest_path(DETOUR))
        image = object()
        received = []

        def actor(observation):
            self.assertIs(observation, image)
            received.append(observation)
            return {"text": next(actions)}

        episode = run_episode(DETOUR, "vlm", actor=actor, renderer=lambda layout, position: image)
        self.assertTrue(episode["success"])
        self.assertEqual(len(received), 6)
        replay_episode(DETOUR, episode)

    def test_report_escapes_model_text_and_keeps_invalid_actions(self):
        text = "<script>alert(1)</script>"
        episode = run_episode(DETOUR, "vlm", actor=lambda image: {"text": text},
                              renderer=lambda layout, position: object())
        self.assertEqual(episode["invalid_actions"], 8)
        self.assertFalse(episode["success"])
        report = replay_html({"layouts": {"eval": [DETOUR.to_dict()]},
                              "episodes": [episode], "metrics": {"vlm": summarize([episode])}})
        self.assertNotIn(text, report)
        self.assertIn("&lt;script&gt;", report)

    def test_stage3_bounds_are_checked_before_launch(self):
        for kwargs in [{"episodes": 0}, {"episodes": 17}, {"max_steps": 9}, {"dataset_seed": -1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Stage3Config(**kwargs)


if __name__ == "__main__":
    unittest.main()
