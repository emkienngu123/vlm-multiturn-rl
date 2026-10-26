"""CPU contracts for tool calls, history isolation, action targets, and loss masks."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from vlm_multiturn_rl.gridworld import Layout, shortest_path
from vlm_multiturn_rl.stage4 import Stage4Config, build_demonstrations, replay_saved_run, save_report
from vlm_multiturn_rl.trajectories import (
    assistant_labels, collect_trajectory, demonstration_examples, image_paths,
    move_call, parse_move, replay_trajectory,
)


DETOUR = Layout((0, 0), (0, 2), ((0, 1), (1, 1), (3, 3)))


class TrajectoryTests(unittest.TestCase):
    def collect(self, actor=None, assets=None, key="test"):
        return collect_trajectory(
            DETOUR, "vlm" if actor else "oracle", key=key, assets={} if assets is None else assets,
            renderer=lambda layout, position: tuple(position), extension="fake", actor=actor,
        )

    def test_only_one_well_formed_move_is_accepted(self):
        for direction in ("up", "down", "left", "right"):
            self.assertEqual(parse_move(move_call(direction)), direction)
        malformed = ["left", "move(left)", move_call("left") * 2, "Explanation: " + move_call("left"),
                     move_call("left").replace('"left"', '"LEFT"'),
                     move_call("left").replace('"move"', '"teleport"'),
                     move_call("left").replace('"direction":"left"', '"direction":"left","x":1'),
                     move_call("left").replace('"direction":"left"', '"direction":["left"]'),
                     move_call("left").replace('"direction":"left"', '"direction":"up","direction":"left"'),
                     '<tool_call>[]</tool_call>', '<tool_call>broken</tool_call>']
        for text in malformed:
            with self.subTest(text=text):
                self.assertIsNone(parse_move(text))

    def test_newest_observation_and_full_history_reach_the_actor_without_state(self):
        actions = iter(shortest_path(DETOUR))
        received = []

        def actor(messages, images):
            self.assertEqual(len(images), len(received) + 1)
            self.assertEqual(len(image_paths(messages)), len(images))
            self.assertEqual(images[-1], expected_positions[len(received)])
            self.assertEqual([m["role"] for m in messages],
                             ["system", "user"] + ["assistant", "tool"] * len(received))
            self.assertNotIn("position", json.dumps(messages))
            self.assertNotIn(DETOUR.layout_id, json.dumps(messages))
            received.append(deepcopy(messages))
            return {"text": move_call(next(actions)), "generated_ids": [151657, 42, 151645]}

        expected_positions = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)]
        episode = self.collect(actor)
        self.assertTrue(episode["success"])
        self.assertEqual(len(received), 6)
        self.assertEqual(episode["observations"][-1]["position"], list(DETOUR.goal))
        self.assertEqual(episode["transitions"][0]["generation"]["generated_ids"], [151657, 42, 151645])
        replay_trajectory(DETOUR, json.loads(json.dumps(episode)))

    def test_malformed_actions_remain_in_history_and_consume_turns(self):
        episode = self.collect(lambda messages, images: {"text": "left", "generated_ids": [99]})
        self.assertEqual(episode["invalid_actions"], 8)
        self.assertEqual(episode["end_reason"], "step_limit")
        self.assertAlmostEqual(episode["return"], -0.08)
        self.assertEqual(episode["transitions"][1]["messages"][2], {"role": "assistant", "content": "left"})
        replay_trajectory(DETOUR, episode)

    def test_histories_reset_and_do_not_alias_previous_turns(self):
        assets = {}
        first = self.collect(assets=assets, key="first")
        second = self.collect(assets=assets, key="second")
        self.assertEqual(len(second["transitions"][0]["messages"]), 2)
        self.assertFalse(set(image_paths(first["messages"])) & set(image_paths(second["messages"])))
        first["transitions"][0]["messages"].append({"role": "assistant", "content": "mutated"})
        self.assertEqual(len(first["transitions"][1]["messages"]), 4)

    def test_replay_rejects_stale_images_changed_feedback_and_missing_final_observation(self):
        original = self.collect()
        edits = [lambda e: e["transitions"][1]["messages"][3]["content"][1].update(image="stale.png"),
                 lambda e: e["transitions"][1]["messages"][3]["content"][0].update(text="goal reached"),
                 lambda e: e["observations"].pop(),
                 lambda e: e["messages"].pop()]
        for edit in edits:
            changed = deepcopy(original)
            edit(changed)
            with self.assertRaises(ValueError):
                replay_trajectory(DETOUR, changed)

    def test_targets_are_current_actions_and_never_future_feedback(self):
        episode = self.collect()
        examples = demonstration_examples([episode])
        self.assertEqual(len(examples), 6)
        for example, record in zip(examples, episode["transitions"]):
            self.assertEqual(example["messages"], record["messages"])
            self.assertEqual(example["target"], record["text"])
            self.assertEqual(len(image_paths(example["messages"])), record["step"])
            self.assertEqual(example["messages"][-1]["role"], "user" if record["step"] == 1 else "tool")

    def test_mask_and_causal_alignment_include_first_action_token_and_eos(self):
        # Prefix stands for text, image tokens, previous assistant action, and tool result.
        prefix, target = [1, 999, 999, 2, 3, 4], [51, 52, 151645]
        ids, labels = assistant_labels(prefix, target, image_token_id=999)
        self.assertEqual(ids, prefix + target)
        self.assertEqual(labels[:len(prefix)], [-100] * len(prefix))
        self.assertEqual(labels[len(prefix):], target)
        scored = [(position, label) for position, label in enumerate(labels[1:]) if label != -100]
        self.assertEqual(scored, [(5, 51), (6, 52), (7, 151645)])
        for prompt, action in [([], target), (prefix, []), (prefix, [999])]:
            with self.assertRaises(ValueError):
                assistant_labels(prompt, action, 999)

    def test_cpu_report_round_trip_and_heldout_training_contract(self):
        config = Stage4Config(train_layouts=2, episodes=2)
        assets = {}
        data = build_demonstrations(config, assets)
        train_ids = {x["layout_id"] for x in data["layouts"]["train"]}
        self.assertEqual({e["layout_id"] for e in data["examples"]}, train_ids)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "stage4-test"
            save_report(directory, config, data, {"mode": "cpu"}, {}, assets)
            self.assertEqual(replay_saved_run(directory), 2)

    def test_training_limits_are_validated_before_gpu_allocation(self):
        for kwargs in [{"updates": 0}, {"updates": 21}, {"gradient_accumulation": 0},
                       {"train_layouts": 33}, {"max_new_tokens": 16}, {"learning_rate": float("nan")}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Stage4Config(**kwargs)


if __name__ == "__main__":
    unittest.main()
