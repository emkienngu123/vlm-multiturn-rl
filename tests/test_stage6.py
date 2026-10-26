"""CPU checks for resume contracts, paired outcomes, and report consistency."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from vlm_multiturn_rl import checkpoints
from vlm_multiturn_rl.grpo import relative_advantages
from vlm_multiturn_rl.gridworld import make_layouts, render_svg
from vlm_multiturn_rl.reporting import compare_policies, wilson_interval
from vlm_multiturn_rl.stage6 import Stage6Config, find_checkpoint, validate_history
from vlm_multiturn_rl.trajectories import collect_trajectory
from vlm_multiturn_rl.workspace import write_manifest


class Stage6Tests(unittest.TestCase):
    def test_resume_allows_target_extension_but_rejects_changed_training_or_evaluation(self):
        previous = asdict(Stage6Config(updates=2))
        current = asdict(Stage6Config(updates=4, resume_run_id="stage6-parent"))
        checkpoints.validate_resume_config(previous, current, 2)
        for field, value in (("seed", 7), ("max_pixels", 50176), ("learning_rate", 2e-5),
                             ("episodes", 8), ("dataset_seed", 123), ("sft_run_id", "different")):
            changed = {**current, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                checkpoints.validate_resume_config(previous, changed, 2)
        with self.assertRaises(ValueError):
            checkpoints.validate_resume_config(previous, previous, 2)

    def test_latest_checkpoint_ignores_an_incomplete_round_and_needs_no_final_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "stage6-parent" / "checkpoints"
            for index in (1, 2, 3):
                path = parent / f"round-{index:04d}"
                path.mkdir(parents=True)
                for name in checkpoints.FILES:
                    (path / name).write_text(json.dumps({"completed_round": index}))
                if index < 3:
                    write_manifest(path, {"kind": "checkpoint", "protocol": "move-json-v1"}, stage=6, artifact_files=checkpoints.FILES)
            config = Stage6Config(updates=4, resume_run_id="stage6-parent")
            self.assertEqual(find_checkpoint(config, root).name, "round-0002")
            selected = find_checkpoint(Stage6Config(updates=4, resume_run_id="stage6-parent", resume_round=1), root)
            self.assertEqual(selected.name, "round-0001")
            (selected / "training-state.pt").write_text("corrupted")
            with self.assertRaises(ValueError):
                checkpoints.metadata(selected)

    def test_wilson_intervals_express_small_sample_uncertainty(self):
        lo, hi = wilson_interval(1, 4)
        self.assertAlmostEqual(lo, .0455872608)
        self.assertAlmostEqual(hi, .6993581574)
        self.assertAlmostEqual(wilson_interval(0, 4)[0], 0)
        self.assertAlmostEqual(wilson_interval(4, 4)[1], 1)
        self.assertLess(wilson_interval(25, 100)[1]-wilson_interval(25, 100)[0], hi-lo)
        for successes, total in ((0, 0), (5, 4), (-1, 4)):
            with self.assertRaises(ValueError):
                wilson_interval(successes, total)

    def test_paired_outcomes_detect_changed_boards_despite_equal_success_rates(self):
        layouts = [{"layout_id": str(i)} for i in range(4)]
        episodes = []
        for policy, good_board in (("base", None), ("sft", "0"), ("grpo", "1")):
            for layout in layouts:
                success = layout["layout_id"] == good_board
                episodes.append({"policy": policy, "layout_id": layout["layout_id"], "success": success,
                                 "steps": 2 if success else 8, "return": .98 if success else -.08,
                                 "path_efficiency": 1 if success else 0, "invalid_actions": 0, "blocked_moves": 0})
        report = compare_policies(episodes, {x: x for x in ("base", "sft", "grpo")}, layouts)
        self.assertEqual(report["grpo_vs_sft"], {"wins": 1, "losses": 1, "ties": 2, "success_rate_delta": 0})
        self.assertEqual(report["policies"]["sft"]["success_rate"], .25)
        with self.assertRaises(ValueError):
            compare_policies(episodes[:-1], {x: x for x in ("base", "sft", "grpo")}, layouts)

    def test_history_replay_checks_round_advantages_skips_and_membership(self):
        layout = make_layouts("train", 1)[0]
        episodes, assets = [], {}
        for index in range(4):
            e = collect_trajectory(layout, "oracle", key=f"fixture-{index}", assets=assets,
                                   renderer=render_svg, extension="svg", seed=index)
            e["policy"] = "rollout"
            for record in e["transitions"]:
                record["generation"].update(generated_ids=[42], old_log_probs=[0.0], behavior_fingerprint="same")
            episodes.append(e)
        group = {**relative_advantages([e["return"] for e in episodes]), "round": 1,
                 "layout_id": layout.layout_id, "episode_indices": [0, 1, 2, 3],
                 "behavior_fingerprint": "same", "update": {"optimizer_step": False, "after_fingerprint": "same"}}
        history = {"layouts": {"train": [layout.to_dict()], "eval": []}, "episodes": episodes, "groups": [group]}
        validate_history(history)
        edits = [lambda h: h["groups"][0].update(round=2),
                 lambda h: h["groups"][0]["advantages"].__setitem__(0, 1),
                 lambda h: h["groups"][0]["update"].update(optimizer_step=True),
                 lambda h: h["groups"][0].update(episode_indices=[0, 1, 2, 2])]
        for edit in edits:
            changed = deepcopy(history)
            edit(changed)
            with self.assertRaises(ValueError):
                validate_history(changed)

    def test_stage6_limits_reject_bad_resume_and_intervals_before_gpu_allocation(self):
        for kwargs in ({"checkpoint_every": 0}, {"eval_every": 21}, {"resume_round": 1},
                       {"resume_run_id": "../escape"}, {"verify_resume": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Stage6Config(**kwargs)


if __name__ == "__main__":
    unittest.main()
