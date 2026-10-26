"""Train-only pool, checkpoint and strict on-policy probability contracts; no Torch."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import test_browser_difficulty
from vlm_multiturn_rl.stage10 import trajectory_seed
from vlm_multiturn_rl.stage15 import (ExplorationConfig, GPU_FILES, readiness, replay_saved_run,
                                     source_receipts, strict_group, study_plan, validate_shard)
from vlm_multiturn_rl.workspace import sha256_file, write_json, write_manifest
from vlm_multiturn_rl.browser import browser_prompt


class ExplorationTests(unittest.TestCase):
    def fixture(self):
        config = ExplorationConfig(sft_run_id='final', warmup_run_id='early')
        plan = study_plan(config); shard = plan['shards'][2]; case = plan['cases'][2]
        episodes = []
        for i, reward in enumerate((1., 0., 1., 0.)):
            e = test_browser_difficulty.DifficultyTests().episode(case, f'sample-final-2-{i}', reward)
            e['sampling_seed'] = trajectory_seed(config, 2, i)
            e['transitions'][0]['generation'] = {
                'prompt_ids': [1, 2], 'generated_ids': [7, 8], 'sampled_log_probs': [-.1, -.2],
                'old_log_probs': [-.1, -.2], 'behavior_fingerprint': 'fp', 'policy_scoring': 'grad-cache-v1',
                'image_paths': [e['observations'][0]['image']], 'action_loss_span': [2, 4]}
            episodes.append(e)
        return config, shard, episodes, [strict_group(config, episodes, 'fp')]

    def test_fixed_study_is_disjoint_and_has_unique_rng_streams(self):
        c = ExplorationConfig(); p = study_plan(c)
        self.assertEqual(len(p['cases']), 6); self.assertEqual(len(p['shards']), 8)
        self.assertTrue(all(e['split'] == 'train' and e['seed'] in (80000, 80001) for e in p['cases']))
        self.assertEqual([s['candidate'] for s in p['shards']], ['final']*6+['early']*2)
        streams = [trajectory_seed(c, g, m) for g in range(8) for m in range(4)]
        self.assertEqual(len(set(streams)), 32)
        for kw in ({'episodes': 1}, {'ratio_tolerance': .1}, {'sft_run_id': '../bad'},
                   {'sft_run_id': 'same', 'warmup_run_id': 'same'}):
            with self.assertRaises(ValueError): ExplorationConfig(**kw)

    def test_strict_ratio_catches_drift_hidden_by_older_logp_tolerance(self):
        c, shard, episodes, groups = self.fixture()
        validate_shard(c, shard, episodes, groups, 'fp')
        episodes[0]['transitions'][0]['generation']['old_log_probs'][0] += .001
        with self.assertRaisesRegex(ValueError, 'ratio'): strict_group(c, episodes, 'fp')

    def test_shards_reject_rng_checkpoint_eval_and_score_changes(self):
        c, shard, episodes, groups = self.fixture()
        for mutation in ('seed', 'eval', 'candidate', 'scores', 'count'):
            e = deepcopy(episodes); g = deepcopy(groups)
            if mutation == 'seed': e[0]['sampling_seed'] += 1
            if mutation == 'eval':
                for row in e: row['case'] = {**row['case'], 'split': 'eval'}
            if mutation == 'candidate': e[0]['policy'] = 'sample-early-2-0'
            if mutation == 'scores': g[0]['advantages'][0] = 0.
            if mutation == 'count': e.pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_shard(c, shard, e, g, 'fp')

    def test_report_separates_mixed_groups_from_all_success_and_failure(self):
        c, shard, episodes, groups = self.fixture()
        self.assertTrue(readiness(episodes, groups)['ready_for_nonzero_grpo_signal'])
        for reward in (0., 1.):
            for e in episodes: e['return'] = reward; e['success'] = bool(reward)
            r = readiness(episodes, [strict_group(c, episodes, 'fp')])
            self.assertFalse(r['ready_for_nonzero_grpo_signal'])
            self.assertEqual(r['optimizer_steps'], 0)
            self.assertEqual(r['within_group_reward_variances'], [0.])

    def test_reader_recomputes_semantics_even_after_rehash(self):
        c, shard, episodes, groups = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)/'probe'; d.mkdir()
            for name in GPU_FILES: (d/name).write_bytes(b'fixture')
            sources = {k: {'adapter_fingerprint': 'fp', 'requirements_sha256': sha256_file(d/'requirements-browser.txt')}
                       for k in ('final', 'early')}
            objects = {'config.json': asdict(c), 'sources.json': sources, 'plan.json': study_plan(c),
                       'episodes.json': episodes, 'checks.json': {'verified': True}, 'groups.json': groups,
                       'readiness.json': readiness(episodes, groups),
                       'environment.json': {'shard': shard, 'optimizer_steps': 0, 'before_fingerprint': 'fp',
                                            'after_fingerprint': 'fp', 'policy_scoring': 'grad-cache-v1'}}
            for name, value in objects.items(): write_json(d/name, value)
            (d/'policy-prompt.txt').write_text(browser_prompt(12)+'\n')
            with zipfile.ZipFile(d/'images.zip', 'w') as z:
                for o in episodes[0]['observations']: z.writestr(o['image'], b'image')
            def seal(): write_manifest(d, {'mode': 'gpu'}, stage=15, artifact_files=GPU_FILES)
            seal()
            with patch('vlm_multiturn_rl.stage15.source_receipts', return_value=sources):
                self.assertEqual(replay_saved_run(d), 4)
                objects['environment.json']['after_fingerprint'] = 'changed'
                write_json(d/'environment.json', objects['environment.json']); seal()
                with self.assertRaisesRegex(ValueError, 'unchanged'): replay_saved_run(d)

    def test_missing_source_and_wrong_checkpoint_stage_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Supply both'): source_receipts(ExplorationConfig(), Path(tmp))
            with self.assertRaises((FileNotFoundError, ValueError)):
                source_receipts(ExplorationConfig(sft_run_id='final', warmup_run_id='early'), Path(tmp))
