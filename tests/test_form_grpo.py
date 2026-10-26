"""GRPO schedule and continuation semantics, without local training libraries."""
from copy import deepcopy
import unittest

from vlm_multiturn_rl.grpo import relative_advantages
from vlm_multiturn_rl.stage10 import trajectory_seed
from vlm_multiturn_rl.stage16 import FormGRPOConfig, plan, validate_history


class FormGRPOTests(unittest.TestCase):
    def rows(self):
        c = FormGRPOConfig(); rows = []; fp = 'initial'; steps = 0
        for index, returns in enumerate(([1., 0., 0., 0.], [0.]*4)):
            mixed = len(set(returns)) > 1
            steps += int(mixed); after = f'updated-{index}' if mixed else fp
            group = {**relative_advantages(returns), 'case_id': plan(c)['round_case_ids'][index],
                'behavior_fingerprint': fp, 'trajectory_seeds': [trajectory_seed(c, 8+index, m) for m in range(4)],
                'max_ratio_deviation': 0.}
            rows.append({'round': index+1, 'group': group, 'optimizer_steps': steps, 'seconds': 20.,
                'update': {'optimizer_step': mixed, 'after_fingerprint': after, 'gradient_norm': 1. if mixed else 0.,
                           'max_ratio_deviation': 0., 'loss': 0., 'clipped_fraction': 0.}})
            fp = after
        return c, rows

    def test_pool_schedule_and_rng_are_predeclared(self):
        c = FormGRPOConfig(); p = plan(c)
        self.assertEqual(len(p['train']), 10); self.assertEqual(len(p['dev']), 16)
        self.assertEqual(len(p['round_case_ids']), 20)
        self.assertEqual(p['round_case_ids'][:2], ['forms16-train-000']*2)
        self.assertEqual(set(p['round_case_ids']), {c['case_id'] for c in p['train']})
        train = {c['seed'] for c in p['train']}; dev = {c['seed'] for c in p['dev']}
        self.assertFalse(train & dev); self.assertFalse(train & set(range(70000, 70032)))
        streams = {trajectory_seed(c, 8+r, m) for r in range(20) for m in range(4)}
        old = {trajectory_seed(c, r, m) for r in range(8) for m in range(4)}
        self.assertEqual(len(streams), 80); self.assertFalse(streams & old)

    def test_round_count_is_distinct_from_optimizer_steps(self):
        c, rows = self.rows()
        fp, steps = validate_history(c, rows, 'initial')
        self.assertEqual(steps, 1); self.assertEqual(fp, 'updated-0'); self.assertEqual(len(rows), 2)

    def test_resume_history_rejects_stale_weights_advantages_seeds_and_counter(self):
        c, rows = self.rows()
        for mutation in ('weights', 'advantages', 'seed', 'counter', 'case', 'ratio', 'gradient', 'skip'):
            changed = deepcopy(rows)
            if mutation == 'weights': changed[1]['group']['behavior_fingerprint'] = 'initial'
            if mutation == 'advantages': changed[0]['group']['advantages'][0] = 0.
            if mutation == 'seed': changed[0]['group']['trajectory_seeds'][0] += 1
            if mutation == 'counter': changed[1]['optimizer_steps'] = 2
            if mutation == 'case': changed[0]['group']['case_id'] = 'forms13-dev-000'
            if mutation == 'ratio': changed[0]['group']['max_ratio_deviation'] = .001
            if mutation == 'gradient': changed[0]['update']['gradient_norm'] = 0.
            if mutation == 'skip': changed[1]['update']['after_fingerprint'] = 'different'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_history(c, changed, 'initial')

    def test_compute_and_probability_guards_cannot_be_silently_changed(self):
        for kw in ({'max_rounds': 100}, {'training_seconds': 9999}, {'ratio_tolerance': .1},
                   {'kl_beta': .1}, {'group_size': 2}, {'learning_rate': .001}, {'sft_run_id': '../bad'}):
            with self.assertRaises(ValueError): FormGRPOConfig(**kw)
