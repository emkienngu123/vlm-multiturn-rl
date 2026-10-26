"""Browser SFT split, loss-boundary, and paired-comparison contracts, without Torch."""

from copy import deepcopy
from dataclasses import asdict
import unittest

from vlm_multiturn_rl.browser import BrowserConfig, cases
from vlm_multiturn_rl.stage8 import demonstrations
from vlm_multiturn_rl.stage9 import (BrowserSFTConfig, browser_config, paired_comparison,
                                   training_rollout_metrics, validate_demonstrations, validate_mask)
import test_browser


class BrowserSFTTests(unittest.TestCase):
    def test_training_settings_do_not_change_environment_contract(self):
        self.assertEqual(asdict(browser_config(BrowserSFTConfig(updates=20))), asdict(BrowserConfig()))
        self.assertEqual(asdict(browser_config(BrowserSFTConfig(updates=100))), asdict(BrowserConfig()))
        for kwargs in ({'updates':0}, {'updates':101}, {'updates':True}, {'gradient_accumulation':5},
                       {'learning_rate':float('nan')}, {'lora_rank':16}, {'diagnostic_sft_run_id':'../bad'}):
            with self.assertRaises(ValueError): BrowserSFTConfig(**kwargs)

    def test_loss_mask_excludes_observations_and_prior_assistant_turns(self):
        inspection = {'supervised_span':[4,7], 'input_ids':[90,91,92,93,10,11,99],
                      'labels':[-100,-100,-100,-100,10,11,99], 'eos_token_id':99,
                      'prompt_tokens':4, 'target_tokens':3, 'turn':2,
                      'image_paths':['initial.png','after-action.png']}
        validate_mask(inspection)
        for mutate in (lambda x: x['labels'].__setitem__(2,92),
                       lambda x: x['labels'].__setitem__(4,-100),
                       lambda x: x['image_paths'].pop(),
                       lambda x: x.__setitem__('eos_token_id',98)):
            changed = deepcopy(inspection); mutate(changed)
            with self.assertRaises(ValueError): validate_mask(changed)

    def test_demo_list_must_be_exactly_training_oracle_targets(self):
        episode = test_browser.BrowserTests().episode()
        examples = demonstrations([episode])
        validate_demonstrations(cases(BrowserConfig()), [episode], examples)
        changed = deepcopy(examples); changed[0]['target'] = 'click(1,2)'
        with self.assertRaises(ValueError): validate_demonstrations(cases(BrowserConfig()), [episode], changed)
        changed = deepcopy(episode); changed['case']['split'] = 'eval'
        with self.assertRaises(ValueError): validate_demonstrations([changed['case']], [changed], examples)

    def test_pairing_requires_same_instances_and_initial_pixels(self):
        base = test_browser.BrowserTests().episode(); base['case'] = cases(BrowserConfig())[6]; base['policy'] = 'base'
        base['observations'][0]['pixel_sha256'] = 'pixels'
        sft = deepcopy(base); sft['policy'] = 'sft'; sft['observations'][0]['image'] = 'sft.png'
        self.assertEqual(paired_comparison([base,sft])['paired_cases'],1)
        for changes in ('pixels','missing','duplicate'):
            changed = deepcopy(sft)
            if changes == 'pixels': changed['observations'][0]['pixel_sha256'] = 'different'
            group = [base] if changes == 'missing' else [base,changed]
            if changes == 'duplicate': group.append(changed)
            with self.assertRaises(ValueError): paired_comparison(group)

    def test_training_fit_is_separate_from_evaluation_and_demonstrations(self):
        reference = test_browser.BrowserTests().episode(); reference['policy'] = 'sft_reference'
        current = deepcopy(reference); current['policy'] = 'sft'
        evaluation = deepcopy(current); evaluation['case']['split'] = 'eval'; evaluation['success'] = False
        oracle = deepcopy(current); oracle['policy'] = 'oracle'
        metrics = training_rollout_metrics([reference,current,evaluation,oracle])
        self.assertEqual(set(metrics), {'sft_reference','sft'})
        self.assertEqual(metrics['sft']['all']['episodes'],1)
        self.assertEqual(metrics['sft']['all']['successes'],1)
        self.assertEqual(len(demonstrations([reference,current,evaluation,oracle])),1)
