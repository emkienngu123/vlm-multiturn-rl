"""Full-pass scheduling, recovery masks, and development-only comparisons."""

from copy import deepcopy
import random
import unittest

import test_browser
from vlm_multiturn_rl.stage14 import (FormSFTConfig, schedule, validate_training,
                                     validate_inspections, paired_comparison, development_metrics)
from vlm_multiturn_rl.stage13 import pools


class FormSFTTests(unittest.TestCase):
    def test_full_pass_includes_tail_once_without_changing_global_rng(self):
        config=FormSFTConfig();state=random.getstate();plan=schedule(config,853)
        self.assertEqual(state,random.getstate())
        self.assertEqual(len(plan),214)
        self.assertEqual(len(plan[-1]['example_indices']),1)
        self.assertEqual(sorted(i for r in plan for i in r['example_indices']),list(range(853)))
        self.assertEqual(plan,schedule(config,853))
        two=schedule(FormSFTConfig(epochs=2),853)
        self.assertEqual(len(two),428)
        for epoch in (1,2):
            indices=[i for r in two if r['epoch']==epoch for i in r['example_indices']]
            self.assertEqual(sorted(indices),list(range(853)))
        self.assertNotEqual(two[:214],two[214:])

    def training(self):
        config=FormSFTConfig();plan=schedule(config,5)
        rows=[{**r,'mean_action_loss':.1,'gradient_norm':1.,'seconds':.2,'target_tokens':8} for r in plan]
        return {'updates':rows,'completed_updates':2,'total_updates':2,'examples_processed':5,'dataset_passes':1.}

    def test_reordered_examples_bad_gradients_and_false_exposure_fail(self):
        config=FormSFTConfig();training=self.training();validate_training(config,training,5)
        for mutate in (lambda t:t['updates'][0]['example_indices'].reverse(),
                       lambda t:t.update(examples_processed=8),
                       lambda t:t.update(dataset_passes=2.),
                       lambda t:t['updates'][0].update(gradient_norm=0.),
                       lambda t:t['updates'][0].update(mean_action_loss=float('nan'))):
            changed=deepcopy(training);mutate(changed)
            with self.assertRaises(ValueError): validate_training(config,changed,5)

    def test_fixed_compute_settings_and_paths_fail_before_gpu(self):
        for kw in ({'epochs':0},{'epochs':True},{'epochs':3},{'gradient_accumulation':4.0},
                   {'learning_rate':2e-4},{'lora_rank':16},{'dataset_run_id':'../bad'}):
            with self.assertRaises(ValueError): FormSFTConfig(**kw)

    def test_recovery_history_is_masked_and_image_order_is_exact(self):
        messages=[{'role':'user','content':[{'type':'image','image':'initial.png'}]},
                  {'role':'assistant','content':'deliberately incorrect action'},
                  {'role':'tool','content':[{'type':'image','image':'after-mistake.png'}]}]
        example={'case_id':'train-recovery','turn':2,'messages':messages,'target':'corrective action'}
        inspection={'case_id':'train-recovery','turn':2,'target':'corrective action',
                    'supervised_span':[4,7],'input_ids':[90,91,92,93,10,11,99],
                    'labels':[-100,-100,-100,-100,10,11,99],'eos_token_id':99,
                    'prompt_tokens':4,'target_tokens':3,'image_paths':['initial.png','after-mistake.png']}
        validate_inspections([example],[inspection])
        for mutate in (lambda i:i['labels'].__setitem__(1,91),
                       lambda i:i['image_paths'].reverse(),
                       lambda i:i.update(target='deliberately incorrect action')):
            changed=deepcopy(inspection);mutate(changed)
            with self.assertRaises(ValueError): validate_inspections([example],[changed])

    def episodes(self):
        episodes=[]
        for policy in ('source-sft','forms-sft'):
            for case in pools(FormSFTConfig())['dev']:
                e=deepcopy(test_browser.BrowserTests().episode())
                e.update(case=case,policy=policy)
                e['observations'][0]['pixel_sha256']='same reset'
                episodes.append(e)
        return episodes

    def test_comparison_requires_same_sixteen_dev_instances_and_pixels(self):
        config=FormSFTConfig();episodes=self.episodes()
        self.assertEqual(paired_comparison(episodes,config)['paired_cases'],16)
        self.assertEqual(development_metrics(episodes)['forms-sft']['all']['episodes'],16)
        for mutate in (lambda e:e.pop(),lambda e:e.append(deepcopy(e[0])),
                       lambda e:e[-1]['observations'][0].update(pixel_sha256='different'),
                       lambda e:e[-1].update(case=pools(config)['final'][0])):
            changed=deepcopy(episodes);mutate(changed)
            with self.assertRaises(ValueError): paired_comparison(changed,config)
        wrong=deepcopy(episodes);wrong[0]['case']['split']='train'
        with self.assertRaises(ValueError):development_metrics(wrong)
