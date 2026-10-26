"""Browser evaluation separation and frozen-policy sampling contracts, no Torch."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import test_browser

from vlm_multiturn_rl.browser import initial_history
from vlm_multiturn_rl.stage10 import (BrowserReadinessConfig, build_cases, group_record,
                                     GPU_FILES, readiness_report, replay_saved_run, trajectory_seed)
from vlm_multiturn_rl.stage10 import _write_progress
from vlm_multiturn_rl.browser import summarize
from vlm_multiturn_rl.workspace import write_json, write_manifest


class BrowserReadinessTests(unittest.TestCase):
    def test_new_instances_and_old_training_pools_are_disjoint(self):
        config=BrowserReadinessConfig()
        cases,pools=build_cases(config)
        self.assertEqual(len(cases),30)
        self.assertEqual(len(pools['fresh_eval']),24)
        pairs=lambda pool:{(c['task'],c['seed']) for c in cases if c['case_id'] in pools[pool]}
        self.assertFalse(pairs('seen_train') & pairs('new_train'))
        self.assertFalse((pairs('seen_train') | pairs('new_train')) & pairs('fresh_eval'))
        self.assertTrue(all(c['seed']>=20000 and c['split']=='eval' for c in cases if c['case_id'] in pools['fresh_eval']))
        for kwargs in ({'fresh_cases':0},{'fresh_cases':17},{'probe_cases':3},
                       {'probe_cases':2,'episodes':1},{'group_size':2},{'sft_run_id':'../bad'}):
            with self.assertRaises(ValueError):BrowserReadinessConfig(**kwargs)

    def members(self,returns,pool='seen_train'):
        config=BrowserReadinessConfig()
        cases,pools=build_cases(config)
        case=next(c for c in cases if c['case_id']==pools[pool][0])
        result=[]
        for i,reward in enumerate(returns):
            observation={'goal':'Click the button.', 'image':f'images/{pool}-{i}.png',
                         'width':498,'height':321,'pixel_sha256':'identical pixels'}
            generation={'prompt_ids':[1,2,3], 'generated_ids':[7,8], 'sampled_log_probs':[-.1,-.2],
                        'old_log_probs':[-.1,-.2], 'behavior_fingerprint':'frozen',
                        'image_paths':[observation['image']], 'action_loss_span':[3,5]}
            result.append({'case':case,'max_steps':8,'observations':[observation],
                'transitions':[{'messages':initial_history(observation),'step':1,'generation':generation}],
                'sampling_seed':trajectory_seed(config,0,i),'return':reward,'episode_id':f'{pool}-{i}',
                'policy':f'sample-{pool}-{i}', 'success':bool(reward),'steps':1,
                'invalid_actions':0,'execution_errors':0})
        return result

    def test_mixed_and_zero_variance_groups_preserve_actual_returns(self):
        config=BrowserReadinessConfig()
        members=self.members([1.,0.,1.,0.])
        group=group_record(config,'seen_train',members,'frozen')
        self.assertFalse(group['zero_variance'])
        self.assertEqual(group['mean'],.5)
        self.assertEqual(group['std'],.5)
        self.assertGreater(group['advantages'][0],0)
        self.assertLess(group['advantages'][1],0)
        for reward in (0.,1.):
            group=group_record(config,'seen_train',self.members([reward]*4),'frozen')
            self.assertTrue(group['zero_variance'])
            self.assertEqual(group['advantages'],[0.]*4)

    def test_batched_drift_is_diagnostic_but_actual_behavior_scores_remain_checked(self):
        members=self.members([1.]*4)
        for e in members:
            g=e['transitions'][0]['generation']
            g['batched_diagnostic_log_probs']=[p-.430226 for p in g['sampled_log_probs']]
        group=group_record(BrowserReadinessConfig(),'seen_train',members,'frozen')
        self.assertEqual(group['max_sampling_scoring_logp_delta'],0.)
        self.assertAlmostEqual(group['max_batched_diagnostic_logp_delta'],.430226)
        members[0]['transitions'][0]['generation']['old_log_probs'][0]-=.430226
        with self.assertRaisesRegex(ValueError,'worst='):
            group_record(BrowserReadinessConfig(),'seen_train',members,'frozen')

    def test_failure_progress_is_retained_without_a_completion_manifest(self):
        episode=test_browser.BrowserTests().episode()
        data={'episodes':[episode]}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with patch('vlm_multiturn_rl.stage10.RUNS_PATH',root),patch('vlm_multiturn_rl.stage10._images_zip',return_value=b'images'):
                _write_progress('partial',BrowserReadinessConfig(),data,{},[],error='scoring failed')
            directory=root/'partial'
            self.assertTrue((directory/'partial-episodes.json').is_file())
            self.assertIn('scoring failed',(directory/'progress.json').read_text())
            self.assertFalse((directory/'manifest.json').exists())

    def test_group_rejects_changed_states_rng_weights_and_scoring(self):
        members=self.members([1.,0.,1.,0.])
        for mutation in ('state','rng','weights','span','delta','image_order','length','eval'):
            group=deepcopy(members)
            g=group[1]['transitions'][0]['generation']
            if mutation=='state':group[1]['observations'][0]['pixel_sha256']='different'
            if mutation=='rng':group[1]['sampling_seed']=group[0]['sampling_seed']
            if mutation=='weights':g['behavior_fingerprint']='changed'
            if mutation=='span':g['action_loss_span']=[2,5]
            if mutation=='delta':g['old_log_probs'][0]=-1.
            if mutation=='image_order':g['image_paths']=['stale.png']
            if mutation=='length':g['old_log_probs'].pop()
            if mutation=='eval':
                for e in group:e['case']={**e['case'],'split':'eval'}
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                group_record(BrowserReadinessConfig(),'seen_train',group,'frozen')

    def test_eval_rewards_cannot_create_training_reward_variance(self):
        seen=self.members([1.]*4)
        new=self.members([0.]*4,'new_train')
        groups=[group_record(BrowserReadinessConfig(),pool,members,'frozen')
                for pool,members in (('seen_train',seen),('new_train',new))]
        evaluation=deepcopy(seen[0]);evaluation.update(episode_id='eval',policy='sft')
        evaluation['case']={**evaluation['case'],'split':'eval'}
        report=readiness_report(seen+new+[evaluation],groups)
        self.assertFalse(report['ready_for_nonzero_grpo_signal'])
        self.assertEqual(report['pools']['seen_train']['all_success_groups'],1)
        self.assertEqual(report['pools']['new_train']['all_failure_groups'],1)
        self.assertEqual(report['fresh_evaluation']['all']['episodes'],1)
        self.assertEqual(report['optimizer_steps'],0)
        evaluation.update(success=False)
        evaluation['return']=0.
        report2=readiness_report(seen+new+[evaluation],groups)
        self.assertEqual(report['pools'],report2['pools'])

    def test_random_streams_are_unique_and_reproducible_across_all_groups(self):
        config=BrowserReadinessConfig()
        streams=[trajectory_seed(config,g,m) for g in range(6) for m in range(4)]
        self.assertEqual(len(set(streams)),24)
        self.assertEqual(streams,[trajectory_seed(config,g,m) for g in range(6) for m in range(4)])

    def test_saved_report_recomputes_groups_and_rejects_corrupt_advantages(self):
        config=BrowserReadinessConfig(sft_run_id='source')
        cases,pools=build_cases(config)
        template=test_browser.BrowserTests().episode()
        template['observations'][0]['pixel_sha256']='pixels'
        def episode(case,policy):
            result=deepcopy(template)
            result.update(case=case,policy=policy,episode_id=policy+'/'+case['case_id'])
            return result
        episodes=[episode(case,'oracle') for case in cases]
        episodes += [episode(case,'sft') for case in cases if case['split']=='eval']
        groups=[]
        for pool in ('seen_train','new_train'):
            for case_id in pools[pool]:
                case=next(c for c in cases if c['case_id']==case_id)
                members=[]
                for i in range(4):
                    e=episode(case,f'sample-{len(groups):03d}-{i}')
                    e['sampling_seed']=trajectory_seed(config,len(groups),i)
                    e['transitions'][0]['generation']={
                        'prompt_ids':[1,2,3],'generated_ids':[7,8], 'sampled_log_probs':[-.1,-.2],
                        'old_log_probs':[-.1,-.2],'behavior_fingerprint':'frozen',
                        'image_paths':[e['observations'][0]['image']],'action_loss_span':[3,5]}
                    members.append(e)
                groups.append(group_record(config,pool,members,'frozen'))
                episodes.extend(members)
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)/'stage10-fixture';directory.mkdir()
            for name in GPU_FILES:(directory/name).write_bytes(b'fixture')
            for name,value in {'config.json':asdict(config),'cases.json':cases,'pools.json':pools,
                               'episodes.json':episodes,'metrics.json':summarize(episodes),
                               'groups.json':groups,'readiness.json':readiness_report(episodes,groups)}.items():
                write_json(directory/name,value)
            with zipfile.ZipFile(directory/'images.zip','w') as archive:
                for o in template['observations']:archive.writestr(o['image'],b'fixture image')
            write_manifest(directory,{'mode':'gpu'},stage=10,artifact_files=GPU_FILES)
            self.assertEqual(replay_saved_run(directory),78)
            groups[0]['advantages'][0]=1.
            write_json(directory/'groups.json',groups)
            write_manifest(directory,{'mode':'gpu'},stage=10,artifact_files=GPU_FILES)
            with self.assertRaisesRegex(ValueError,'advantages'):
                replay_saved_run(directory)
