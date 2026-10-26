"""Browser GRPO pool, checkpoint and saved-objective contracts without Torch."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import test_browser_difficulty
from vlm_multiturn_rl.stage10 import group_record, trajectory_seed
from vlm_multiturn_rl.stage12 import (BrowserGRPOConfig, artifact_files, build_cases, paired_comparison,
                                    pilot_config, replay_saved_run)
from vlm_multiturn_rl.browser import summarize
from vlm_multiturn_rl.workspace import sha256_file,write_json,write_manifest


class BrowserGRPOTests(unittest.TestCase):
    def test_fixed_training_and_reserved_evaluation_contract(self):
        config=BrowserGRPOConfig(sft_run_id='source')
        cases=build_cases(config)
        self.assertEqual(len(cases),10)
        self.assertEqual({c['task'] for c in cases},{'multi-orderings'})
        self.assertEqual([c['seed'] for c in cases if c['split']=='train'],[3000,3001])
        self.assertEqual([c['seed'] for c in cases if c['split']=='eval'],list(range(30000,30008)))
        self.assertNotIn('updates',asdict(pilot_config(config)))
        for kwargs in ({'updates':3},{'updates':0},{'episodes':1},{'learning_rate':1e-3},{'ratio_tolerance':.1}):
            with self.assertRaises(ValueError):BrowserGRPOConfig(**kwargs)

    def fixture(self,directory):
        config=BrowserGRPOConfig(sft_run_id='source');cases=build_cases(config)
        factory=test_browser_difficulty.DifficultyTests().episode
        episodes=[factory(c,'oracle') for c in cases]
        for policy in ('sft','grpo'):
            episodes.extend(factory(c,policy) for c in cases if c['split']=='eval')
        groups=[]
        for index,case in enumerate(cases[:2]):
            fingerprint=f'weights-{index}';members=[]
            for i,reward in enumerate([1.,0.,1.,0.]):
                e=factory(case,f'round-{index+1:03d}-sample-{i}',reward)
                e['sampling_seed']=trajectory_seed(config,index,i)
                e['transitions'][0]['generation']={
                    'prompt_ids':[1,2],'generated_ids':[7,8], 'sampled_log_probs':[-.1,-.2],
                    'old_log_probs':[-.1,-.2],'behavior_fingerprint':fingerprint,
                    'image_paths':[e['observations'][0]['image']],'action_loss_span':[2,4]}
                members.append(e)
            group=group_record(config,'new_train',members,fingerprint)
            group.update(round=index+1,update={'optimizer_step':True,'gradient_norm':1.,'loss':0.,
                'max_ratio_deviation':0.,'clipped_fraction':0.,'after_fingerprint':f'weights-{index+1}'})
            groups.append(group);episodes.extend(members)
        for name in artifact_files(config):
            path=directory/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
        for r in (1,2):
            cp=directory/f'checkpoints/round-{r:03d}'
            write_json(cp/'checkpoint.json',{'round':r,'optimizer_steps':r,'adapter_fingerprint':f'weights-{r}',
                'sha256':{n:sha256_file(cp/n) for n in ('adapter.zip','training-state.pt')}})
        objects={'config.json':asdict(config),'cases.json':cases,'episodes.json':episodes,'groups.json':groups,
                 'training.json':{'optimizer_steps':2,'paired_comparison':paired_comparison(episodes)},
                 'metrics.json':summarize(episodes)}
        for name,value in objects.items():write_json(directory/name,value)
        from vlm_multiturn_rl.browser import browser_prompt
        (directory/'policy-prompt.txt').write_text(browser_prompt(12)+'\n')
        with zipfile.ZipFile(directory/'images.zip','w') as archive:
            for o in episodes[0]['observations']:archive.writestr(o['image'],b'image')
        write_manifest(directory,{'mode':'gpu'},stage=12,artifact_files=artifact_files(config))
        return config,objects

    def test_reader_checks_full_report_and_rejects_stale_groups_or_checkpoint_steps_after_rehash(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)/'stage12-fixture';directory.mkdir()
            config,objects=self.fixture(directory)
            self.assertEqual(replay_saved_run(directory),34)
            groups=deepcopy(objects['groups.json']);groups[1]['behavior_fingerprint']='stale'
            write_json(directory/'groups.json',groups)
            write_manifest(directory,{},stage=12,artifact_files=artifact_files(config))
            with self.assertRaises(ValueError):replay_saved_run(directory)
            write_json(directory/'groups.json',objects['groups.json'])
            cp=directory/'checkpoints/round-002/checkpoint.json'
            info=json.loads(cp.read_text());info['optimizer_steps']=1;write_json(cp,info)
            write_manifest(directory,{},stage=12,artifact_files=artifact_files(config))
            with self.assertRaisesRegex(ValueError,'Checkpoint'):replay_saved_run(directory)

    def test_paired_evaluation_requires_identical_initial_states(self):
        case=next(c for c in build_cases(BrowserGRPOConfig()) if c['split']=='eval')
        a=test_browser_difficulty.DifficultyTests().episode(case,'sft');b=test_browser_difficulty.DifficultyTests().episode(case,'grpo')
        self.assertEqual(len(paired_comparison([a,b])),1)
        b['observations'][0]['pixel_sha256']='different'
        with self.assertRaises(ValueError):paired_comparison([a,b])
