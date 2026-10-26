"""Hard-task transfer, reactive teacher, and train-only report contracts."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import zipfile

import test_browser
from vlm_multiturn_rl.browser import (BrowserConfig, HARD_TASKS, browser_prompt, initial_history,
                                     oracle_next_action, parse_action, summarize, validate_episode)
from vlm_multiturn_rl.stage11 import (BrowserDifficultyConfig, GPU_FILES, build_cases, reserved_eval,
                                    difficulty_report, replay_saved_run, source_receipt)
from vlm_multiturn_rl.stage10 import group_record, trajectory_seed
from vlm_multiturn_rl.stage8 import demonstrations
from vlm_multiturn_rl.workspace import write_json, write_manifest


class DifficultyTests(unittest.TestCase):
    def test_suite_and_budget_are_explicit_and_old_contract_is_preserved(self):
        config = BrowserDifficultyConfig()
        self.assertEqual(config.tasks,HARD_TASKS)
        self.assertEqual(len(build_cases(config)),6)
        self.assertEqual(len(reserved_eval(config)),24)
        self.assertFalse({(c['task'],c['seed']) for c in build_cases(config)} &
                         {(c['task'],c['seed']) for c in reserved_eval(config)})
        self.assertEqual(browser_prompt(12).replace('twelve','eight'),browser_prompt(8))
        for kwargs in ({'episodes':3},{'max_steps':8},{'max_sequence_tokens':4096},{'tasks':BrowserConfig().tasks}):
            with self.assertRaises(ValueError): BrowserDifficultyConfig(**kwargs)
        with self.assertRaises(ValueError): BrowserConfig(tasks=HARD_TASKS)

    def test_popup_teacher_reacts_before_typing_and_never_chooses_exit(self):
        class Locator:
            def __init__(self,page,selector): self.page,self.selector=page,selector
            def is_visible(self): return self.page.popup
            def bounding_box(self):
                return {'x':10,'y':20,'width':20,'height':10} if self.selector=='#popup-cancel' else {'x':40,'y':50,'width':20,'height':10}
            def input_value(self): return self.page.values[self.selector]
            def evaluate(self,code): return self.page.focus==self.selector
        class Page:
            _bgym_scale_factor = 1.5
            popup = True
            focus = '#username'
            values = {'#username':'','#password':''}
            def locator(self,selector): return Locator(self,selector)
        page = Page()
        world = SimpleNamespace(case={'task':'login-user-popup'},obs={'goal':'Enter "alice" and "secret"'},
                                env=SimpleNamespace(unwrapped=SimpleNamespace(page=page)))
        action = parse_action(oracle_next_action(world),498,321)
        self.assertEqual(action['arguments'],{'x':30.,'y':37.5})
        page.popup = False
        self.assertEqual(parse_action(oracle_next_action(world),498,321)['arguments'],{'text':'alice'})
        page.values['#username']='alice'
        self.assertEqual(parse_action(oracle_next_action(world),498,321)['name'],'click')
        page.focus='#password'
        self.assertEqual(parse_action(oracle_next_action(world),498,321)['arguments'],{'text':'secret'})

    def episode(self,case,policy,reward=1.):
        e = deepcopy(test_browser.BrowserTests().episode())
        e.update(case=case,policy=policy,episode_id=policy+'/'+case['case_id'],max_steps=12)
        e['transitions'][0]['messages']=initial_history(e['observations'][0],12)
        e['messages'][0]['content']=browser_prompt(12)
        e['transitions'][0].update(reward=reward,raw_reward=1. if reward else -1.)
        feedback = json.loads(e['messages'][-1]['content'][0]['text']);feedback['reward']=reward
        e['messages'][-1]['content'][0]['text']=json.dumps(feedback,sort_keys=True)
        e.update(success=bool(reward),return_value=reward)
        e['return']=reward
        e['observations'][0]['pixel_sha256']='same reset'
        validate_episode(e)
        return e

    def fixture(self):
        config = BrowserDifficultyConfig(sft_run_id='source')
        cases = build_cases(config)
        task = config.tasks[0]
        episodes = [self.episode(c,'oracle') for c in cases]
        groups = []
        for case in [c for c in cases if c['task']==task]:
            members=[]
            for i,reward in enumerate([1.,0.,1.,0.]):
                e=self.episode(case,f'sample-{len(groups):03d}-{i}',reward)
                e['sampling_seed']=trajectory_seed(config,len(groups),i)
                e['transitions'][0]['generation']={
                    'prompt_ids':[1,2],'generated_ids':[7,8], 'sampled_log_probs':[-.1,-.2],
                    'old_log_probs':[-.1,-.2],'behavior_fingerprint':'frozen',
                    'image_paths':[e['observations'][0]['image']],'action_loss_span':[2,4]}
                members.append(e)
            groups.append(group_record(config,'new_train',members,'frozen'));episodes.extend(members)
        return config,cases,task,episodes,groups

    def test_train_metrics_and_group_signal_are_separate_from_reserved_eval(self):
        config,cases,task,episodes,groups=self.fixture()
        report=difficulty_report(episodes,groups,task)
        self.assertEqual(summarize(episodes),{})
        self.assertEqual(report['training_metrics'][task]['successes'],4)
        self.assertEqual(report['mixed_return_groups'],2)
        self.assertEqual(len(demonstrations(episodes)),6)
        with self.assertRaises(ValueError):
            changed=deepcopy(episodes[-1]);changed['max_steps']=8;validate_episode(changed)

    def test_saved_shard_rejects_eval_leakage_and_corrupt_advantages_even_after_rehash(self):
        config,cases,task,episodes,groups=self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)/'stage11-fixture';directory.mkdir()
            for name in GPU_FILES:(directory/name).write_bytes(b'fixture')
            objects={'config.json':asdict(config),'cases.json':cases,'reserved-eval.json':reserved_eval(config),
                     'episodes.json':episodes,'metrics.json':{},'demonstrations.json':demonstrations(episodes),
                     'groups.json':groups,'difficulty.json':difficulty_report(episodes,groups,task)}
            for name,value in objects.items():write_json(directory/name,value)
            (directory/'policy-prompt.txt').write_text(browser_prompt(12)+'\n')
            with zipfile.ZipFile(directory/'images.zip','w') as archive:
                for o in episodes[0]['observations']:archive.writestr(o['image'],b'image')
            def seal():write_manifest(directory,{'mode':'gpu','task':task},stage=11,artifact_files=GPU_FILES)
            seal();self.assertEqual(replay_saved_run(directory),14)
            groups[0]['advantages'][0]=0.
            write_json(directory/'groups.json',groups);seal()
            with self.assertRaisesRegex(ValueError,'advantages'):replay_saved_run(directory)
            groups[0]=group_record(config,'new_train',episodes[6:10],'frozen')
            write_json(directory/'groups.json',groups)
            episodes[-1]['case']=reserved_eval(config)[0]
            write_json(directory/'episodes.json',episodes);seal()
            with self.assertRaisesRegex(ValueError,'evaluation'):replay_saved_run(directory)
