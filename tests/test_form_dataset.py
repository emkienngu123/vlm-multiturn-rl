"""Grounding geometry and dataset leakage checks without Torch, PIL or a browser."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

from vlm_multiturn_rl.browser import action_text, parse_action, initial_history, feedback_message, executable_action, browser_prompt
from vlm_multiturn_rl.stage13 import (FILES, FormDatasetConfig, pools, scaled_box, contains,
                                     teacher_action, demonstrations, validate_audit, summary, replay_saved_run)
from vlm_multiturn_rl.workspace import write_json, write_manifest


class FormDatasetTests(unittest.TestCase):
    def test_disjoint_pools_and_fixed_budget(self):
        config=FormDatasetConfig()
        data=pools(config)
        self.assertEqual([len(v) for v in data.values()],[100,20,16,32])
        pairs=[(c['task'],c['seed']) for cs in data.values() for c in cs]
        self.assertEqual(len(pairs),len(set(pairs)))
        self.assertTrue(all(c['split']=='train' for c in data['clean']+data['recovery']))
        self.assertTrue(all(c['split']!='train' for c in data['dev']+data['final']))
        for kw in ({'clean_episodes':99},{'clean_episodes':100.0},{'max_steps':8},{'max_sequence_tokens':4096}):
            with self.assertRaises(ValueError): FormDatasetConfig(**kw)

    def state(self):
        targets=[]
        for i,label in enumerate(('Year','Director','Genre','')):
            box={'x':50.,'y':30.+i*25,'width':90.,'height':20.}
            targets.append({'label':label,'tag':'INPUT' if label else 'DIV','id':'',
                            'value':'','focused':False,'box':box,'screenshot_box':scaled_box(box,1.5)})
        return {'scale':1.5,'viewport':{'width':332,'height':214},'targets':targets}

    def test_geometry_preserves_fractional_scale_and_excludes_right_boundary(self):
        box=scaled_box({'x':10.,'y':20.,'width':30.,'height':10.},1.5)
        self.assertEqual(box,{'x':15.,'y':30.,'width':45.,'height':15.})
        self.assertTrue(contains(box,15,30))
        self.assertFalse(contains(box,60,35))
        state=self.state()
        call=parse_action(teacher_action('Search for western movies directed by Curry from year 1974.',state),498,321)
        self.assertEqual(call['arguments'],{'x':142.5,'y':135.})

    def test_teacher_repairs_nonempty_field_before_typing_and_submits_last(self):
        goal='Search for western movies directed by Curry from year 1974.'
        state=self.state();genre=state['targets'][2]
        genre.update(value='incorrect-training-value',focused=True)
        self.assertEqual(teacher_action(goal,state),action_text('press',key='Control+a'))
        self.assertEqual(teacher_action(goal,state,action_text('press',key='Control+a')),action_text('type',text='western'))
        for target,value in zip(state['targets'],('1974','Curry','western','')): target['value']=value
        call=parse_action(teacher_action(goal,state),498,321)
        self.assertEqual(call['name'],'click')
        self.assertTrue(contains(state['targets'][-1]['screenshot_box'],**call['arguments']))

    def test_perturbations_and_heldout_cases_never_become_targets(self):
        transitions=[{'step':i+1,'messages':[{'role':'user','content':'screenshot only'}],
                      'text':action_text('type',text='wrong' if i<2 else 'correct'),
                      'generation':{'token_source':'synthetic_perturbation' if i<2 else 'privileged_oracle'}} for i in range(3)]
        episode={'case':pools(FormDatasetConfig())['recovery'][0],'policy':'recovery_teacher',
                 'success':True,'transitions':transitions}
        dev=deepcopy(episode);dev['case']=pools(FormDatasetConfig())['dev'][0]
        final=deepcopy(episode);final['case']=pools(FormDatasetConfig())['final'][0]
        model=deepcopy(episode);model['policy']='sft'
        result=demonstrations([episode,dev,final,model])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['turn'],3)
        self.assertEqual(set(result[0]),{'case_id','turn','messages','target','source','kind'})

    def audit(self):
        row={'step':1,'text':action_text('click',x=142.5,y=135.),'screenshot_size':[498,321],
             'state':self.state(),'point':{'x':142.5,'y':135.},'expected_css':{'x':95.,'y':90.},
             'actual_pointer':{'x':95.,'y':90.},'pointer_max_error_css':0.,'target_hits':['Genre']}
        return [{'case':{'task':'multi-orderings','seed':30000+i,'split':'eval'},'policy':policy,
                 'turns':[deepcopy(row)]} for policy in ('sft','grpo') for i in range(8)]

    def test_audit_recomputes_mapping_hits_and_requires_complete_pairs(self):
        audit=self.audit();validate_audit(audit)
        for mutate in (lambda a:a.pop(),
                       lambda a:a[0]['turns'][0]['actual_pointer'].update(x=190.),
                       lambda a:a[0]['turns'][0].update(target_hits=[]),
                       lambda a:a[0]['turns'][0]['state'].update(scale=1.),
                       lambda a:a[0]['case'].update(split='final')):
            changed=deepcopy(audit);mutate(changed)
            with self.assertRaises(ValueError): validate_audit(changed)

    def fixture(self, directory):
        config=FormDatasetConfig();registered=pools(config);episodes=[]
        for case in registered['clean']+registered['recovery']:
            recovery=case in registered['recovery']
            n=3 if recovery else 1
            obs=[{'goal':'Click the button.','image':'frame.png','width':498,'height':321,'pixel_sha256':'fixture'} for _ in range(n+1)]
            history=initial_history(obs[0],12);turns=[]
            for i in range(n):
                text=action_text('click',x=142.5,y=135.)
                call=parse_action(text,498,321);final=i==n-1
                turn={'step':i+1,'messages':deepcopy(history),'text':text,'action':call,
                      'executed':executable_action(call,498,321),'valid_action':True,'execution_error':False,
                      'reward':float(final),'raw_reward':float(final),'terminated':final,'truncated':False,
                      'generation':{'generated_ids':None,'token_source':'synthetic_perturbation' if recovery and i<2 else 'privileged_oracle'}}
                turns.append(turn);history.extend([{'role':'assistant','content':text},feedback_message(turn,obs[i+1]['image'])])
            episodes.append({'case':case,'policy':'recovery_teacher' if recovery else 'oracle','max_steps':12,
                             'observations':obs,'transitions':turns,'messages':history,'steps':n,'success':True,
                             'return':1.,'invalid_actions':0,'execution_errors':0,'end_reason':'task_terminal',
                             'field_order':['Year','Director','Genre']})
        audit=self.audit()
        for e in audit:
            for t in e['turns']:t['image']='frame.png'
        for name in FILES:(directory/name).write_text('fixture')
        # Header sufficient for the standard-library dimension check; this test
        # exercises integrity/split validation, not a PNG decoder.
        png=b'\x89PNG\r\n\x1a\n'+b'\x00\x00\x00\rIHDR'+struct.pack('>II',498,321)
        with zipfile.ZipFile(directory/'images.zip','w') as archive:archive.writestr('frame.png',png)
        objects={'config.json':asdict(config),'pools.json':registered,'episodes.json':episodes,
                 'demonstrations.json':demonstrations(episodes),'audit.json':audit,'summary.json':summary(episodes,audit),
                 'image-index.json':{'frame.png':{'sha256':hashlib.sha256(png).hexdigest(),'size':[498,321]}}}
        for name,value in objects.items():write_json(directory/name,value)
        (directory/'policy-prompt.txt').write_text(browser_prompt(12)+'\n')
        return objects

    def test_reader_rejects_heldout_training_and_injected_targets_after_rehash(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)/'stage13-fixture';directory.mkdir()
            data=self.fixture(directory)
            def seal():write_manifest(directory,{'mode':'cpu'},stage=13,artifact_files=FILES)
            seal();self.assertEqual(replay_saved_run(directory),120)
            original=deepcopy(data['demonstrations.json'])
            data['demonstrations.json'].append({'target':'deliberate mistake'})
            write_json(directory/'demonstrations.json',data['demonstrations.json']);seal()
            with self.assertRaisesRegex(ValueError,'targets'): replay_saved_run(directory)
            write_json(directory/'demonstrations.json',original)
            data['episodes.json'][-1]['case']=pools(FormDatasetConfig())['final'][0]
            write_json(directory/'episodes.json',data['episodes.json']);seal()
            with self.assertRaisesRegex(ValueError,'leakage'): replay_saved_run(directory)

    def test_reader_rejects_changed_png_after_manifest_is_rehashed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)/'stage13-fixture';directory.mkdir()
            self.fixture(directory)
            with zipfile.ZipFile(directory/'images.zip','w') as archive:archive.writestr('frame.png',b'changed')
            write_manifest(directory,{'mode':'cpu'},stage=13,artifact_files=FILES)
            with self.assertRaisesRegex(ValueError,'hash'):replay_saved_run(directory)
