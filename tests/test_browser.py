"""Browser policy boundaries and saved-history checks, with no local browser/Torch."""

from copy import deepcopy
import unittest

from vlm_multiturn_rl.browser import (BrowserConfig, action_text, cases, executable_action, feedback_message,
                                     initial_history, parse_action, public_observation, summarize, validate_episode)
from vlm_multiturn_rl.stage8 import demonstrations


class BrowserTests(unittest.TestCase):
    def test_action_parser_rejects_code_multiple_calls_duplicate_keys_and_bad_coordinates(self):
        invalid = ['mouse_click(1,2)', action_text('click', x=1, y=2)+action_text('click', x=3,y=4),
                   '<tool_call>{"name":"click","arguments":{"x":1,"x":2,"y":3}}</tool_call>',
                   action_text('click', x=True,y=1), action_text('click',x=498,y=1),
                   action_text('click',x=-1,y=1), action_text('click',x=float('nan'),y=1),
                   action_text('press',key='F12'), action_text('type',text='a\nb'),
                   action_text('__import__', text='os'), action_text('scroll',dy=601)]
        for text in invalid:
            with self.subTest(text=text): self.assertIsNone(parse_action(text,498,321))
        call = parse_action(action_text('click',x=100.5,y=50),498,321)
        self.assertEqual(executable_action(call,498,321), 'mouse_click(100.5, 50)')
        self.assertEqual(executable_action(None,498,321), 'noop(0)')

    def test_type_payload_is_data_and_observation_is_an_explicit_whitelist(self):
        payload = "'); __import__('os').system('bad') #"
        call = parse_action(action_text('type',text=payload),498,321)
        self.assertEqual(executable_action(call,498,321), f'keyboard_type({payload!r})')
        class Screenshot: shape = (321,498,3)
        obs = {'screenshot':Screenshot(), 'goal':'Enter example', 'dom_object':'PRIVATE', 'axtree_object':'PRIVATE',
               'extra_element_properties':{'secret':'label'}, 'last_action_error':'selector secret'}
        public = public_observation(obs,'images/0.png')
        self.assertEqual(set(public),{'goal','image','width','height'})
        self.assertNotIn('PRIVATE',str(initial_history(public)))

    def test_browser_cases_have_disjoint_seed_pairs_and_bounds(self):
        config = BrowserConfig(episodes=2)
        data = cases(config)
        self.assertEqual(len(data),12)
        pairs = lambda split: {(c['task'],c['seed']) for c in data if c['split']==split}
        self.assertFalse(pairs('train') & pairs('eval'))
        for kwargs in ({'episodes':0},{'episodes':5},{'max_steps':20},{'max_new_tokens':16}):
            with self.assertRaises(ValueError): BrowserConfig(**kwargs)

    def episode(self):
        observations = [{'goal':'Click the button.', 'image':f'images/test/{i}.png', 'width':498,'height':321} for i in range(2)]
        history = initial_history(observations[0])
        text = action_text('click',x=120,y=130)
        turn = {'step':1,'messages':deepcopy(history),'text':text,'generation':{'generated_ids':None},
                'action':parse_action(text,498,321),'executed':'mouse_click(120, 130)','valid_action':True,
                'execution_error':False,'reward':1.,'raw_reward':1.,'terminated':True,'truncated':False}
        history += [{'role':'assistant','content':text},feedback_message(turn,observations[1]['image'])]
        return {'case':cases(BrowserConfig())[0], 'policy':'oracle','max_steps':8,'observations':observations,
                'transitions':[turn],'messages':history,'steps':1,'success':True,'return':1.,
                'invalid_actions':0,'execution_errors':0,'end_reason':'task_terminal'}

    def test_replay_validates_exact_history_actions_rewards_and_final_observation(self):
        ep = self.episode()
        validate_episode(ep)
        for key,value in [('return',0),('success',False),('steps',2)]:
            changed = deepcopy(ep);changed[key]=value
            with self.assertRaises(ValueError):validate_episode(changed)
        changed=deepcopy(ep);changed['transitions'][0]['messages'][0]['content']='leaked state'
        with self.assertRaises(ValueError):validate_episode(changed)
        changed=deepcopy(ep);changed['transitions'][0]['raw_reward']=-1
        with self.assertRaises(ValueError):validate_episode(changed)

    def test_demonstrations_never_include_eval_or_model_actions(self):
        train=self.episode();evaluation=deepcopy(train);evaluation['case']['split']='eval'
        model=deepcopy(train);model['policy']='base'
        self.assertEqual(len(demonstrations([train,evaluation,model])),1)
        self.assertEqual(summarize([train,evaluation])['oracle']['all']['episodes'],1)
