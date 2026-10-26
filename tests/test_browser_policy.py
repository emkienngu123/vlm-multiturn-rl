"""Prefix factorization and EOS boundaries without local Torch."""

import math
import unittest

from vlm_multiturn_rl.browser_policy import walk_prefixes


class BrowserPrefixTests(unittest.TestCase):
    def test_sampling_and_replay_use_identical_past_only_prefixes(self):
        calls=[]
        def next_probs(prefix):
            calls.append(prefix)
            probabilities=(.7,.2,.1) if not prefix else ((.1,.8,.1) if len(prefix)==1 else (.1,.1,.8))
            return [math.log(p) for p in probabilities]
        choices=iter([0,1,2])
        ids,sampled=walk_prefixes(next_probs,choose=lambda _:next(choices),eos_ids=[2])
        sample_calls=list(calls); calls.clear()
        replayed,scored=walk_prefixes(next_probs,target_ids=ids,eos_ids=[2])
        self.assertEqual(ids,[0,1,2])
        self.assertEqual(replayed,ids)
        self.assertEqual(sampled,scored)
        self.assertEqual(sample_calls,[(),(0,),(0,1)])
        self.assertEqual(calls,sample_calls)

    def test_length_cutoff_keeps_all_actual_tokens_and_eos_is_supervised(self):
        next_probs=lambda _: [math.log(.8),math.log(.2)]
        ids,scores=walk_prefixes(next_probs,choose=lambda _:0,max_tokens=2,eos_ids=[1])
        self.assertEqual(ids,[0,0])
        self.assertEqual(len(scores),2)
        ids,scores=walk_prefixes(next_probs,choose=lambda _:1,max_tokens=64,eos_ids=[1])
        self.assertEqual(ids,[1])
        self.assertEqual(scores,[math.log(.2)])

    def test_invalid_or_post_eos_targets_fail_before_scoring_a_wrong_prefix(self):
        fn=lambda _: [-1.,-2.,-3.]
        for kwargs in ({'target_ids':[]},{'target_ids':[2,1],'eos_ids':[2]},
                       {'target_ids':[3]},{'choose':lambda _:True},{'choose':lambda _:0,'max_tokens':0},
                       {'choose':lambda _:0,'target_ids':[0]},{}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):walk_prefixes(fn,**kwargs)


class CachedReplayCursorTests(unittest.TestCase):
    def test_replay_uses_exact_prefix_ids_and_includes_eos(self):
        from vlm_multiturn_rl.browser_policy import ReplayCursor
        cursor=ReplayCursor([7,8,2],eos_ids=[2])
        self.assertEqual(cursor.next([]),7)
        self.assertEqual(cursor.next([7]),8)
        self.assertEqual(cursor.next([7,8]),2)
        with self.assertRaises(RuntimeError):cursor.next([7,8,2])
        with self.assertRaises(ValueError):ReplayCursor([7,2,8],eos_ids=[2])
        with self.assertRaises(ValueError):ReplayCursor([])
        cursor=ReplayCursor([7,8])
        cursor.next([])
        with self.assertRaises(RuntimeError):cursor.next([9])
