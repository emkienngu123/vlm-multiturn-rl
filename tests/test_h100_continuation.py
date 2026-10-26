"""Migration/source guards and the separate H100 budget need no Torch."""
import unittest
from vlm_multiturn_rl.stage18 import budget_decision, validate_source_transition


class H100ContinuationTests(unittest.TestCase):
    def test_only_receipted_origin_can_cross_the_source_boundary(self):
        profile = {'origin_run_id':'origin', 'origin_source_sha256':{'code':'old'}}
        old = {'code':'old'}; current = {'code':'new'}
        validate_source_transition(profile,'origin',old,current,None)
        validate_source_transition(profile,'next',current,current,profile)
        validate_source_transition(None,'old-parent',old,old,None)
        for args in ((None,'origin',old,current,None),
                     (profile,'other',old,current,None),
                     (profile,'origin',current,old,None),
                     (profile,'next',old,current,profile),
                     (None,'next',current,current,profile),
                     (profile,'next',current,current,None)):
            with self.subTest(args=args),self.assertRaises(ValueError): validate_source_transition(*args)

    def test_new_budget_counts_only_new_work_but_predicts_conservatively(self):
        old = [{'seconds':1000.}]*4
        self.assertEqual(budget_decision(old),{'elapsed':0,'predicted':450.,'can_start':True})
        decision=budget_decision([*old,{'seconds':300.}])
        self.assertEqual(decision,{'elapsed':300.,'predicted':405.,'can_start':True})
        self.assertFalse(budget_decision([*old,*[{'seconds':300.}]*5])['can_start'])
