import copy
import unittest

from scripts.robocasa365.human300_scaleup import scaleup_plan


class ScaleupPlanTest(unittest.TestCase):
    def setUp(self):
        self.saved = dict(world_size=1, microbatch=1, accumulation=64, seed=42,
                          steps=10000, warmup=500, manifest_sha256='data',
                          initialization_sha256='cosmos', action_lr=1e-4,
                          source_sha256={'trainer.py':'original', 'adapter.py':'adapter'})
        self.requested = {**self.saved, 'world_size':4, 'accumulation':16}

    def test_retains_existing_rng_and_gives_new_ranks_unique_reproducible_seeds(self):
        original = copy.deepcopy(self.saved)
        plan = scaleup_plan(self.saved, self.requested, 2000)
        self.assertEqual(plan, scaleup_plan(self.saved, self.requested, 2000))
        self.assertEqual(plan['preserve_rank_rng'], [0])
        self.assertEqual(len(set(plan['new_rank_seeds'].values())), 3)
        self.assertEqual(plan['parent_step'], 2000)
        self.assertFalse(plan['exact_sample_sequence_preserved'])
        self.assertEqual(self.saved, original)

    def test_rejects_data_initialization_budget_lr_or_batch_changes(self):
        for key, value in [('manifest_sha256','other'), ('initialization_sha256','other'),
                           ('steps',20000), ('warmup',100), ('action_lr',1e-3), ('accumulation',32)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                scaleup_plan(self.saved, {**self.requested,key:value}, 2000)

    def test_source_changes_require_exact_explicit_transition(self):
        target = copy.deepcopy(self.requested)
        target['source_sha256']['trainer.py'] = 'new'
        with self.assertRaises(ValueError):
            scaleup_plan(self.saved, target, 2000)
        transition = {'trainer.py':dict(before='original',after='new')}
        self.assertEqual(scaleup_plan(self.saved,target,2000,source_transition=transition)['source_transition'], transition)
        with self.assertRaises(ValueError):
            scaleup_plan(self.saved,target,2000,source_transition={'trainer.py':dict(before='wrong',after='new')})


if __name__ == '__main__':
    unittest.main()
