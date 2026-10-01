"""Synthetic reporting adversaries; none of these fixtures are policy trials."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from scripts.robocasa365.compare_development_checkpoints import compare, sha
from scripts.robocasa365.watch_sampling_comparison import declaration, ready_inputs, validate_result


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class SamplingFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)
        self.tasks = [f'Task{i}' for i in range(16)]
        self.selected = [self.tasks[i] for i in [0, 2, 4]]
        self.seeds = list(range(100, 120))
        self.manifest = dict(task_set='composite_seen', task_scope='full_official_task_set',
                             tasks=[dict(task=name) for name in self.tasks],
                             evaluation=dict(selection_seeds=self.seeds, seeds=list(range(1000, 1100))))
        self.plan = dict(branches=['paired_uniform', 'initial_switch'], additional_updates_per_branch=2000,
                         manifest=str(self.run / 'manifest.json'), development_tasks=self.selected[::-1],
                         development_seeds=self.seeds, development_trials_per_branch=60)
        self.freeze()

    def freeze(self):
        put(self.run / 'manifest.json', self.manifest)
        self.plan['manifest_sha256'] = sha(self.run / 'manifest.json')
        put(self.run / 'plan.json', self.plan)
        put(self.run / 'cloud_source_hashes.json', {})
        put(self.run / 'prequeue_acceptance.json', dict(status='complete', evidence_sha256={
            name: sha(self.run / name) for name in ['plan.json', 'cloud_source_hashes.json']}))

    def reports(self):
        self.report = dict(status='complete', purpose='development', task_scope='declared_task_subset',
                           planned_tasks=self.selected, seeds=self.seeds, completed_trials=60, planned_trials=60)
        for branch in self.plan['branches']:
            folder = self.run / branch / 'development_2000'
            put(folder / 'report.json', self.report)
            put(folder / 'identity.json', {})
        return self.run / self.plan['branches'][1] / 'development_2000/report.json'


class DeclaredSamplingScreenTest(SamplingFixture):
    def test_screen_keeps_original_manifest_and_canonical_task_order(self):
        before = (self.run / 'manifest.json').read_bytes()
        _, tasks = declaration(self.run)
        self.assertEqual(tasks, self.selected)
        self.assertEqual((self.run / 'manifest.json').read_bytes(), before)
        self.assertIsNone(ready_inputs(self.run, self.plan, tasks))
        self.reports()
        self.assertEqual(len(ready_inputs(self.run, self.plan, tasks)), 2)

    def test_posthoc_plan_change_is_rejected(self):
        self.plan['development_tasks'][0] = 'Task6'
        put(self.run / 'plan.json', self.plan)
        with self.assertRaisesRegex(ValueError, 'Frozen experiment changed'):
            declaration(self.run)

    def test_final_seed_contamination_is_rejected(self):
        self.manifest['evaluation']['seeds'][0] = 100
        self.freeze()
        with self.assertRaisesRegex(ValueError, 'Final seed contamination'):
            declaration(self.run)

    def test_truncated_training_manifest_is_rejected(self):
        self.manifest['tasks'] = self.manifest['tasks'][:3]
        self.freeze()
        with self.assertRaisesRegex(ValueError, 'full Composite16'):
            declaration(self.run)

    def test_missing_trials_are_pending_and_false_complete_is_rejected(self):
        path = self.reports()
        put(path, dict(self.report, status='incomplete', completed_trials=59))
        self.assertIsNone(ready_inputs(self.run, self.plan, self.selected))
        put(path, dict(self.report, completed_trials=59))
        with self.assertRaisesRegex(ValueError, 'denominator'):
            ready_inputs(self.run, self.plan, self.selected)

    def test_wrong_scope_tasks_and_seeds_cannot_enter_comparison(self):
        path = self.reports()
        changes = [dict(purpose='final'), dict(task_scope='full_official_task_set'),
                   dict(planned_tasks=self.selected[:2]), dict(seeds=list(range(101, 121)))]
        for change in changes:
            with self.subTest(change=change):
                put(path, dict(self.report, **change))
                with self.assertRaises(ValueError):
                    ready_inputs(self.run, self.plan, self.selected)

    def test_scene_mismatch_never_keeps_an_improvement_claim(self):
        signatures = [dict(report='synthetic')]
        result = dict(status='scene_mismatch', input_signatures=signatures, branches=self.plan['branches'],
                      left_step=2000, right_step=2000, purpose='development', task_scope='declared_task_subset',
                      planned_pairs=60, matched_pairs=59, mismatched_pairs=1, full16_success_rate=None,
                      paired_success_rate_delta=None, paired_delta_bootstrap_95=None, gained=None, lost=None)
        validate_result(result, signatures, self.plan)
        for field, bad in [('paired_success_rate_delta', .1), ('gained', 2), ('full16_success_rate', .1)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_result(dict(result, **{field: bad}), signatures, self.plan)


class EqualStepScenesTest(SamplingFixture):
    def setUp(self):
        super().setUp()
        self.dirs = [self.run / branch / 'development_2000' for branch in self.plan['branches']]
        for index, folder in enumerate(self.dirs):
            rows = []
            for task in self.selected:
                trials = []
                for seed in self.seeds:
                    trial = folder / task / str(seed)
                    trial.mkdir(parents=True)
                    (trial / 'initial_model.xml').write_text('<mujoco><camera pos="0 0 1"/></mujoco>')
                    np.savez(trial / 'trajectory.npz', initial_state=np.arange(5, dtype=float))
                    trials.append(dict(seed=seed, success=(index == 1 and seed == 100), artifact=str(trial)))
                rows.append(dict(task=task, trials=trials))
            put(folder / 'report.json', dict(tasks=rows))
            normal = folder / 'normalization.json'
            put(normal, dict(action='synthetic fixture'))
            config = folder / 'data_config.yaml'
            config.write_text(yaml.safe_dump(dict(image_size=[128, 128], statistics_path=str(normal))))
            artifacts = {key: dict(sha256='same') for key in ['base_weights', 'base_config', 'base_statistics']}
            for key, path in [('normalization', normal), ('data_config', config)]:
                artifacts[key] = dict(path=str(path), sha256=sha(path))
            put(folder / 'identity.json', dict(purpose='development', tasks=self.selected, seeds=self.seeds,
                manifest_sha256=sha(self.run / 'manifest.json'), execute_horizon=8, task_set='composite_seen',
                task_scope='declared_task_subset', policy_artifacts=artifacts))
        # Only scene pairing/dispatch is tested here. Real action re-audits are
        # mandatory in production and have separate protocol/action tests.
        def reviewed(directory, manifest, step, purpose, tasks):
            self.assertEqual(tasks, self.selected)
            self.assertEqual((step, purpose), (2000, 'development'))
            return dict(status='complete', saved_action_chain_passed=True,
                        report_sha256=sha(directory / 'report.json'), identity_sha256=sha(directory / 'identity.json'))
        self.auditor = patch('scripts.robocasa365.compare_development_checkpoints.review', side_effect=reviewed)
        self.auditor.start()
        self.addCleanup(self.auditor.stop)

    def call(self, **kwargs):
        return compare(*self.dirs, self.run / 'manifest.json', 2000, 2000,
                       requested_tasks=self.selected, **kwargs)

    def test_equal_step_comparison_requires_explicit_branch_mode(self):
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            self.call()

    def test_all60_scenes_are_compared_and_subset_is_retained(self):
        result = self.call(same_step_branches=True)
        self.assertEqual((result['planned_pairs'], result['matched_pairs']), (60, 60))
        self.assertEqual(result['task_scope'], 'declared_task_subset')
        self.assertEqual((result['gained'], result['lost']), (3, 0))
        self.assertAlmostEqual(result['paired_success_rate_delta'], .05)
        self.assertTrue(result['data_config_equivalence']['statistics_path_relocation'])

    def change_config(self, **changes):
        folder = self.dirs[1]
        path = folder / 'data_config.yaml'
        config = yaml.safe_load(path.read_text())
        config.update(changes)
        path.write_text(yaml.safe_dump(config))
        identity = json.loads((folder / 'identity.json').read_text())
        identity['policy_artifacts']['data_config']['sha256'] = sha(path)
        put(folder / 'identity.json', identity)

    def test_different_camera_preprocessing_is_rejected(self):
        self.change_config(image_size=[256, 256])
        with self.assertRaisesRegex(ValueError, 'beyond the statistics path'):
            self.call(same_step_branches=True)

    def test_statistics_path_must_resolve_to_declared_normalization(self):
        alternative = self.run / 'other_stats.json'
        put(alternative, dict(action='different values'))
        self.change_config(statistics_path=str(alternative))
        with self.assertRaisesRegex(ValueError, 'different normalization artifact'):
            self.call(same_step_branches=True)

    def test_changed_statistics_bytes_are_rejected_even_with_mocked_review(self):
        put(self.dirs[1] / 'normalization.json', dict(action='changed after identity was written'))
        with self.assertRaisesRegex(ValueError, 'statistics changed'):
            self.call(same_step_branches=True)

    def test_one_different_scene_suppresses_aggregate_paired_estimate(self):
        trial = self.dirs[1] / self.selected[1] / '110'
        np.savez(trial / 'trajectory.npz', initial_state=np.arange(5, dtype=float) + .01)
        result = self.call(same_step_branches=True)
        self.assertEqual(result['status'], 'scene_mismatch')
        self.assertEqual((result['matched_pairs'], result['mismatched_pairs']), (59, 1))
        self.assertIsNone(result['paired_success_rate_delta'])


if __name__ == '__main__':
    unittest.main()
