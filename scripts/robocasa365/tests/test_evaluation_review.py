"""Synthetic protocol adversaries; these fixtures are not policy success data."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.robocasa365.review_evaluation import review
from scripts.robocasa365.scene_protocol import STABLE, protocol_spec


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class EvaluationReviewTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / 'evaluation'
        self.manifest = self.root / 'manifest.json'
        tasks = [dict(task=f'Task{i}', horizon=900) for i in range(18)]
        put(self.manifest, dict(tasks=tasks, task_set='atomic_seen', task_scope='full_official_task_set',
                               evaluation=dict(selection_seeds=[100], seeds=[1000])))
        normal = self.root / 'normalization.json'
        put(normal, {})
        artifacts = dict(normalization=dict(path=str(normal), sha256=hashlib.sha256(normal.read_bytes()).hexdigest()))
        names = [task['task'] for task in tasks]
        identity = dict(policy_artifacts=artifacts, checkpoint='test_step10000.pt', tasks=names, seeds=[100],
                        manifest_sha256=hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                        task_set='atomic_seen', task_scope='full_official_task_set', purpose='development', execute_horizon=8)
        put(self.directory / 'identity.json', identity)
        rows = []
        for task in tasks:
            name = task['task']
            folder = self.directory / name / 'attempt_0'
            artifact = folder / 'trial_000'
            result = dict(trial=0, seed=100, demo_episode=None, success=False, steps=900,
                          action_chain_runtime_passed=True, controller_input_type='delta', terminated=False, truncated=False)
            put(artifact / 'result.json', result)
            put(folder / 'evaluation.json', dict(task=name, split='target', initialization='fresh_gym_target',
                                                horizon=900, official_horizon=900, execution_horizon=8, episodes=[result]))
            verification = dict(policy_artifacts=artifacts, global_step=10000, task=name, checkpoint=identity['checkpoint'],
                                normalization_sha256=artifacts['normalization']['sha256'], architecture=dict(framework='DiT4DiT'),
                                max_steps_override=None, demo_episodes=None, seeds=[100], execute_horizon=8)
            put(folder / 'checkpoint_verification.json', verification)
            rows.append(dict(task=name, complete=True, planned_trials=1, completed_trials=1, successes=0,
                             success_rate=0., missing_seeds=[], attempts=[dict(status='complete')],
                             trials=[dict(**result, artifact=str(artifact))]))
        self.report = dict(status='complete', task_set='atomic_seen', task_scope='full_official_task_set', purpose='development',
                           planned_tasks=names, seeds=[100], planned_trials=18, completed_trials=18, successes=0, tasks=rows,
                           pooled_success_rate=0., equal_task_macro_success_rate=0.)
        put(self.directory / 'report.json', self.report)
        self.audit = patch('scripts.robocasa365.review_evaluation.audit_trial', return_value=dict(
            steps=900, passed=True, executed_arm_clipped_component_fraction=0.,
            executed_gripper_closed_fraction=.5, executed_base_mode_fraction=0.)).start()
        self.addCleanup(patch.stopall)

    def call(self):
        return review(self.directory, self.manifest, 10000, 'development')

    def test_complete_zero_result_keeps_cause_unresolved(self):
        result = self.call()
        self.assertEqual(result['completed_trials'], 18)
        self.assertEqual(result['pooled_success_rate'], 0.)
        self.assertEqual(result['behavioral_root_cause'], 'not established by this artifact review')
        self.assertIn('does not identify', result['recommended_next_action'])

    def test_explicit_development_subset_is_not_reported_as_full_training_task_set(self):
        manifest_bytes=self.manifest.read_bytes()
        identity_path=self.directory/'identity.json';identity=json.loads(identity_path.read_text())
        selected=['Task0','Task1']
        identity.update(tasks=selected,task_scope='declared_task_subset');put(identity_path,identity)
        self.report.update(tasks=self.report['tasks'][:2],planned_tasks=selected,
                           planned_trials=2,completed_trials=2,task_scope='declared_task_subset')
        put(self.directory/'report.json',self.report)
        with self.assertRaisesRegex(ValueError,'Subset is not a full benchmark'):self.call()
        result=review(self.directory,self.manifest,10000,'development',selected)
        self.assertEqual(result['planned_trials'],2)
        self.assertEqual(result['completed_trials'],2)
        self.assertEqual(result['task_scope'],'declared_task_subset')
        self.assertEqual(self.manifest.read_bytes(),manifest_bytes)
        with self.assertRaisesRegex(ValueError,'restricted to development'):
            review(self.directory,self.manifest,10000,'final',selected)

    def test_invalid_explicit_subset_is_rejected(self):
        for selected in [[],['Task0','Task0'],['Unknown']]:
            with self.subTest(selected=selected),self.assertRaisesRegex(ValueError,'Invalid explicit development task list'):
                review(self.directory,self.manifest,10000,'development',selected)

    def test_partial_has_no_aggregate_success_rate(self):
        value = copy.deepcopy(self.report)
        value.update(status='incomplete', tasks=value['tasks'][:1], completed_trials=1,
                     pooled_success_rate=None, equal_task_macro_success_rate=None)
        put(self.directory / 'report.json', value)
        result = self.call()
        self.assertIsNone(result['pooled_success_rate'])
        self.assertEqual(result['status'], 'incomplete')

    def test_duplicate_seed_rejected(self):
        self.report['tasks'][0]['trials'] *= 2
        put(self.directory / 'report.json', self.report)
        with self.assertRaisesRegex(ValueError, 'Duplicate or unexpected trial seed'):
            self.call()

    def test_empty_report_has_no_action_chain_proof(self):
        value = copy.deepcopy(self.report)
        value.update(status='incomplete', tasks=[], completed_trials=0,
                     pooled_success_rate=None, equal_task_macro_success_rate=None)
        put(self.directory / 'report.json', value)
        result = self.call()
        self.assertFalse(result['saved_action_chain_passed'])
        self.assertIsNone(result['pooled_success_rate'])

    def test_smoke_and_wrong_checkpoint_rejected(self):
        path = self.directory / 'Task0/attempt_0/checkpoint_verification.json'
        original = json.loads(path.read_text())
        for change, message in [(dict(max_steps_override=8), 'Truncated smoke'),
                                (dict(global_step=12), 'Unexpected checkpoint step'),
                                (dict(demo_episodes=[0]), 'Demo initialization')]:
            with self.subTest(change=change):
                put(path, dict(original, **change))
                with self.assertRaisesRegex(ValueError, message):
                    self.call()

    def test_tampered_numerator_rejected(self):
        self.report['successes'] = 1
        put(self.directory / 'report.json', self.report)
        with self.assertRaisesRegex(ValueError, 'Aggregate counts mismatch'):
            self.call()

    def test_normalization_change_rejected(self):
        put(self.root / 'normalization.json', dict(changed=True))
        with self.assertRaisesRegex(ValueError, 'Normalization file changed'):
            self.call()

    def test_failed_action_audit_propagates(self):
        self.audit.side_effect = AssertionError('delta decoding mismatch')
        with self.assertRaisesRegex(AssertionError, 'delta decoding mismatch'):
            self.call()

    def test_mixed_scene_protocol_is_rejected_before_action_audit(self):
        path = self.directory / 'Task0/attempt_0/checkpoint_verification.json'
        value = json.loads(path.read_text())
        value['scene_protocol'] = protocol_spec(STABLE)
        put(path, value)
        with self.assertRaisesRegex(ValueError, 'Scene protocol differs'):
            self.call()
        self.audit.assert_not_called()

    def test_final_seed_contamination_rejected(self):
        manifest = json.loads(self.manifest.read_text())
        manifest['evaluation']['seeds'] = [100]
        put(self.manifest, manifest)
        with self.assertRaisesRegex(ValueError, 'Development and final seeds overlap'):
            self.call()


if __name__ == '__main__':
    unittest.main()
