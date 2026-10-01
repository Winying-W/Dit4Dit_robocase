"""Recovery must retain failed attempts and never repeat completed policy trials."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.robocasa365.evaluate_multitask import evaluate_task
from scripts.robocasa365.scene_protocol import STABLE, protocol_spec


class EvaluationRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.task = dict(task='TestTask', horizon=10, path='/unused/dataset')
        self.artifacts = dict(checkpoint='same-checkpoint')
        self.args = SimpleNamespace(output=self.root, checkpoint=Path('/unused/action.pt'),
                                    sim_python='unused-python', scene_protocol=STABLE)
        (self.root / 'TestTask').mkdir()

    def attempt(self, number, seeds, *, status='failed', artifacts=None):
        folder = self.root / 'TestTask' / f'attempt_{number}'
        folder.mkdir(exist_ok=True)
        (folder / 'checkpoint_verification.json').write_text(json.dumps(dict(
            policy_artifacts=self.artifacts if artifacts is None else artifacts,
            scene_protocol=protocol_spec(STABLE))))
        (folder / 'evaluation.json').write_text(json.dumps(dict(
            task='TestTask', initialization='fresh_gym_target', horizon=10, official_horizon=10,
            scene_protocol=protocol_spec(STABLE), status=status, error=None if status=='complete' else 'quota',
            scene_protocol_runtime=dict(protocol=protocol_spec(STABLE), installed_file_unchanged=True,
                python_hash_seed='0',
                source_sha256=protocol_spec(STABLE)['counter_region_patch']['source_sha256'],
                effective_function_sha256=protocol_spec(STABLE)['counter_region_patch']['effective_function_sha256']),
            episodes=[dict(trial=i, seed=seed, demo_episode=None, success=False)
                      for i, seed in enumerate(seeds)])))
        return folder

    def call(self):
        with patch('scripts.robocasa365.evaluate_multitask.audit_phase', return_value=dict(passed=True)):
            return evaluate_task(self.args, self.task, [100, 101, 102], {}, self.artifacts, 0, {})

    def test_existing_budget_does_not_restart_failed_or_completed_trials(self):
        self.attempt(0, [100])
        (self.root / 'TestTask/attempt_1').mkdir()
        with patch('scripts.robocasa365.evaluate_multitask.subprocess.run') as worker:
            row = self.call()
        worker.assert_not_called()
        self.assertEqual(row['completed_trials'], 1)
        self.assertEqual(row['missing_seeds'], [101, 102])
        self.assertIsNone(row['success_rate'])
        self.assertEqual([x['status'] for x in row['attempts']], ['failed', 'infrastructure_failure'])

    def test_explicit_extended_budget_submits_only_missing_seeds_and_keeps_evidence(self):
        folder = self.attempt(0, [100])
        original = (folder / 'evaluation.json').read_bytes()
        (self.root / 'TestTask/attempt_1').mkdir()
        self.args.max_attempts = 4
        def worker(cmd, **kwargs):
            self.assertEqual(cmd[cmd.index('--seeds')+1:], ['101', '102'])
            self.assertTrue(cmd[cmd.index('--output')+1].endswith('/attempt_2'))
            self.attempt(2, [101, 102], status='complete')
            return SimpleNamespace(returncode=0)
        with patch('scripts.robocasa365.evaluate_multitask.subprocess.run', side_effect=worker) as run:
            row = self.call()
        self.assertEqual(run.call_count, 1)
        self.assertEqual((folder / 'evaluation.json').read_bytes(), original)
        self.assertTrue(row['complete'])
        self.assertEqual([x['seed'] for x in row['trials']], [100, 101, 102])
        self.assertEqual(len(row['attempts']), 3)

    def test_duplicate_result_is_rejected(self):
        self.attempt(0, [100]); self.attempt(1, [100])
        with self.assertRaises(AssertionError): self.call()

    def test_changed_policy_is_rejected(self):
        self.attempt(0, [100], artifacts=dict(checkpoint='other-checkpoint'))
        with self.assertRaises(RuntimeError): self.call()

    def test_malformed_report_is_not_silently_discarded(self):
        folder = self.attempt(0, [100])
        (folder / 'evaluation.json').write_text('{')
        with self.assertRaises(json.JSONDecodeError): self.call()

    def test_existing_attempts_outside_budget_or_with_gap_are_rejected(self):
        self.attempt(0, [100]); self.attempt(2, [101])
        for budget in (2, 4):
            self.args.max_attempts=budget
            with self.assertRaises(ValueError): self.call()


if __name__ == '__main__':
    unittest.main()
