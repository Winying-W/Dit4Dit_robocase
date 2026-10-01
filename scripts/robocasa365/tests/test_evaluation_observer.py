"""Observer metadata adversaries; these fixtures are not training or evaluation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.robocasa365.scene_protocol import STABLE, protocol_spec
from scripts.robocasa365.watch_evaluation_reviews import scan, validation_windows


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class EvaluationObserverTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / 'run'
        self.training = self.root / 'training'
        self.output = self.root / 'observer'
        self.output.mkdir()
        self.manifest = self.run / 'prepared/manifest.json'
        tasks = [dict(task='TaskA', train_episodes=[0, 1], validation_episodes=[2]),
                 dict(task='TaskB', train_episodes=[0, 1], validation_episodes=[2, 3])]
        put(self.manifest, dict(task_set='composite_seen', tasks=tasks))
        self.split = dict(task_set='composite_seen',
                          manifest_sha256=hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                          tasks=[dict(tasks[0], validation_indices=[15, 30]),
                                 dict(tasks[1], validation_indices=[40, 50, 60])],
                          validation_indices=[[0, 15], [0, 30], [1, 40], [1, 50], [1, 60]])
        put(self.training / 'split.json', self.split)
        put(self.run / 'plan.json', dict(development_checkpoints=[10000], final_checkpoint=50000,
                                        planned_final_trials=1600, scene_protocol=protocol_spec(STABLE)))

    def test_count_is_actual_windows_not_episodes_or_task_multiple(self):
        result = validation_windows(self.training, self.manifest)
        self.assertEqual(result['count'], 5)
        self.assertEqual([row['windows'] for row in result['tasks']], [2, 3])
        self.assertEqual([row['held_out_episodes'] for row in result['tasks']], [1, 2])
        state = scan(self.run, self.training, self.output)
        self.assertEqual(state['fixed_validation_windows'], 5)
        self.assertTrue(all(row['status'] == 'waiting_for_evaluation' for row in state['gates']))

    def test_missing_training_is_pending_not_a_claimed_count(self):
        (self.training / 'split.json').unlink()
        state = scan(self.run, self.training, self.output)
        self.assertIsNone(state['fixed_validation_windows'])
        self.assertEqual(state['validation_window_metadata']['status'], 'waiting_for_training_split')

    def test_foreign_manifest_rejected(self):
        self.split['manifest_sha256'] = 'wrong'
        put(self.training / 'split.json', self.split)
        with self.assertRaisesRegex(ValueError, 'manifest identity'):
            validation_windows(self.training, self.manifest)

    def test_changed_episode_split_rejected(self):
        self.split['tasks'][0]['validation_episodes'] = [3]
        put(self.training / 'split.json', self.split)
        with self.assertRaisesRegex(ValueError, 'validation_episodes changed'):
            validation_windows(self.training, self.manifest)

    def test_duplicate_window_rejected(self):
        self.split['tasks'][0]['validation_indices'] = [15, 15]
        put(self.training / 'split.json', self.split)
        with self.assertRaisesRegex(ValueError, 'Duplicate fixed'):
            validation_windows(self.training, self.manifest)

    def test_inconsistent_flattened_windows_rejected(self):
        self.split['validation_indices'][-1] = [1, 99]
        put(self.training / 'split.json', self.split)
        with self.assertRaisesRegex(ValueError, 'Flattened validation'):
            validation_windows(self.training, self.manifest)

    def test_cached_wrong_scene_protocol_not_reported_as_complete(self):
        source = self.run / 'development_10000/report.json'
        put(source, {})
        signature = hashlib.sha256(source.read_bytes()).hexdigest()
        cached = self.output / 'development_10000' / f'review_{signature}.json'
        put(cached, dict(status='complete', scene_protocol=protocol_spec()))
        with patch('scripts.robocasa365.watch_evaluation_reviews.review') as reviewer:
            state = scan(self.run, self.training, self.output)
            reviewer.assert_not_called()
        self.assertEqual(state['gates'][0]['status'], 'review_failed')
        self.assertIn('scene protocol differs', state['gates'][0]['error'])


if __name__ == '__main__':
    unittest.main()
