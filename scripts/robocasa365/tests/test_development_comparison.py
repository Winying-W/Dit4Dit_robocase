"""Scene and full-pair coverage adversaries; fixtures are not policy trials."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.robocasa365.compare_development_checkpoints import (
    compare, compare_initial_scenes, paired_summary, sha,
)


class SceneComparisonTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folders = [self.root / arm for arm in ['left', 'right']]
        for folder in self.folders:
            folder.mkdir()
            np.savez(folder / 'trajectory.npz', initial_state=np.arange(5, dtype=float))
            (folder / 'initial_model.xml').write_text('<mujoco><camera pos="0 0 1"/></mujoco>')

    def test_equal_initial_scene(self):
        result = compare_initial_scenes(*self.folders)
        self.assertTrue(result['comparable'])
        self.assertEqual(result['initial_state_max_error'], 0)

    def test_same_seed_different_object_position_is_not_a_pair(self):
        np.savez(self.folders[1] / 'trajectory.npz', initial_state=np.arange(5, dtype=float) + .01)
        self.assertFalse(compare_initial_scenes(*self.folders)['comparable'])

    def test_changed_camera_is_not_a_pair(self):
        (self.folders[1] / 'initial_model.xml').write_text('<mujoco><camera pos="0 0 2"/></mujoco>')
        self.assertFalse(compare_initial_scenes(*self.folders)['comparable'])

    def test_invalid_and_different_size_states_never_pass(self):
        for state in [np.array([np.nan]*5), np.array([]), np.arange(4, dtype=float), np.array(['x']*5)]:
            with self.subTest(state=state):
                np.savez(self.folders[1] / 'trajectory.npz', initial_state=state)
                self.assertFalse(compare_initial_scenes(*self.folders)['comparable'])

    def test_redundant_obj_mime_type_is_allowed(self):
        (self.folders[0] / 'initial_model.xml').write_text('<mujoco><mesh file="a.obj"/></mujoco>')
        (self.folders[1] / 'initial_model.xml').write_text('<mujoco><mesh content_type="model/obj" file="a.obj"/></mujoco>')
        result = compare_initial_scenes(*self.folders)
        self.assertTrue(result['comparable'])
        self.assertFalse(result['xml']['exact_text_match'])


class CoverageComparisonTest(unittest.TestCase):
    def setUp(self):
        outcomes = [(False, True), (True, False), (False, True), (False, False)]
        self.rows = [dict(task=task, seed=seed, left_success=left, right_success=right,
                          scene=dict(comparable=True))
                     for (task, seed), (left, right) in zip(
                         [('A', 100), ('A', 101), ('B', 100), ('B', 101)], outcomes)]

    def summary(self, rows=None):
        return paired_summary(self.rows if rows is None else rows, ['A', 'B'], [100, 101])

    def test_complete_pairs_use_seed_blocks_and_equal_task_weights(self):
        result = self.summary()
        self.assertEqual((result['gained'], result['lost']), (2, 1))
        self.assertEqual(result['paired_success_rate_delta'], .25)
        self.assertEqual(result['paired_delta_bootstrap_95'], [-.5, 1.])
        self.assertEqual(result['uncertainty']['independent_seed_blocks'], 2)

    def test_missing_or_duplicate_pairs_fail(self):
        for rows in [self.rows[:-1], self.rows + [self.rows[0]]]:
            with self.assertRaisesRegex(ValueError, 'Missing, duplicate'):
                self.summary(rows)

    def test_mismatch_suppresses_full_comparison_instead_of_dropping_case(self):
        self.rows[-1]['scene']['comparable'] = False
        result = self.summary()
        self.assertEqual(result['status'], 'scene_mismatch')
        self.assertEqual((result['planned_pairs'], result['matched_pairs']), (4, 3))
        for field in ['gained', 'lost', 'paired_success_rate_delta', 'paired_delta_bootstrap_95']:
            self.assertIsNone(result[field])


class ProtocolRefusalTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root/'manifest.json'
        self.manifest.write_text(json.dumps(dict(tasks=[dict(task='Task')],task_set='atomic_seen',
            evaluation=dict(selection_seeds=list(range(100,120)),seeds=list(range(1000,1100))))))
        self.dirs=[self.root/'left',self.root/'right']
        identity=dict(purpose='development',tasks=['Task'],seeds=list(range(100,120)),
            manifest_sha256=sha(self.manifest),execute_horizon=8,task_set='atomic_seen',
            task_scope='full_official_task_set',policy_artifacts={key:dict(sha256='same') for key in
                ['normalization','data_config','base_weights','base_config','base_statistics']})
        for directory in self.dirs:
            directory.mkdir();(directory/'identity.json').write_text(json.dumps(identity))
            (directory/'report.json').write_text(json.dumps(dict(tasks=[])))
        # These tests cover early protocol refusal only. The real auditor is
        # covered separately and used without mocks in the archived360-pair run.
        def audited(directory,*args):
            return dict(status='complete',saved_action_chain_passed=True,
                        report_sha256=sha(directory/'report.json'),identity_sha256=sha(directory/'identity.json'))
        self.mock=patch('scripts.robocasa365.compare_development_checkpoints.review',side_effect=audited)
        self.mock.start();self.addCleanup(self.mock.stop)

    def call(self):
        return compare(*self.dirs,self.manifest,10000,25000)

    def change(self, **fields):
        path=self.dirs[1]/'identity.json';data=json.loads(path.read_text());data.update(fields)
        path.write_text(json.dumps(data))

    def test_final_trials_cannot_select_strategy(self):
        self.change(purpose='final')
        with self.assertRaisesRegex(ValueError,'Final evaluation'):self.call()

    def test_execution_horizon_difference_is_refused(self):
        self.change(execute_horizon=16)
        with self.assertRaisesRegex(ValueError,'Protocol differs: execute_horizon'):self.call()

    def test_normalization_difference_is_refused(self):
        path=self.dirs[1]/'identity.json';data=json.loads(path.read_text())
        data['policy_artifacts']['normalization']['sha256']='different';path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'artifact differs: normalization'):self.call()


if __name__ == '__main__':
    unittest.main()
