"""Reject an A/B conclusion when the actual scenes or declared controls differ."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.robocasa365.scene_protocol import STABLE, protocol_spec
from scripts.robocasa365.select_ab import main


class PairedSelectionTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.run=self.root/'run';self.train=self.root/'train'
        specification=protocol_spec(STABLE);recipe=specification['counter_region_patch']
        receipt=dict(protocol=specification,installed_file_unchanged=True,
            source_sha256=recipe['source_sha256'],effective_function_sha256=recipe['effective_function_sha256'],python_hash_seed='0')
        for arm in ['frozen','partial_joint']:
            folder=self.run/'ab'/arm/'eval';folder.mkdir(parents=True)
            config_folder=self.train/arm;config_folder.mkdir(parents=True)
            self.write(folder/'evaluation.json',dict(status='complete',completed_trials=10,
                initialization='fresh_gym_target',horizon=2400,official_horizon=2400,
                episodes=[dict(seed=seed) for seed in range(100,110)],successes=0,
                scene_protocol=specification,scene_protocol_runtime=receipt))
            self.write(folder/'checkpoint_verification.json',dict(global_step=1000,normalization_sha256='common',
                validation_action_fm_loss=1.,scene_protocol=specification))
            self.write(folder.parent/'action_audit.json',dict(passed=True,completed_trials=10))
            self.write(config_folder/'run_config.json',dict(accumulation=4,seed=42,action_lr=1e-4,split={},
                warm_start=dict(global_step=2000,optimizer_reset=True,checkpoint='same_original'),
                training_schedule=dict(paired_ab_protocol='paired_video_action_rng_v1',paired_ab_seed=42,
                    gradient_clipping='per_optimizer_group',warmup_steps=None)))
            for trial in range(10):
                trial_folder=folder/f'trial_{trial:03d}';trial_folder.mkdir()
                np.savez(trial_folder/'trajectory.npz',initial_state=np.arange(5,dtype=float))
                (trial_folder/'initial_model.xml').write_text('<scene/>')

    @staticmethod
    def write(path, value):path.write_text(json.dumps(value))

    def select(self):
        with patch('sys.argv',['select_ab','--run',str(self.run),'--train-root',str(self.train),
                               '--paired-ab','--scene-protocol',STABLE]):
            main()

    def test_matching_scenes_and_controls_allow_conservative_tie_selection(self):
        self.select()
        result=json.loads((self.run/'ab_selection.json').read_text())
        self.assertEqual(result['recipe'],'frozen')
        self.assertEqual(result['scene_protocol'],protocol_spec(STABLE))

    def test_same_seed_different_physical_state_rejects_selection(self):
        path=self.run/'ab/partial_joint/eval/trial_003/trajectory.npz'
        np.savez(path,initial_state=np.arange(5,dtype=float)+.01)
        with self.assertRaises(AssertionError):self.select()
        self.assertFalse((self.run/'ab_selection.json').exists())

    def test_global_clipping_arm_rejects_selection(self):
        path=self.train/'partial_joint/run_config.json';data=json.loads(path.read_text())
        data['training_schedule']['gradient_clipping']='global';self.write(path,data)
        with self.assertRaises(AssertionError):self.select()

    def test_missing_simulator_protocol_receipt_rejects_selection(self):
        path=self.run/'ab/partial_joint/eval/evaluation.json';data=json.loads(path.read_text())
        del data['scene_protocol_runtime'];self.write(path,data)
        with self.assertRaisesRegex(ValueError,'receipt'):self.select()


if __name__=='__main__':unittest.main()
