"""Small synthetic checkpoint adversaries; no DiT training or policy results."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.robocasa365.audit_distributed_checkpoint import verify_initialization
from scripts.robocasa365.watch_checkpoint_backups import backup_step, copy_verified, sha256


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class CheckpointBackupTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / 'source.bin'
        self.source.write_bytes(b'original checkpoint bytes')
        self.target = self.root / 'backup/checkpoint.bin'

    def test_verified_copy_is_idempotent_and_leaves_source(self):
        before = self.source.read_bytes()
        first = copy_verified(self.source, self.target, sha256(self.source))
        second = copy_verified(self.source, self.target, sha256(self.source))
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(self.target.read_bytes(), before)
        self.assertTrue(first['new_copy'])
        self.assertFalse(second['new_copy'])
        self.assertEqual(first['sha256'], first['readback_sha256'])

    def test_conflicting_backup_is_never_overwritten(self):
        self.target.parent.mkdir()
        self.target.write_bytes(b'another checkpoint')
        with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
            copy_verified(self.source, self.target)
        self.assertEqual(self.target.read_bytes(), b'another checkpoint')

    def test_source_mutation_during_copy_is_not_published(self):
        def changed_digest(path):
            if '.copying-' in path.name:
                self.source.write_bytes(b'new concurrent source')
            return sha256(path)
        with patch('scripts.robocasa365.watch_checkpoint_backups.sha256', side_effect=changed_digest):
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                copy_verified(self.source, self.target)
        self.assertFalse(self.target.exists())
        self.assertFalse(list(self.target.parent.glob('*.copying-*')))

    def test_wrong_accepted_hash_is_not_published(self):
        with self.assertRaisesRegex(ValueError, 'accepted artifact'):
            copy_verified(self.source, self.target, '0' * 64)
        self.assertFalse(self.target.exists())

    def make_training_fixture(self):
        self.run = self.root / 'run'
        self.training = self.root / 'training'
        self.destination = self.root / 'destination'
        self.manifest = dict(task_set='composite_seen', tasks=[])
        put(self.run / 'prepared/manifest.json', self.manifest)
        put(self.destination / 'run/prepared/manifest.json', self.manifest)
        initialization = dict(protocol='atomic_to_composite_action_warmstart_v1', checkpoint='/source/atomic50k.pt',
                              checkpoint_sha256='a' * 64, source_run='/source/atomic', source_global_step=50000,
                              local_initial_step=0)
        put(self.run / 'plan.json', dict(warm_start=initialization))
        lineage = {k: initialization[k] for k in ('protocol', 'checkpoint', 'checkpoint_sha256', 'source_run')}
        lineage.update(global_step=50000, transferred_tensors=247, optimizer_reset=True,
                       scheduler_reset=True, sampler_reset=True, local_steps_start_at_zero=True)
        self.initialization = initialization
        split = dict(task_set='composite_seen', manifest_sha256=sha256(self.run / 'prepared/manifest.json'))
        schedule = dict(world_size=4, initialization_checkpoint_sha256='a' * 64)
        self.config = dict(base_checkpoint='/source/gr1.pt', training_schedule=schedule, warm_start=lineage)
        for name, value in [('run_config.json', self.config), ('split.json', split),
                            ('dataset_manifest.json', self.manifest), ('normalization.json', {}),
                            ('trainer_source.json', {})]:
            put(self.training / name, value)
        (self.training / 'data_config.yaml').write_text('action_horizon: 16\n')
        weights = {f'action_model.fixture_{index}': torch.ones(1) for index in range(247)}
        put(self.training / 'action_trainable.json', {name: [1] for name in weights})
        rngs = []
        for rank in range(4):
            rngs.append(dict(sampler=np.random.default_rng(rank).bit_generator.state,
                             python=random.Random(rank).getstate(), numpy=np.random.RandomState(rank).get_state(),
                             torch=torch.Generator().manual_seed(rank).get_state(),
                             cuda=torch.zeros(4, dtype=torch.uint8)))
        self.payload = dict(format_version=2, phase='action', global_step=2, phase_step=2,
                            trained_state=weights, split=split, base_checkpoint=self.config['base_checkpoint'],
                            training_schedule=schedule, normalization_sha256=sha256(self.training / 'normalization.json'),
                            warm_start=lineage, distributed_rng=rngs,
                            optimizer=dict(param_groups=[dict(params=list(range(247)), lr=1e-4)],
                                           state={index: dict(step=torch.tensor(2.), exp_avg=torch.zeros(1),
                                                              exp_avg_sq=torch.ones(1)) for index in range(247)}),
                            scheduler=dict(last_epoch=2, _last_lr=[1e-4]))
        self.checkpoint = self.training / 'action_step_000002.pt'
        torch.save(self.payload, self.checkpoint)
        put(self.training / 'latest.json', dict(global_step=2, checkpoint_readback_passed=True))

    def test_snapshot_waits_for_trainer_publication(self):
        self.make_training_fixture()
        self.checkpoint.write_bytes(b'incomplete snapshot being written')
        put(self.training / 'latest.json', dict(global_step=1, checkpoint_readback_passed=True))
        result = backup_step(self.run, self.training, self.destination, 2)
        self.assertEqual(result['status'], 'waiting_for_checkpoint_publication')
        self.assertFalse((self.destination / 'training/action_step_000002.pt').exists())

    def test_full_synthetic_copy_and_cpu_readback(self):
        self.make_training_fixture()
        result = backup_step(self.run, self.training, self.destination, 2)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['cpu_readback']['global_step'], 2)
        self.assertEqual(result['cpu_readback']['initialization']['source_global_step'], 50000)
        self.assertEqual(result['cpu_readback']['trained_tensors'], 247)
        self.assertEqual(backup_step(self.run, self.training, self.destination, 2), result)

    def test_nonfinite_checkpoint_never_gets_completion_receipt(self):
        self.make_training_fixture()
        next(iter(self.payload['trained_state'].values())).fill_(float('nan'))
        torch.save(self.payload, self.checkpoint)
        with self.assertRaises(AssertionError):
            backup_step(self.run, self.training, self.destination, 2)
        self.assertFalse((self.destination / 'checkpoint_000002.json').exists())

    def test_warm_start_requires_explicit_plan(self):
        self.make_training_fixture()
        with self.assertRaisesRegex(AssertionError, 'explicit run plan'):
            verify_initialization(self.payload, self.config, None)

    def test_wrong_warm_start_source_rejected(self):
        self.make_training_fixture()
        wrong = dict(self.initialization, checkpoint_sha256='b' * 64)
        with self.assertRaisesRegex(AssertionError, 'checkpoint_sha256'):
            verify_initialization(self.payload, self.config, wrong)

    def test_inherited_optimizer_is_not_a_fresh_composite_run(self):
        self.make_training_fixture()
        changed = copy.deepcopy(self.payload)
        changed['warm_start']['optimizer_reset'] = False
        config = dict(self.config, warm_start=changed['warm_start'])
        with self.assertRaises(AssertionError):
            verify_initialization(changed, config, self.initialization)


if __name__ == '__main__':
    unittest.main()
