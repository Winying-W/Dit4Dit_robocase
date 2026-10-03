import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from scripts.robocasa365.target_transfer import transfer_probe_split, validate_transfer_manifest


class TargetTransferTest(unittest.TestCase):
    def test_transfer_is_bound_to_training_manifest_and_normalization(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'manifest.json'
            path.write_text(json.dumps(dict(task_set='pretrain_human300')))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = dict(task_set='target50', split='target', training_manifest_sha256=digest,
                            statistics_source='unchanged human300 checkpoint normalization',
                            tasks=[dict(task=str(i)) for i in range(50)])
            split = dict(manifest_sha256=digest)
            validate_transfer_manifest(manifest, path, split)
            for field, value in [('training_manifest_sha256','different'), ('statistics_source','target stats'),
                                 ('tasks',manifest['tasks'][:49]), ('split','pretrain')]:
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_transfer_manifest({**manifest,field:value}, path, split)

    def test_target_demo_is_only_an_interface_probe_with_no_training_episodes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'extras').mkdir()
            metadata = dict(env_name='TargetTask', env_kwargs=dict(obj_instance_split='target'))
            path = root/'extras/dataset_meta.json'
            path.write_text(json.dumps(dict(env_args=metadata)))
            ds = SimpleNamespace(all_steps=[(7,t) for t in range(40)]+[(8,t) for t in range(40)],
                                 trajectory_lengths=[40,40], get_trajectory_index=lambda ep: ep-7)
            split = transfer_probe_split(SimpleNamespace(dataset=ds), root, 'TargetTask')
            self.assertEqual(split['train_episodes'], [])
            self.assertEqual(split['validation_episodes'], [7])
            for i in split['validation_indices']:
                self.assertEqual(ds.all_steps[i][0], 7)
                self.assertLess(ds.all_steps[i][1]+16, 40)
            metadata['env_kwargs']['obj_instance_split'] = 'pretrain'
            path.write_text(json.dumps(dict(env_args=metadata)))
            with self.assertRaises(ValueError):
                transfer_probe_split(SimpleNamespace(dataset=ds), root, 'TargetTask')


if __name__ == '__main__':
    unittest.main()
