import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from omegaconf import OmegaConf
from scripts.robocasa365.artifact_identity import policy_identity


class JointIdentityTest(unittest.TestCase):
    def test_saved_policy_changes_and_frozen_component_corruption_are_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root/'cosmos'
            base.mkdir()
            (base/'weights').write_bytes(b'original')
            receipt = base/'CONVERSION_VERIFIED.json'
            receipt.write_text(json.dumps(dict(files={'weights':dict(bytes=8, sha256=hashlib.sha256(b'original').hexdigest())})))
            schedule = dict(phase='joint', preflight=False, initialization_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
            (root/'run_config.json').write_text(json.dumps(dict(schedule=schedule)))
            OmegaConf.save(OmegaConf.create(dict(framework=dict(cosmos25=dict(base_model=str(base))))), root/'config.yaml')
            for name in ['joint.pt','normalization.json','data_config.yaml','dataset_manifest.json']:
                (root/name).write_bytes(b'original')
            initial = policy_identity(root/'joint.pt')
            for name in ['joint.pt','normalization.json','data_config.yaml','dataset_manifest.json']:
                (root/name).write_bytes(b'modified')
                self.assertNotEqual(policy_identity(root/'joint.pt'), initial)
                (root/name).write_bytes(b'original')
            (base/'weights').write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError, 'component changed'):
                policy_identity(root/'joint.pt')
            (base/'weights').write_bytes(b'original')
            schedule['preflight'] = True
            (root/'run_config.json').write_text(json.dumps(dict(schedule=schedule)))
            with self.assertRaisesRegex(ValueError, 'non-diagnostic'):
                policy_identity(root/'joint.pt')


if __name__ == '__main__':
    unittest.main()
