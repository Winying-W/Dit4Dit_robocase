"""Protocol identity adversaries, not simulator or policy success fixtures."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.robocasa365 import scene_protocol as protocol


class SceneProtocolRecordTest(unittest.TestCase):
    def test_legacy_original_records_remain_reviewable(self):
        self.assertEqual(protocol.validate_protocol_records({}, {}, {}), protocol.protocol_spec())

    def test_mixed_original_and_stable_cannot_be_pooled(self):
        stable = dict(scene_protocol=protocol.protocol_spec(protocol.STABLE))
        with self.assertRaisesRegex(ValueError, 'Scene protocol differs'):
            protocol.validate_protocol_records(stable, stable, {})

    def test_stable_label_without_actual_receipt_is_rejected(self):
        stable = dict(scene_protocol=protocol.protocol_spec(protocol.STABLE))
        with self.assertRaisesRegex(ValueError, 'lacks simulator application receipt'):
            protocol.validate_protocol_records(stable, stable, stable)

    def test_changed_recipe_or_effective_code_cannot_pass(self):
        stable = dict(scene_protocol=protocol.protocol_spec(protocol.STABLE))
        runtime = dict(protocol=deepcopy(stable['scene_protocol']), source_sha256=protocol.COUNTER_SHA256,
                       effective_function_sha256=protocol.STABLE_FUNCTION_SHA256, installed_file_unchanged=True,
                       python_hash_seed='0')
        evaluation = dict(stable, scene_protocol_runtime=runtime)
        self.assertEqual(protocol.validate_protocol_records(stable, stable, evaluation), stable['scene_protocol'])
        runtime['effective_function_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'counter function differs'):
            protocol.validate_protocol_records(stable, stable, evaluation)
        changed = deepcopy(stable)
        changed['scene_protocol']['counter_region_patch']['replacement'] = 'remove all geoms'
        with self.assertRaisesRegex(ValueError, 'altered scene protocol recipe'):
            protocol.validate_protocol_records(changed, changed, changed)

    def test_hash_seed_is_set_before_child_start_and_gpu_binding_is_retained(self):
        parent = dict(CUDA_VISIBLE_DEVICES='3', MUJOCO_EGL_DEVICE_ID='3', PYTHONHASHSEED='random')
        child = protocol.simulator_environment(protocol.STABLE, parent)
        self.assertEqual(child, dict(parent, PYTHONHASHSEED='0'))
        self.assertEqual(parent['PYTHONHASHSEED'], 'random')
        self.assertEqual(protocol.simulator_environment(protocol.OFFICIAL, parent), parent)


class SceneProtocolProcessTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'counter_fixture.py'
        self.path.write_text('class Counter:\n    def get_reset_regions(self):\n'
                             '        valid_geoms = []\n'
                             '        valid_geoms = list(set(valid_geoms))\n'
                             '        return valid_geoms\n')
        spec = importlib.util.spec_from_file_location('counter_fixture', self.path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.original = self.module.Counter.get_reset_regions
        self.addCleanup(patch.stopall)
        patch.object(protocol, '_counter_module', return_value=self.module).start()
        patch.object(protocol, '_installed', None).start()
        patch.dict(os.environ, PYTHONHASHSEED='0').start()

    def test_default_does_not_replace_function_or_modify_source(self):
        before = self.path.read_bytes()
        receipt = protocol.apply_scene_protocol()
        self.assertIs(self.module.Counter.get_reset_regions, self.original)
        self.assertEqual(before, self.path.read_bytes())
        self.assertEqual(receipt, protocol.apply_scene_protocol())
        receipt['protocol']['name'] = 'caller mutation'
        self.assertEqual(protocol.apply_scene_protocol()['protocol']['name'], protocol.OFFICIAL)

    def test_unreviewed_source_refused_before_mutation(self):
        with self.assertRaisesRegex(RuntimeError, 'reviewed stable_counter_v1 revision'):
            protocol.apply_scene_protocol(protocol.STABLE)
        self.assertIs(self.module.Counter.get_reset_regions, self.original)

    def test_protocol_or_function_cannot_change_midprocess(self):
        protocol.apply_scene_protocol()
        with self.assertRaisesRegex(RuntimeError, 'Cannot change scene protocol'):
            protocol.apply_scene_protocol(protocol.STABLE)
        self.module.Counter.get_reset_regions = lambda self: {}
        with self.assertRaisesRegex(RuntimeError, 'function changed'):
            protocol.apply_scene_protocol()

    def test_source_change_between_resets_is_detected(self):
        protocol.apply_scene_protocol()
        self.path.write_text(self.path.read_text() + '# changed\n')
        with self.assertRaisesRegex(RuntimeError, 'counter source inside'):
            protocol.apply_scene_protocol()


if __name__ == '__main__':
    unittest.main()
