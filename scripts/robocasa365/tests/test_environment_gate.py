"""Prevent CPU placeholders or stale protocols from passing the render gate."""
from copy import deepcopy
import unittest

from scripts.robocasa365.scene_protocol import STABLE, COUNTER_SHA256, STABLE_FUNCTION_SHA256, protocol_spec
from scripts.robocasa365.verify_task_environments import validate_environment_report


class EnvironmentGateTest(unittest.TestCase):
    def setUp(self):
        self.manifest=dict(task_set='composite_seen',tasks=[dict(task='StirVegetables',horizon=1000)])
        spec=protocol_spec(STABLE)
        row=dict(task='StirVegetables',horizon=1000,steps=8,passed=True,returncode=0,
                 controller=dict(action_dim=12,input_type='delta'),action_chain_runtime_passed=True,
                 rendering_enabled=False,image_std_by_camera=[0.,0.,0.],scene_protocol=spec,
                 scene_protocol_runtime=dict(protocol=deepcopy(spec),source_sha256=COUNTER_SHA256,
                     effective_function_sha256=STABLE_FUNCTION_SHA256,python_hash_seed='0',installed_file_unchanged=True))
        self.report=dict(status='complete',task_set='composite_seen',tasks=[row],scene_protocol=spec,rendering_enabled=False)

    def test_cpu_is_accepted_only_as_cpu(self):
        validate_environment_report(self.report,self.manifest,STABLE,False)
        with self.assertRaises(AssertionError):
            validate_environment_report(self.report,self.manifest,STABLE,True)

    def test_black_images_cannot_be_relabelled_rendered(self):
        self.report['rendering_enabled']=True
        self.report['tasks'][0]['rendering_enabled']=True
        with self.assertRaises(AssertionError):
            validate_environment_report(self.report,self.manifest,STABLE,True)

    def test_missing_task_or_wrong_controller_fails(self):
        self.report['tasks'][0]['controller']['input_type']='absolute'
        with self.assertRaises(AssertionError):
            validate_environment_report(self.report,self.manifest,STABLE,False)
        self.report['tasks']=[]
        with self.assertRaises(AssertionError):
            validate_environment_report(self.report,self.manifest,STABLE,False)

    def test_declared_stable_cannot_reuse_official_task_evidence(self):
        self.report['tasks'][0]['scene_protocol']=protocol_spec()
        with self.assertRaises(ValueError):
            validate_environment_report(self.report,self.manifest,STABLE,False)

    def test_runtime_receipt_is_required(self):
        self.report['tasks'][0].pop('scene_protocol_runtime')
        with self.assertRaises(ValueError):
            validate_environment_report(self.report,self.manifest,STABLE,False)


if __name__=='__main__':unittest.main()
