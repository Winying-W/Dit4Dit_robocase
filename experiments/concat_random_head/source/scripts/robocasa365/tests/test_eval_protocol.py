import unittest
from types import SimpleNamespace
import numpy as np
from scripts.robocasa365.eval_protocol import pack,unpack,action_dict,observation_arrays,CAMERAS,STATE_KEYS,expected_env_action,verify_controller_contract
from scripts.robocasa365.continuation_schedule import continuation_lrs


class EvalProtocolTest(unittest.TestCase):
    def test_action_order_and_official_binary_conversion(self):
        from robocasa.wrappers.gym_wrapper import PandaOmronKeyConverter
        x=np.array([.11,.12,.13,.14,-1,.21,.22,.23,.31,.32,.33,1])
        result=PandaOmronKeyConverter.unmap_action(action_dict(x))
        np.testing.assert_array_equal(result['robot0_right'],x[5:11])
        np.testing.assert_array_equal(result['robot0_base'],x[:3])
        np.testing.assert_array_equal(result['robot0_torso'],x[3:4])
        self.assertEqual(result['robot0_base_mode'],-1)
        self.assertEqual(result['robot0_right_gripper'],1)
        x[11]=.49;x[4]=.5
        result=PandaOmronKeyConverter.unmap_action(action_dict(x))
        self.assertEqual(result['robot0_base_mode'],1)
        self.assertEqual(result['robot0_right_gripper'],-1)

    def test_npz_transport_and_camera_state_order(self):
        obs={'state.'+key:np.full(width,i,dtype=np.float32) for i,(key,width) in enumerate(zip(STATE_KEYS,[3,4,3,4,2]))}
        obs.update({'video.'+key:np.full((256,256,3),i,dtype=np.uint8) for i,key in enumerate(CAMERAS)})
        obs['annotation.human.task_description']='Stir vegetables.'
        data=observation_arrays(obs);recovered=unpack(pack(**data))
        for key in data:np.testing.assert_array_equal(data[key],recovered[key])
        np.testing.assert_array_equal(data['state'],np.repeat(np.arange(5),[3,4,3,4,2]))
        self.assertEqual(data['images'][:,0,0,0].tolist(),[0,1,2])
        with self.assertRaises(ValueError):action_dict(np.full(12,np.nan))

    def test_controller_rejects_absolute_and_wrong_scaling(self):
        arm=SimpleNamespace(name='OSC_POSE',input_type='delta',input_ref_frame='base',impedance_mode='fixed',
            input_min=np.full(6,-1.),input_max=np.ones(6),output_min=-np.array([.05]*3+[.5]*3),output_max=np.array([.05]*3+[.5]*3))
        composite=SimpleNamespace(name='HYBRID_MOBILE_BASE',_action_split_indexes=dict(right=(0,6),right_gripper=(6,7),base=(7,10),torso=(10,11)))
        core=SimpleNamespace(action_dim=12,control_freq=20,robots=[SimpleNamespace(part_controllers={'right':arm},composite_controller=composite)])
        self.assertEqual(verify_controller_contract(core)['input_type'],'delta')
        arm.input_type='absolute'
        with self.assertRaises(AssertionError):verify_controller_contract(core)
        arm.input_type='delta';arm.output_max[:3]=1.
        with self.assertRaises(AssertionError):verify_controller_contract(core)

    def test_full_gym_numeric_oracle(self):
        from robocasa.wrappers.gym_wrapper import PandaOmronKeyConverter
        rng=np.random.default_rng(0)
        for raw in rng.uniform(-1,1,(100,12)):
            parts=PandaOmronKeyConverter.unmap_action(action_dict(raw))
            received=np.concatenate([parts['robot0_right'],[parts['robot0_right_gripper']],parts['robot0_base'],parts['robot0_torso'],[parts['robot0_base_mode']]])
            np.testing.assert_array_equal(received,expected_env_action(raw))

    def test_continuation_resume_is_absolute_and_has_no_lr_jump(self):
        spec=dict(start_step=300,end_step=2000,initial_lrs=[1e-5],peak_lrs=[3e-5])
        self.assertAlmostEqual(continuation_lrs(301,spec)[0],1e-5)
        self.assertAlmostEqual(continuation_lrs(320,spec)[0],3e-5)
        self.assertAlmostEqual(continuation_lrs(2000,spec)[0],3e-6)
        full=[continuation_lrs(i,spec) for i in range(301,2001)]
        resumed=[continuation_lrs(i,spec) for i in range(501,2001)]
        self.assertEqual(full[200:],resumed)

if __name__=='__main__':unittest.main()
