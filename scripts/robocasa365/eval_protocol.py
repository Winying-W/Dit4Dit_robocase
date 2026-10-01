"""Serialization and RoboCasa365 schema shared by the two Python environments."""
import io
import numpy as np

CAMERAS=('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand')
STATE_KEYS=('base_position','base_rotation','end_effector_position_relative','end_effector_rotation_relative','gripper_qpos')
ACTION_KEYS=('base_motion','control_mode','end_effector_position','end_effector_rotation','gripper_close')
ACTION_WIDTHS=(4,1,3,3,1)


def pack(**arrays):
    buffer=io.BytesIO();np.savez(buffer,**arrays);return buffer.getvalue()


def unpack(data):
    with np.load(io.BytesIO(data),allow_pickle=False) as arrays:
        return {key:arrays[key].copy() for key in arrays.files}


def observation_arrays(obs):
    state=np.concatenate([obs['state.'+key] for key in STATE_KEYS]).astype(np.float32)
    if state.shape!=(16,) or not np.isfinite(state).all():raise ValueError('Invalid robot state')
    images=np.stack([obs['video.'+key] for key in CAMERAS])
    if images.shape!=(3,256,256,3) or images.dtype!=np.uint8:raise ValueError(('Invalid cameras',images.shape,images.dtype))
    return dict(state=state,images=images,language=np.asarray(obs['annotation.human.task_description']))


def action_dict(action):
    action=np.asarray(action,dtype=np.float64)
    if action.shape!=(12,) or not np.isfinite(action).all():raise ValueError('Invalid predicted action')
    result={};offset=0
    for key,width in zip(ACTION_KEYS,ACTION_WIDTHS):
        result['action.'+key]=action[offset:offset+width].copy();offset+=width
    return result


def expected_env_action(action):
    """Independent numeric oracle for the official Gym conversion (dataset order)."""
    action=np.asarray(action,dtype=np.float64).copy()
    if action.shape!=(12,) or not np.isfinite(action).all():raise ValueError('Invalid action')
    action[4]=-1. if action[4]<.5 else 1.
    action[11]=-1. if action[11]<.5 else 1.
    return action[[5,6,7,8,9,10,11,0,1,2,3,4]]


def verify_controller_contract(core):
    arm=core.robots[0].part_controllers['right']
    composite=core.robots[0].composite_controller
    assert core.action_dim==12
    assert core.control_freq==20
    assert composite.name=='HYBRID_MOBILE_BASE'
    assert arm.name=='OSC_POSE'
    assert arm.input_type=='delta', f'Unsafe action semantic mismatch: {arm.input_type}'
    assert arm.input_ref_frame=='base'
    assert arm.impedance_mode=='fixed'
    np.testing.assert_allclose(arm.input_min,[-1]*6,rtol=0,atol=0)
    np.testing.assert_allclose(arm.input_max,[1]*6,rtol=0,atol=0)
    np.testing.assert_allclose(arm.output_max,[.05,.05,.05,.5,.5,.5],rtol=0,atol=1e-12)
    np.testing.assert_allclose(arm.output_min,[-.05,-.05,-.05,-.5,-.5,-.5],rtol=0,atol=1e-12)
    assert dict(composite._action_split_indexes)==dict(right=(0,6),right_gripper=(6,7),base=(7,10),torso=(10,11)),composite._action_split_indexes
    return dict(action_dim=core.action_dim,control_hz=core.control_freq,controller=arm.name,
        input_type=arm.input_type,reference_frame=arm.input_ref_frame,rotation='axis-angle rotation-vector delta',
        input_min=arm.input_min.tolist(),input_max=arm.input_max.tolist(),output_min=arm.output_min.tolist(),output_max=arm.output_max.tolist(),
        dataset_to_env_permutation=[5,6,7,8,9,10,11,0,1,2,3,4],binary_threshold=.5,
        physical_scaling_applied_by='OSC controller only; adapter returns raw [-1,1] controller commands')
