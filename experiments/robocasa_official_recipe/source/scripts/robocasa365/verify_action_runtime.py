"""CPU-only real environment check of controller semantics; never a policy evaluation."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import robosuite
import robocasa
from robocasa.wrappers.gym_wrapper import PandaOmronKeyConverter
from scripts.robocasa365.eval_protocol import action_dict,expected_env_action,verify_controller_contract

p=argparse.ArgumentParser();p.add_argument('--env-metadata',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
meta=json.loads(a.env_metadata.read_text());kw=copy.deepcopy(meta['env_kwargs'])
kw.update(env_name=meta['env_name'],seed=0,has_renderer=False,has_offscreen_renderer=False,use_camera_obs=False)
env=robosuite.make(**kw)
try:
    env.reset()
    report=verify_controller_contract(env)
    arm=env.robots[0].part_controllers['right']
    # This calls the real controller's scaling routine, without taking a physics step.
    command=np.array([1.,-1.,.5,.4,-.3,.2])
    scaled=arm.scale_action(command)
    np.testing.assert_allclose(scaled,[.05,-.05,.025,.2,-.15,.1],atol=1e-12,rtol=0)
    zero=np.zeros(12);zero[4]=-1;zero[11]=-1
    parts=PandaOmronKeyConverter.unmap_action(action_dict(zero))
    real=np.zeros(env.action_dim)
    for part,(start,end) in env.robots[0].composite_controller._action_split_indexes.items():
        real[start:end]=parts['robot0_'+part]
    real[-1]=parts['robot0_base_mode']
    np.testing.assert_array_equal(real,expected_env_action(zero))
    env.step(real)
    assert arm.input_type=='delta' and arm._goal_update_mode=='achieved'
    report.update(passed=True,test_command=command.tolist(),actual_scaled_delta=scaled.tolist(),zero_action_real_step=True,
        goal_update_mode=arm._goal_update_mode,note='Real headless environment controller check, not trained-policy rollout.')
    (a.output/'controller_runtime.json').write_text(json.dumps(report,indent=2)+'\n')
    print('REAL_CONTROLLER_VERIFIED',json.dumps(report),flush=True)
finally:env.close()
