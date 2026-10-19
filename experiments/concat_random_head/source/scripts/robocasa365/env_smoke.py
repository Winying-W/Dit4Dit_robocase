"""RoboCasa365 environment-only reset/step/video acceptance check."""
import argparse
import json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import gymnasium as gym
import robocasa
from robocasa.utils.dataset_registry import TARGET_TASKS, get_ds_meta
from scripts.robocasa365.eval_protocol import observation_arrays,action_dict,expected_env_action,verify_controller_contract

p = argparse.ArgumentParser()
p.add_argument('--task', default='StirVegetables')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--steps', type=int, default=120)
p.add_argument('--allow-mobile', action='store_true')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
task_set=next((name for name in ['composite_seen','atomic_seen'] if a.task in TARGET_TASKS[name]),None)
assert task_set is not None
meta = get_ds_meta(task=a.task, split='target', source='human')
env = gym.make(f'robocasa/{a.task}', split='target', seed=0,
               camera_widths=256, camera_heights=256)
try:
    obs, info = env.reset()
    rng=np.random.default_rng(0)
    core=env.unwrapped.env;contract=verify_controller_contract(core)
    original_step=core.step;expected=None
    def audited_step(incoming):
        np.testing.assert_allclose(incoming,expected,rtol=0,atol=1e-7)
        return original_step(incoming)
    core.step=audited_step
    observation_arrays(obs)
    report = dict(task=a.task, split='target', task_set=task_set,controller=contract,
                  horizon=meta['horizon'], observation_shapes={k:list(np.asarray(v).shape) for k,v in obs.items()},
                  action_space=str(env.action_space), steps=0)
    print(json.dumps(report, indent=2), flush=True)
    with imageio.get_writer(str(a.output/'random_rollout.mp4'), fps=20,ffmpeg_params=['-threads','1']) as writer:
        for t in range(a.steps):
            raw=rng.uniform(-1,1,12)
            if not a.allow_mobile:raw[:4]=0
            expected=expected_env_action(raw)
            obs, reward, terminated, truncated, info = env.step(action_dict(raw))
            arrays=observation_arrays(obs)
            assert arrays['images'].shape==(3,256,256,3) and arrays['state'].shape==(16,)
            assert np.isfinite(arrays['state']).all() and 'success' in info
            writer.append_data(np.concatenate(list(arrays['images']),axis=1))
            report['steps'] += 1
            if (t+1)%20 == 0:
                print(f'step {t+1}/{a.steps}', flush=True)
            if terminated or truncated:break
    report['passed'] = report['steps'] == a.steps or bool(terminated) or bool(truncated)
    report['action_chain_runtime_passed']=True
    report['scope']='Environment reset/step/observation/controller/video smoke only; not learned policy or task success.'
    (a.output/'env.json').write_text(json.dumps(report, indent=2)+'\n')
finally:
    env.close()
