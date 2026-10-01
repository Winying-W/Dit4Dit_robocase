"""Test timing repair, same-runtime replay, and task predicate independently.

State-only checks below are explicitly labelled diagnostic, never action replay.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import robosuite
import robocasa.utils.lerobot_utils as LU
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from scripts.robocasa365.replay_utils import apply_collection_startup


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,nargs='+',default=[0,3,11])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    meta=LU.get_env_metadata(a.dataset)
    output=[]
    for ep in a.episodes:
        states=LU.get_episode_states(a.dataset,ep);actions=LU.get_episode_actions(a.dataset,ep)
        initial=dict(states=states[0],model=LU.get_episode_model_xml(a.dataset,ep),ep_meta=json.dumps(LU.get_episode_meta(a.dataset,ep)))
        kw=copy.deepcopy(meta['env_kwargs']);kw.update(env_name=meta['env_name'],has_renderer=False,has_offscreen_renderer=False,use_camera_obs=False,seed=0)
        env=robosuite.make(**kw)
        try:
            traces=[]
            # Two full genuine action rollouts in the same runtime, separately reset.
            for trial in range(2):
                reset_to(env,initial)
                startup=apply_collection_startup(env,states)
                trace=[]
                for action in actions:
                    env.step(action);trace.append(env.sim.get_state().flatten().copy())
                traces.append(np.asarray(trace))
            repeat_error=float(np.max(np.abs(traces[0]-traces[1])))
            item=dict(episode=ep,startup=startup,repeat_action_rollout_max_error=repeat_error,
                      repeat_action_success=bool(env._check_success()),repeat_action_success_time=int(env.success_time))
            item['repeat_exact'] = repeat_error == 0
            np.savez_compressed(a.output/f'repeat_{ep:06d}.npz', first=traces[0], second=traces[1])
            # The recorded trajectory itself must satisfy the current official predicate.
            reset_to(env,initial)
            success=False
            for state in states:
                reset_to(env,dict(states=state))
                success=success or bool(env._check_success())
            item.update(recorded_state_only_success=success,recorded_state_only_success_time=int(env.success_time),
                        note='State-only result verifies task predicate, NOT action replay success.')
            output.append(item)
            (a.output/'contract.json').write_text(json.dumps(output,indent=2)+'\n')
            print('CONTRACT',json.dumps(item),flush=True)
        finally:env.close()


if __name__=='__main__':main()
