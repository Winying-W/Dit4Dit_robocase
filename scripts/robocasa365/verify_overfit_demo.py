"""GT action replay using precisely the Gym initialization used for policy overfit."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import gymnasium as gym
import robocasa
import robocasa.utils.lerobot_utils as LU
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from scripts.robocasa365.replay_utils import apply_collection_startup
from scripts.robocasa365.eval_protocol import action_dict,expected_env_action,verify_controller_contract,CAMERAS,observation_arrays


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--episode',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=100)
    p.add_argument('--task',default='StirVegetables')
    p.add_argument('--allow-mobile',action='store_true',help='Replay the recorded full 12D commands, including base and torso')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    np.random.seed(a.seed);random.seed(a.seed)
    frame=pd.read_parquet(next((a.dataset/'data').glob(f'*/episode_{a.episode:06d}.parquet')))
    actions=np.stack(frame.action)
    assert actions.shape[1]==12 and np.isfinite(actions).all()
    assert LU.get_env_metadata(a.dataset)['env_name']==a.task, 'Dataset/task mismatch'
    if not a.allow_mobile:
        assert np.all(actions[:,:4]==0) and np.all(actions[:,4]==-1)
    env=gym.make('robocasa/'+a.task,split='target',seed=a.seed,camera_widths=256,camera_heights=256)
    try:
        obs,info=env.reset(seed=a.seed);core=env.unwrapped.env;states=LU.get_episode_states(a.dataset,a.episode)
        reset_to(core,dict(states=states[0],model=LU.get_episode_model_xml(a.dataset,a.episode),ep_meta=json.dumps(LU.get_episode_meta(a.dataset,a.episode))))
        startup=apply_collection_startup(core,states)
        obs=env.unwrapped.get_observation(core._get_observations(force_update=True))
        contract=verify_controller_contract(core)
        first=observation_arrays(obs)
        np.savez_compressed(a.output/'initial_observation.npz',**first)
        initial=core.sim.get_state().flatten().copy()
        (a.output/'initial_model.xml').write_text(core.sim.model.get_xml())
        received=[];expected=None;original=core.step
        def audited(incoming):
            np.testing.assert_allclose(incoming,expected,rtol=0,atol=1e-7)
            received.append(np.asarray(incoming).copy());return original(incoming)
        core.step=audited
        success=False;first_success=None;steps=0
        with imageio.get_writer(str(a.output/'rollout.mp4'),fps=5,ffmpeg_params=['-threads','1']) as writer:
            writer.append_data(np.concatenate([obs['video.'+key] for key in CAMERAS],axis=1))
            for t,action in enumerate(actions):
                expected=expected_env_action(action)
                obs,_,terminated,truncated,info=env.step(action_dict(action));steps=t+1
                if info['success'] and first_success is None:first_success=steps
                success=success or bool(info['success'])
                if steps%4==0:writer.append_data(np.concatenate([obs['video.'+key] for key in CAMERAS],axis=1))
                if steps%200==0:print('OVERFIT_GT_REPLAY',steps,len(actions),success,flush=True)
                if terminated or truncated:break
        np.savez_compressed(a.output/'trajectory.npz',initial_state=initial,final_state=core.sim.get_state().flatten(),actions_dataset_order=actions[:steps],actions_actually_received_by_simulator=np.asarray(received))
        result=dict(task=a.task,episode=a.episode,seed=a.seed,success=success,steps=steps,first_success_step=first_success,
            success_time=int(core.success_time) if hasattr(core,'success_time') else None,
            startup=startup,controller=contract,action_chain_runtime_passed=True,scope='GT action replay under policy Gym/rendering protocol; never counted as model success.')
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print('OVERFIT_GT_RESULT',json.dumps(result),flush=True)
        if not success:raise RuntimeError('Selected demo failed GT replay under policy protocol; do not train until diagnosis')
    finally:env.close()

if __name__=='__main__':main()
