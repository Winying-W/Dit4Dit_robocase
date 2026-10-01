"""Official Gym closed-loop policy trials. No demonstration actions are supplied."""
import argparse
import json
import random
from multiprocessing.connection import Client
from pathlib import Path
import time
import numpy as np
import imageio.v2 as imageio
import gymnasium as gym
import robocasa
from robocasa.utils.dataset_registry import TARGET_TASKS,get_ds_meta
from scripts.robocasa365.eval_protocol import pack,unpack,observation_arrays,action_dict,CAMERAS,expected_env_action,verify_controller_contract
from scripts.robocasa365.scene_protocol import OFFICIAL, STABLE, PROTOCOLS, protocol_spec, apply_scene_protocol


def write_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--socket',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute-horizon',type=int,default=8);p.add_argument('--seeds',nargs='+',type=int,required=True)
    p.add_argument('--task',default='StirVegetables')
    p.add_argument('--max-steps',type=int);p.add_argument('--dataset',type=Path);p.add_argument('--demo-episodes',nargs='+',type=int)
    p.add_argument('--scene-protocol',choices=PROTOCOLS,default=OFFICIAL)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.scene_protocol==STABLE and a.demo_episodes:
        raise ValueError('stable_counter_v1 is for fresh scenes, not restored demo diagnostics')
    scene_runtime=apply_scene_protocol(a.scene_protocol)
    task_set=next((name for name in ['composite_seen','atomic_seen'] if a.task in TARGET_TASKS[name]),None)
    assert task_set is not None, 'Task must belong to a supported official target set'
    official_horizon=get_ds_meta(task=a.task,split='target',source='human')['horizon']
    horizon=min(a.max_steps or official_horizon,official_horizon)
    trials=list(enumerate(a.seeds)) if not a.demo_episodes else list(enumerate(a.demo_episodes))
    report=dict(task=a.task,task_set=task_set,split='target',official_horizon=official_horizon,horizon=horizon,
        execution_horizon=a.execute_horizon,initialization='demo_initial_state' if a.demo_episodes else 'fresh_gym_target',
        status='running',episodes=[],scene_protocol=protocol_spec(a.scene_protocol),scene_protocol_runtime=scene_runtime,
        note='Official per-task horizon; GT and demo-initialized diagnostics are excluded from fresh-task success rates.')
    write_json(a.output/'evaluation.json',report)
    with Client(a.socket,family='AF_UNIX') as conn:
        for trial_id,value in trials:
            seed=a.seeds[trial_id%len(a.seeds)] if a.demo_episodes else value
            conn.send_bytes(f'SEED {seed}'.encode());assert conn.recv_bytes()==b'OK'
            np.random.seed(seed);random.seed(seed)
            assert apply_scene_protocol(a.scene_protocol)==scene_runtime, 'Scene protocol changed between trials'
            directory=a.output/f'trial_{trial_id:03d}';directory.mkdir(exist_ok=True)
            tick=time.monotonic()
            env=gym.make('robocasa/'+a.task,split='target',seed=seed,camera_widths=256,camera_heights=256)
            try:
                obs,info=env.reset(seed=seed)
                startup=None
                if a.demo_episodes:
                    import robocasa.utils.lerobot_utils as LU
                    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
                    from scripts.robocasa365.replay_utils import apply_collection_startup
                    core=env.unwrapped.env;states=LU.get_episode_states(a.dataset,value)
                    reset_to(core,dict(states=states[0],model=LU.get_episode_model_xml(a.dataset,value),ep_meta=json.dumps(LU.get_episode_meta(a.dataset,value))))
                    startup=apply_collection_startup(core,states)
                    obs=env.unwrapped.get_observation(core._get_observations(force_update=True))
                core=env.unwrapped.env
                contract=verify_controller_contract(core)
                write_json(directory/'controller_contract.json',contract)
                actual_env_actions=[];expected_action=None
                original_step=core.step
                def audited_step(incoming):
                    np.testing.assert_allclose(incoming,expected_action,rtol=0,atol=1e-7)
                    actual_env_actions.append(np.asarray(incoming).copy())
                    return original_step(incoming)
                core.step=audited_step
                initial=core.sim.get_state().flatten().copy()
                (directory/'initial_model.xml').write_text(core.sim.model.get_xml())
                write_json(directory/'episode_meta.json',core.get_ep_meta())
                actions=[];queries=0;clips=0;success=False;normalized_chunks=[];query_steps=[];query_states=[]
                terminated=truncated=False;step=0
                with imageio.get_writer(str(directory/'rollout.mp4'),fps=5,ffmpeg_params=['-threads','1']) as writer:
                    writer.append_data(np.concatenate([obs['video.'+k] for k in CAMERAS],axis=1))
                    while step<horizon and not success and not terminated and not truncated:
                        query=observation_arrays(obs);query_states.append(query['state'].copy())
                        conn.send_bytes(pack(**query))
                        if not conn.poll(180):raise TimeoutError('Policy prediction timeout')
                        response=unpack(conn.recv_bytes(1024*1024));chunk=response['actions']
                        assert chunk.shape==(16,12) and np.isfinite(chunk).all()
                        clips+=int(response['clip_count']);queries+=1
                        normalized_chunks.append(response['normalized']);query_steps.append(step)
                        for action in chunk[:a.execute_horizon]:
                            # Official converter thresholds mode/gripper at 0.5; retain all 12 dimensions.
                            expected_action=expected_env_action(action)
                            obs,reward,terminated,truncated,info=env.step(action_dict(action))
                            step+=1;actions.append(action.copy());success=bool(info['success'])
                            if step%4==0:writer.append_data(np.concatenate([obs['video.'+k] for k in CAMERAS],axis=1))
                            if step>=horizon or success or terminated or truncated:break
                        if step%200==0:print('EVAL_PROGRESS',trial_id,seed,step,horizon,success,flush=True)
                np.savez_compressed(directory/'trajectory.npz',initial_state=initial,final_state=core.sim.get_state().flatten(),
                    actions_dataset_order=np.asarray(actions),actions_actually_received_by_simulator=np.asarray(actual_env_actions),normalized_chunks=np.asarray(normalized_chunks),query_steps=np.asarray(query_steps),query_states=np.asarray(query_states))
                item=dict(trial=trial_id,seed=seed,demo_episode=value if a.demo_episodes else None,success=success,steps=step,
                    success_time=int(core.success_time) if hasattr(core,'success_time') else None,action_chain_runtime_passed=True,controller_input_type=contract['input_type'],terminated=bool(terminated),truncated=bool(truncated),policy_queries=queries,
                    clipped_components_in_predicted_chunks=clips,base_abs_max=float(np.abs(np.asarray(actions)[:,:4]).max()),
                    predicted_base_mode_positive_steps=int(np.sum(np.asarray(actions)[:,4]>=.5)),seconds=time.monotonic()-tick,startup=startup)
                write_json(directory/'result.json',item);report['episodes'].append(item)
                report.update(completed_trials=len(report['episodes']),successes=sum(x['success'] for x in report['episodes']))
                report['success_rate']=report['successes']/report['completed_trials']
                write_json(a.output/'evaluation.json',report);print('POLICY_TRIAL',json.dumps(item),flush=True)
            except Exception as exc:
                report.update(status='failed',error=repr(exc),failed_trial=trial_id)
                write_json(a.output/'evaluation.json',report)
                raise
            finally:env.close()
        report['status']='complete';write_json(a.output/'evaluation.json',report)
        conn.send_bytes(b'STOP')

if __name__=='__main__':main()
