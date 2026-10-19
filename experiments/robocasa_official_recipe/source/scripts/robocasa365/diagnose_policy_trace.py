"""Replay immutable saved policy commands and locate missing task milestones.

Never count these diagnostic replays as new policy trials. Initial/query/final
states are compared to the original run before interpreting predicates.
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import gymnasium as gym
import robocasa
import robocasa.utils.lerobot_utils as LU
import robocasa.utils.object_utils as OU
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from scripts.robocasa365.replay_utils import apply_collection_startup
from scripts.robocasa365.eval_protocol import action_dict, observation_arrays, verify_controller_contract


def predicates(core):
    return {**{name+'_in_pot':bool(OU.check_obj_in_receptacle(core,name,'pot')) for name in ('veg1','veg2')},
        **{name+'_grasped':bool(OU.check_obj_grasped(core,name)) for name in ('veg1','veg2','spatula')},
        'pot_on_target_burner':bool(core.stove.check_obj_location_on_stove(env=core,obj_name='pot',threshold=.15)==core.knob)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--episode',type=int,default=5)
    p.add_argument('--seed',type=int,default=100)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    trace=np.load(a.reference/'trajectory.npz',allow_pickle=False)
    actions=trace['actions_dataset_order']
    np.random.seed(a.seed);random.seed(a.seed)
    env=gym.make('robocasa/StirVegetables',split='target',seed=a.seed,camera_widths=256,camera_heights=256)
    try:
        env.reset(seed=a.seed);core=env.unwrapped.env;states=LU.get_episode_states(a.dataset,a.episode)
        reset_to(core,dict(states=states[0],model=LU.get_episode_model_xml(a.dataset,a.episode),
                         ep_meta=json.dumps(LU.get_episode_meta(a.dataset,a.episode))))
        startup=apply_collection_startup(core,states)
        obs=env.unwrapped.get_observation(core._get_observations(force_update=True))
        verify_controller_contract(core)
        initial_error=float(np.max(np.abs(core.sim.get_state().flatten()-trace['initial_state'])))
        assert initial_error < 1e-10, initial_error
        query_lookup={int(step):i for i,step in enumerate(trace['query_steps'])} if 'query_steps' in trace else {}
        rows=[];query_errors=[];first_success=None
        for step,action in enumerate(actions):
            if step in query_lookup:
                error=float(np.max(np.abs(observation_arrays(obs)['state']-trace['query_states'][query_lookup[step]])))
                query_errors.append(dict(step=step,state_max_error=error))
            obs,_,terminated,truncated,info=env.step(action_dict(action))
            if info['success'] and first_success is None:first_success=step+1
            rows.append(dict(step=step+1,**predicates(core),success=bool(info['success']),
                             positions={name:core.sim.data.body_xpos[core.obj_body_id[name]].tolist()
                                        for name in ('veg1','veg2','spatula','pot')}))
            if (step+1)%200==0:print('DIAGNOSTIC_REPLAY',step+1,len(actions),flush=True)
            if terminated or truncated:break
        final_error=float(np.max(np.abs(core.sim.get_state().flatten()-trace['final_state'])))
        max_query_error=max((row['state_max_error'] for row in query_errors),default=0.)
        summary=dict(reference=str(a.reference),steps=len(rows),first_success_step=first_success,startup=startup,
            initial_state_max_error=initial_error,final_state_max_error=final_error,
            max_query_state_error=max_query_error,
            trajectory_matches_reference=bool(final_error < 1e-8 and max_query_error < 1e-6),
            first_true_step={key:next((row['step'] for row in rows if row[key]),None) for key in predicates(core)},
            final_predicates={key:rows[-1][key] for key in predicates(core)},
            scope='Diagnostic playback of saved commands, not fresh policy inference; interpret original-run predicates only if trajectory matches reference.')
        (a.output/'predicates.json').write_text(json.dumps(rows)+'\n')
        (a.output/'query_errors.json').write_text(json.dumps(query_errors)+'\n')
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary),flush=True)
    finally:env.close()


if __name__=='__main__':main()
