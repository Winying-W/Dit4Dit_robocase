"""Locate divergence and inspect task predicates on saved traces, without rerunning physics.

Both trace inspections inject state into a separate diagnostic environment. Neither
is an action replay result. The original action replay evidence remains immutable.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import robosuite
import robocasa.utils.lerobot_utils as LU
import robocasa.utils.object_utils as OU
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to


def inspect_trace(meta, initial, trace):
    kw=copy.deepcopy(meta['env_kwargs'])
    kw.update(env_name=meta['env_name'],has_renderer=False,has_offscreen_renderer=False,use_camera_obs=False,seed=0)
    env=robosuite.make(**kw)
    rows=[]
    try:
        reset_to(env,initial)
        for step,state in enumerate(trace,1):
            # Mirrors official state-only playback. reset_to calls update_state once.
            reset_to(env,dict(states=state))
            row=dict(step=step,success_time=int(env.success_time),contacts=int(env.sim.data.ncon),
                veg1_in_pot=bool(OU.check_obj_in_receptacle(env,'veg1','pot')),
                veg2_in_pot=bool(OU.check_obj_in_receptacle(env,'veg2','pot')),
                pot_on_target_burner=bool(env.stove.check_obj_location_on_stove(env=env,obj_name='pot',threshold=.15)==env.knob),
                spatula_grasped=bool(OU.check_obj_grasped(env,'spatula')),
                spatula_veg1_contact=bool(env.check_contact(env.objects['spatula'],env.objects['veg1'])),
                spatula_veg2_contact=bool(env.check_contact(env.objects['spatula'],env.objects['veg2'])))
            rows.append(row)
    finally:env.close()
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--traces',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',nargs='+',type=int,default=[0,3]);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    meta=LU.get_env_metadata(a.dataset);report=[]
    for ep in a.episodes:
        source=a.traces/f'episode_{ep:06d}'/'warmup'
        rows=[json.loads(x) for x in (source/'steps.jsonl').read_text().splitlines()]
        first={}
        for threshold in (1e-6,.001,.01):
            hits=[dict(step=r['step'],joint=k.split('/')[0],position_error_m=v) for r in rows for k,v in r.items() if k.endswith('/position_m') and v>threshold]
            first[str(threshold)]=hits[:1]
        per_object={}
        for name in ('veg1','veg2','pot','spatula'):
            key=f'{name}_joint0/position_m'
            per_object[name]={str(th):next((dict(step=r['step'],error_m=r[key]) for r in rows if r.get(key,0)>th),None) for th in (.001,.01)}
        states=LU.get_episode_states(a.dataset,ep)
        actual=np.load(source/'actual_states.npz')['states']
        initial=dict(states=states[0],model=LU.get_episode_model_xml(a.dataset,ep),ep_meta=json.dumps(LU.get_episode_meta(a.dataset,ep)))
        summaries={};all_traces={}
        for label,trace in [('recorded',states[1:]),('action_replay_snapshot',actual[:len(states)-1])]:
            inspected=inspect_trace(meta,initial,trace);all_traces[label]=inspected
            (a.output/f'{ep:06d}_{label}.json').write_text(json.dumps(inspected)+'\n')
            keys=[k for k,v in inspected[0].items() if isinstance(v,bool)]
            summaries[label]=dict(success_time=inspected[-1]['success_time'],predicate_true_frames={k:sum(r[k] for r in inspected) for k in keys},
                success_increment_steps=[r['step'] for i,r in enumerate(inspected) if r['success_time']>(inspected[i-1]['success_time'] if i else 0)])
        events=[]
        for step in summaries['recorded']['success_increment_steps']:
            events.append(dict(step=step,recorded=all_traces['recorded'][step-1],replayed=all_traces['action_replay_snapshot'][step-1]))
        item=dict(episode=ep,first_position_divergence=first,task_object_divergence=per_object,
            original_action_replay_success=any(r['success'] for r in rows),original_success_time=rows[-1]['success_time'],
            diagnostic_state_inspection=summaries,recorded_success_events=events,
            note='State injection for predicate diagnosis only; no change to actions, physics or official checker.')
        report.append(item);(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('REPLAY_FAILURE_ANALYSIS',json.dumps(item),flush=True)

if __name__=='__main__':main()
