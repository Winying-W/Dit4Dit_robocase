"""Isolate omitted integration state on the same MuJoCo model, without XML export.

This is a held-motor physics probe, not a GT replay repair or a policy evaluation.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import robosuite
import robocasa.utils.lerobot_utils as LU
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from replay_utils import apply_collection_startup


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def capture(model,data,mask):
    state=np.empty(mujoco.mj_stateSize(model,mask))
    mujoco.mj_getState(model,data,state,mask)
    return state


def probe(env,folder,ticks):
    folder.mkdir()
    model=env.sim.model._model;native=env.sim.data._data
    full_mask=int(mujoco.mjtState.mjSTATE_INTEGRATION)
    physics_mask=int(mujoco.mjtState.mjSTATE_TIME|mujoco.mjtState.mjSTATE_QPOS|
                     mujoco.mjtState.mjSTATE_QVEL|mujoco.mjtState.mjSTATE_ACT)
    full=capture(model,native,full_mask);physics=capture(model,native,physics_mask)
    assert np.array_equal(physics,env.sim.get_state().flatten())
    ctrl=native.ctrl.copy();warmstart=native.qacc_warmstart.copy()
    traces={};initial={}
    labels=['full','full_repeat','full_without_warmstart','physics_only','physics_with_warmstart']
    for label in labels:
        datum=mujoco.MjData(model)
        is_full=label.startswith('full')
        mujoco.mj_setState(model,datum,full if is_full else physics,full_mask if is_full else physics_mask)
        if label=='full_without_warmstart':datum.qacc_warmstart[:]=0
        if label=='physics_with_warmstart':datum.qacc_warmstart[:]=warmstart
        datum.ctrl[:]=ctrl
        initial[label]=dict(qpos_exact=bool(np.array_equal(datum.qpos,native.qpos)),
            qvel_exact=bool(np.array_equal(datum.qvel,native.qvel)),
            warmstart_max_error=float(np.max(np.abs(datum.qacc_warmstart-warmstart))))
        assert initial[label]['qpos_exact'] and initial[label]['qvel_exact']
        mujoco.mj_forward(model,datum)
        trace=[]
        for _ in range(ticks):
            mujoco.mj_step1(model,datum);datum.ctrl[:]=ctrl;mujoco.mj_step2(model,datum)
            trace.append(np.concatenate(([datum.time],datum.qpos,datum.qvel)))
        traces[label]=np.asarray(trace);assert np.isfinite(traces[label]).all()
    assert np.array_equal(traces['full'],traces['full_repeat']),'Full-state repeat must be exact'
    comparisons={}
    for label in labels[1:]:
        delta=traces[label]-traces['full'];different=np.flatnonzero(np.any(delta!=0,axis=1))
        free={}
        for joint in range(model.njnt):
            if int(model.jnt_type[joint])==int(mujoco.mjtJoint.mjJNT_FREE):
                offset=1+int(model.jnt_qposadr[joint])
                name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,joint)
                free[name]=float(np.linalg.norm(delta[:,offset:offset+3],axis=1).max())
        comparisons[label]=dict(first_different_physics_tick=int(different[0]+1) if len(different) else None,
            qpos_max_abs=float(np.abs(delta[:,1:1+model.nq]).max()),
            qvel_max_abs=float(np.abs(delta[:,1+model.nq:]).max()),
            free_joint_xyz_max_error_m=free)
    np.savez_compressed(folder/'traces.npz',**traces,full_integration_state=full,
                        recorded_state_format=physics,held_ctrl=ctrl,warmstart=warmstart)
    result=dict(physics_ticks=ticks,seconds=ticks*float(model.opt.timestep),integrator=int(model.opt.integrator),
        full_state_size=len(full),recorded_state_format_size=len(physics),warmstart_norm=float(np.linalg.norm(warmstart)),
        qfrc_applied_norm=float(np.linalg.norm(native.qfrc_applied)),xfrc_applied_norm=float(np.linalg.norm(native.xfrc_applied)),
        initial_states=initial,comparisons=comparisons,same_compiled_model=True,xml_roundtrip=False)
    save(folder/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',type=int,nargs='+',default=[0,3,11])
    parser.add_argument('--physics-ticks',type=int,default=250)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Explicit CPU-only execution required')
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',created_utc=datetime.now(timezone.utc).isoformat(),mujoco_version=mujoco.__version__,
        robosuite_version=robosuite.__version__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases=[],new_policy_trials=0,gt_replay_fixed=False,shared_simulator_modified=False,
        scope='Same native model and held motor controls. New locally generated snapshots in the recorded state format, not hidden state recovered from the original demonstrations. Tests a possible source of drift, not its attribution or a repair.')
    save(args.output/'report.json',report);meta=LU.get_env_metadata(args.dataset)
    for episode in args.episodes:
        kw=copy.deepcopy(meta['env_kwargs']);kw.update(env_name=meta['env_name'],seed=0,
            has_renderer=False,has_offscreen_renderer=False,use_camera_obs=False)
        env=robosuite.make(**kw)
        try:
            states=LU.get_episode_states(args.dataset,episode);actions=LU.get_episode_actions(args.dataset,episode)
            reset_to(env,dict(states=states[0],model=LU.get_episode_model_xml(args.dataset,episode),
                ep_meta=json.dumps(LU.get_episode_meta(args.dataset,episode))))
            assert np.array_equal(env.sim.get_state().flatten(),states[0])
            startup=apply_collection_startup(env,states)
            for prefix in [0,32]:
                if prefix:
                    for action in actions[:prefix]:env.step(action)
                label=f'episode_{episode:06d}_prefix_{prefix:03d}'
                result=probe(env,args.output/label,args.physics_ticks)
                report['cases'].append(dict(episode=episode,gt_prefix_steps=prefix,startup=startup,**result))
                save(args.output/'report.json',report)
                print(json.dumps(dict(case=label,comparisons=result['comparisons'])),flush=True)
        finally:env.close()
    report.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat());save(args.output/'report.json',report)


if __name__=='__main__':main()
