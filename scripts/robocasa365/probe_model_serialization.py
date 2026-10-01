"""CPU-only causal probe of MuJoCo model XML serialization precision.

Fresh and XML-restored scenes are tested separately. Three physics rollouts get
the same complete integration state and identical held actuator commands: two
share the native model, one uses the model exported and compiled again. These
are physics probes, not GT action playback or policy success-rate trials.
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


FIELDS=(
    'body_pos','body_quat','body_mass','body_inertia','body_ipos','body_iquat',
    'jnt_pos','jnt_axis','jnt_stiffness','dof_damping','dof_frictionloss','dof_armature',
    'geom_pos','geom_quat','geom_size','geom_friction','geom_solref','geom_solimp',
    'geom_margin','geom_gap','pair_solref','pair_solimp','actuator_gainprm',
    'actuator_biasprm','actuator_ctrlrange','actuator_forcerange','eq_data',
    'qpos0','qpos_spring','body_invweight0','dof_invweight0','tendon_length0',
)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def probe(env, folder, steps):
    folder.mkdir()
    native=env.sim.model._model
    xml=env.sim.model.get_xml()
    (folder/'serialized.xml').write_text(xml)
    restored=mujoco.MjModel.from_xml_string(xml)
    dimensions=['nq','nv','nu','na','nbody','njnt','ngeom','nsite','neq','nmocap','nuserdata']
    assert all(getattr(native,key)==getattr(restored,key) for key in dimensions)
    for kind,number in [(mujoco.mjtObj.mjOBJ_BODY,native.nbody),(mujoco.mjtObj.mjOBJ_JOINT,native.njnt),
                        (mujoco.mjtObj.mjOBJ_GEOM,native.ngeom),(mujoco.mjtObj.mjOBJ_ACTUATOR,native.nu)]:
        assert all(mujoco.mj_id2name(native,kind,i)==mujoco.mj_id2name(restored,kind,i) for i in range(number))
    differences={}
    for name in FIELDS:
        first=np.asarray(getattr(native,name));second=np.asarray(getattr(restored,name))
        assert first.shape==second.shape
        error=np.abs(first-second)
        differences[name]=dict(equal=bool(np.array_equal(first,second)),changed_values=int(np.count_nonzero(error)),
            max_abs_error=float(error.max()) if error.size else 0.)
    integration=int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state=np.empty(mujoco.mj_stateSize(native,integration))
    mujoco.mj_getState(native,env.sim.data._data,state,integration)
    assert len(state)==mujoco.mj_stateSize(restored,integration)
    models=[native,native,restored];data=[mujoco.MjData(model) for model in models]
    fixed_ctrl=env.sim.data.ctrl.copy();traces=[[],[],[]]
    for model,datum in zip(models,data):
        mujoco.mj_setState(model,datum,state,integration)
        mujoco.mj_forward(model,datum)
    for _ in range(steps):
        for model,datum,trace in zip(models,data,traces):
            mujoco.mj_step1(model,datum)
            datum.ctrl[:]=fixed_ctrl
            mujoco.mj_step2(model,datum)
            trace.append(np.concatenate(([datum.time],datum.qpos,datum.qvel)))
    traces=[np.asarray(value) for value in traces]
    for value in traces:assert np.isfinite(value).all()
    repeat=np.abs(traces[0]-traces[1]);change=np.abs(traces[0]-traces[2])
    assert not np.any(repeat), 'Native-model repeat failed; cannot interpret model serialization comparison'
    first=np.flatnonzero(np.any(change>0,axis=1))
    np.savez_compressed(folder/'physics_traces.npz',native=traces[0],native_repeat=traces[1],
                        recompiled_xml=traces[2],integration_state=state,held_actuator_ctrl=fixed_ctrl)
    result=dict(dimensions={key:int(getattr(native,key)) for key in dimensions},
        model_field_differences=differences,changed_model_fields=[key for key,value in differences.items() if not value['equal']],
        physics_steps=steps,seconds=steps*float(native.opt.timestep),same_model_repeat_max_error=float(repeat.max()),
        roundtrip_first_different_physics_step=int(first[0]+1) if len(first) else None,
        roundtrip_qpos_max_error=float(change[:,1:1+native.nq].max()),
        roundtrip_qvel_max_error=float(change[:,1+native.nq:].max()),
        serialized_xml_sha256=digest(folder/'serialized.xml'),integration_state_size=len(state),
        recorded_flat_state_size=len(env.sim.get_state().flatten()),
        scope='Same full integration state and held motor controls, no policy/GT action rollout. Native repeat is the numerical control.')
    save(folder/'report.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',type=int,nargs='*',default=[0,3])
    parser.add_argument('--physics-steps',type=int,default=250)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Use explicit CPU-only execution')
    if args.physics_steps<1:raise ValueError('Positive physics-step count required')
    args.output.mkdir(parents=True,exist_ok=False)
    meta=LU.get_env_metadata(args.dataset)
    report=dict(status='running',created_utc=datetime.now(timezone.utc).isoformat(),mujoco_version=mujoco.__version__,
        robosuite_version=robosuite.__version__,source_sha256=digest(Path(__file__)),cases=[],new_policy_trials=0,
        gt_replay_fixed=False,shared_simulator_modified=False)
    save(args.output/'report.json',report)
    for episode in [None,*args.episodes]:
        kw=copy.deepcopy(meta['env_kwargs']);kw.update(env_name=meta['env_name'],seed=0,
            has_renderer=False,has_offscreen_renderer=False,use_camera_obs=False)
        env=robosuite.make(**kw)
        try:
            if episode is None:
                env.reset();label='fresh_native'
            else:
                states=LU.get_episode_states(args.dataset,episode)
                reset_to(env,dict(states=states[0],model=LU.get_episode_model_xml(args.dataset,episode),
                    ep_meta=json.dumps(LU.get_episode_meta(args.dataset,episode))))
                assert np.array_equal(env.sim.get_state().flatten(),states[0])
                label=f'recorded_xml_{episode:06d}'
            result=probe(env,args.output/label,args.physics_steps)
            report['cases'].append(dict(label=label,**result));save(args.output/'report.json',report)
            print(json.dumps({key:value for key,value in report['cases'][-1].items() if key!='model_field_differences'}),flush=True)
        finally:env.close()
    report.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat())
    save(args.output/'report.json',report)


if __name__=='__main__':main()
