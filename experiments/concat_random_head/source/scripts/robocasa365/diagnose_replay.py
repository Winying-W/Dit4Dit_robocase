"""Compare genuine open-loop GT playback paths without injecting recorded states.

Every episode starts in a fresh environment. Full trajectories and per-step error
components are persisted; only the initial XML/state is restored in action modes.
"""
import argparse
import copy
import hashlib
import os
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import mujoco
import robosuite
import robocasa
import robocasa.utils.lerobot_utils as LU
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to, playback_trajectory_with_env


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def revision(path):
    r = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def controller_snapshot(env):
    result = {}
    for i, robot in enumerate(env.robots):
        for name, controller in robot.part_controllers.items():
            row = {}
            for key in ('name', 'input_type', 'input_ref_frame', '_goal_update_mode', 'lite_physics',
                        'policy_freq', 'kp', 'kd', 'initial_joint', 'goal_pos', 'goal_ori',
                        'ref_pos', 'origin_pos', 'origin_ori'):
                value = getattr(controller, key, None)
                if isinstance(value, np.ndarray): value = value.tolist()
                if isinstance(value, (str, int, float, list, bool)) or value is None: row[key] = value
            result[f'{i}/{name}'] = row
    return result


def joint_layout(env):
    m = env.sim.model
    result = []
    for j in range(m.njnt):
        qstart, vstart = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        qend = int(m.jnt_qposadr[j+1]) if j+1 < m.njnt else m.nq
        vend = int(m.jnt_dofadr[j+1]) if j+1 < m.njnt else m.nv
        result.append(dict(name=m.joint_id2name(j), type=int(m.jnt_type[j]), qstart=qstart, qend=qend, vstart=vstart, vend=vend))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, nargs='+', default=[0, 3, 5, 10, 11])
    p.add_argument('--variants', nargs='+', default=['official', 'adapter', 'float64', 'reset_controller'])
    p.add_argument('--max-steps', type=int)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    meta = LU.get_env_metadata(args.dataset)
    write_json(args.output/'environment_metadata.json', meta)
    extra = json.loads((args.dataset/'extras/dataset_meta.json').read_text())
    write_json(args.output/'dataset_metadata.json', extra)
    versions = dict(mujoco=mujoco.__version__, robosuite=robosuite.__version__,
                    robosuite_file=robosuite.__file__, robocasa_file=robocasa.__file__,
                    robosuite_commit=revision(Path(robosuite.__file__).parents[1]),
                    robocasa_commit=revision(Path(robocasa.__file__).parents[1]),
                    numpy=np.__version__, numba_disable_jit=os.environ.get('NUMBA_DISABLE_JIT','0'), numba_cpu_name=os.environ.get('NUMBA_CPU_NAME'),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(args.output/'versions.json', versions)
    print('VERSIONS', json.dumps(versions), flush=True)
    (args.output/'diagnose_replay_source.py').write_bytes(Path(__file__).read_bytes())
    all_results = []
    for ep in args.episodes:
        states = LU.get_episode_states(args.dataset, ep)
        actions = LU.get_episode_actions(args.dataset, ep)
        raw_df = pd.read_parquet(next((args.dataset/'data').glob(f'*/episode_{ep:06d}.parquet')))
        raw = np.stack(raw_df['action'])
        assert np.array_equal(actions, raw[:, [5,6,7,8,9,10,11,0,1,2,3,4]])
        write_json(args.output/f'timing_{ep:06d}.json', dict(first_times=states[:6,0].tolist(), delta_min=float(np.diff(states[:,0]).min()), delta_max=float(np.diff(states[:,0]).max()), action_dtype=str(actions.dtype)))
        initial = dict(states=states[0], model=LU.get_episode_model_xml(args.dataset, ep), ep_meta=json.dumps(LU.get_episode_meta(args.dataset, ep)))
        for variant in args.variants:
            directory = args.output/f'episode_{ep:06d}'/variant
            directory.mkdir(parents=True, exist_ok=True)
            kw = copy.deepcopy(meta['env_kwargs'])
            kw.update(env_name=meta['env_name'], has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, seed=0)
            env = robosuite.make(**kw)
            try:
                actual = []
                rows = []
                start = time.monotonic()
                original_step = env.step
                limit = min(len(actions), args.max_steps or len(actions))
                reference_actions = actions.astype(np.float64) if variant == 'float64' else actions.copy()
                layouts = None
                run_info = {}

                def observe_step(action):
                    nonlocal layouts
                    t = len(actual)
                    if t == 0:
                        layouts = joint_layout(env)
                        write_json(directory/'joints.json', layouts)
                        run_info.update(initial_full_state_max_error=float(np.max(np.abs(env.sim.get_state().flatten()-states[0]))),
                                        control_freq=env.control_freq, model_timestep=env.model_timestep,
                                        sim_timestep=env.sim.model.opt.timestep, lite_physics=env.lite_physics,
                                        solver=int(env.sim.model.opt.solver), iterations=int(env.sim.model.opt.iterations),
                                        tolerance=float(env.sim.model.opt.tolerance), controller=controller_snapshot(env))
                        write_json(directory/'initial.json', run_info)
                    if variant == 'warmup_forward': env.sim.forward()
                    if variant == 'warmup_device':
                        c=env.robots[0].part_controllers['right']
                        c.delta_to_abs_action(c.scale_action(np.zeros(6)), goal_update_mode=None)
                        c.delta_to_abs_action(c.scale_action(action[:6].copy()), goal_update_mode=None)
                    result = original_step(action)
                    current = env.sim.get_state().flatten().copy()
                    actual.append(current)
                    row = dict(step=t+1, success=bool(env._check_success()), success_time=int(env.success_time), contacts=int(env.sim.data.ncon))
                    if t+1 < len(states):
                        target = states[t+1]
                        nq = env.sim.model.nq
                        dq = current[1:1+nq]-target[1:1+nq]
                        dv = current[1+nq:]-target[1+nq:]
                        row.update(time_error=float(current[0]-target[0]), qpos_linf=float(np.abs(dq).max()), qvel_linf=float(np.abs(dv).max()),
                                   state_l2=float(np.linalg.norm(current-target)))
                        for joint in layouts:
                            name = joint['name'] or f"joint_{joint['qstart']}"
                            qdiff = dq[joint['qstart']:joint['qend']]
                            row[name+'/qpos_max'] = float(np.abs(qdiff).max())
                            row[name+'/qvel_max'] = float(np.abs(dv[joint['vstart']:joint['vend']]).max())
                            if joint['type'] == 0:  # free joint xyz followed by wxyz quaternion
                                row[name+'/position_m'] = float(np.linalg.norm(qdiff[:3]))
                                q1 = current[1+joint['qstart']+3:1+joint['qend']]
                                q2 = target[1+joint['qstart']+3:1+joint['qend']]
                                cosine = abs(np.dot(q1,q2))/(np.linalg.norm(q1)*np.linalg.norm(q2))
                                row[name+'/rotation_rad'] = float(2*np.arccos(np.clip(cosine,-1,1)))
                    rows.append(row)
                    if t < 2 or (t+1) % 250 == 0:
                        print('STEP', json.dumps(dict(episode=ep,variant=variant,**{k:v for k,v in row.items() if '/' not in k})), flush=True)
                    return result

                env.step = observe_step
                if variant in ('official', 'official_repeat'):
                    playback_trajectory_with_env(env, initial, states[:limit], actions=reference_actions[:limit], render=False, video_writer=None, verbose=False)
                else:
                    reset_to(env, initial)
                    run_info['restored_state_max_error'] = float(np.max(np.abs(env.sim.get_state().flatten()-states[0])))
                    if variant.startswith('warmup'):
                        count = 2 if variant == 'warmup_twice' else 1
                        for _ in range(count): original_step(np.zeros(env.action_dim, dtype=np.float64))
                        run_info['unrecorded_zero_steps'] = count
                    if variant == 'reset_controller':
                        for robot in env.robots:
                            for controller in robot.part_controllers.values():
                                controller.update(force=True)
                                if hasattr(controller, 'reset_goal'): controller.reset_goal()
                    if variant == 'full_physics':
                        env.lite_physics = False
                        for robot in env.robots:
                            for controller in robot.part_controllers.values(): controller.lite_physics = False
                    if variant == 'adapter': env._get_observations(force_update=True)
                    for action in reference_actions[:limit]: env.step(action)
                np.savez_compressed(directory/'actual_states.npz', states=np.asarray(actual))
                np.savez_compressed(directory/'recorded_reference.npz', states=states[:limit+1], actions=actions[:limit])
                with (directory/'steps.jsonl').open('w') as f:
                    for row in rows: f.write(json.dumps(row)+'\n')
                errors = [r['state_l2'] for r in rows if 'state_l2' in r]
                summary = dict(episode=ep, variant=variant, steps=len(actual), action_dtype=str(reference_actions.dtype), recorded_state_dtype=str(states.dtype),
                               success=any(r['success'] for r in rows), success_time=int(env.success_time),
                               first_step=rows[0], max_state_l2=max(errors), final_state_l2=errors[-1],
                               initial_full_state_max_error=run_info['initial_full_state_max_error'], seconds=time.monotonic()-start,
                               action_abs_max=np.abs(raw).max(0).tolist())
                official_path = args.output/f'episode_{ep:06d}'/'official'/'actual_states.npz'
                if variant != 'official' and official_path.exists():
                    base = np.load(official_path)['states']
                    if base.shape == np.asarray(actual).shape: summary['max_difference_from_official'] = float(np.max(np.abs(base-np.asarray(actual))))
                write_json(directory/'summary.json', summary)
                all_results.append(summary)
                write_json(args.output/'summary.json', all_results)
                print('REPLAY_RESULT',json.dumps({k:v for k,v in summary.items() if k!='first_step'}),flush=True)
            finally:
                env.close()
    print('DIAGNOSTIC_COMPLETE',flush=True)


if __name__ == '__main__': main()
