"""Locate atomic-policy failure stages using saved actions and verified states.

Fresh reset must match the recorded scene, initial state, every saved query
state and final state. This is diagnostic playback, never a new policy trial.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import gymnasium as gym
import numpy as np
import robocasa
import robocasa.utils.object_utils as OU
import robosuite.utils.transform_utils as T

from scripts.robocasa365.eval_protocol import action_dict, observation_arrays, verify_controller_contract
from scripts.robocasa365.scene_identity import compare_scene_xml


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def longest_true(values):
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def measure(core, task, initial_object_height):
    if task == 'PickPlaceCounterToCabinet':
        obj = core.sim.data.body_xpos[core.obj_body_id['obj']].copy()
        eef = core.sim.data.site_xpos[core.robots[0].eef_site_id['right']].copy()
        return dict(target_grasp_predicate=bool(OU.check_obj_grasped(core, 'obj')),
                    object_inside_cabinet=bool(OU.obj_inside_of(core, 'obj', core.cab)),
                    gripper_far=bool(OU.gripper_obj_far(core)),
                    object_position=obj.tolist(), eef_position=eef.tolist(),
                    eef_object_center_distance_m=float(np.linalg.norm(obj - eef)),
                    object_lift_m=float(obj[2] - initial_object_height),
                    distractor_grasp_predicates={name: bool(OU.check_obj_grasped(core, name))
                                                for name in core.objects if name != 'obj'})
    body = core.sim.model.body_name2id('mobilebase0_base')
    position = core.sim.data.body_xpos[body].copy()
    orientation = T.mat2euler(core.sim.data.body_xmat[body].reshape(3, 3))
    return dict(base_position=position.tolist(), target_position=core.target_pos.tolist(),
                target_xy_distance_m=float(np.linalg.norm(core.target_pos[:2] - position[:2])),
                target_yaw_cosine=float(np.cos(core.target_ori[2] - orientation[2])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['PickPlaceCounterToCabinet', 'NavigateKitchen'], required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference = args.reference.resolve()
    result = json.loads((reference / 'result.json').read_text())
    evaluation = json.loads((reference.parent / 'evaluation.json').read_text())
    assert evaluation['task'] == args.task and evaluation['initialization'] == 'fresh_gym_target'
    assert result['demo_episode'] is None
    assert evaluation['horizon'] == evaluation['official_horizon']
    with np.load(reference / 'trajectory.npz', allow_pickle=False) as trace:
        arrays = {key: trace[key].copy() for key in trace.files}
    seed = result['seed']
    np.random.seed(seed)
    random.seed(seed)
    started = time.monotonic()
    report = dict(status='running', task=args.task, seed=seed, reference=str(reference),
                  original_success=result['success'], policy_trials_added=0,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  reference_trajectory_sha256=hashlib.sha256((reference / 'trajectory.npz').read_bytes()).hexdigest(),
                  scope='Replay of stored learned-policy commands, no new inference or GT actions. '
                        'Interpret predicates only when original recorded states match; images are not compared.')
    save(args.output / 'summary.json', report)
    env = None
    try:
        env = gym.make('robocasa/' + args.task, split='target', seed=seed,
                       camera_widths=256, camera_heights=256)
        obs, _ = env.reset(seed=seed)
        core = env.unwrapped.env
        verify_controller_contract(core)
        generated = core.sim.get_state().flatten().copy()
        np.savez_compressed(args.output / 'initial_states.npz', generated=generated, reference=arrays['initial_state'])
        xml = core.sim.model.get_xml()
        (args.output / 'generated_model.xml').write_text(xml)
        comparison = compare_scene_xml((reference / 'initial_model.xml').read_text(), xml)
        shape_equal = generated.shape == arrays['initial_state'].shape
        initial_error = float(np.max(np.abs(generated - arrays['initial_state']))) if shape_equal else None
        report.update(scene_comparison=comparison, initial_state_max_error=initial_error)
        save(args.output / 'summary.json', report)
        assert comparison['equivalent'], 'Fresh scene differs from the saved scene'
        assert shape_equal and initial_error < 1e-10, 'Fresh initial physical state differs'
        actions = arrays['actions_dataset_order']
        assert len(actions) == result['steps']
        query_lookup = {int(step): i for i, step in enumerate(arrays['query_steps'])}
        assert len(query_lookup) == len(arrays['query_steps'])
        rows, query_errors = [], []
        initial_height = (float(core.sim.data.body_xpos[core.obj_body_id['obj']][2])
                          if args.task == 'PickPlaceCounterToCabinet' else None)
        report['initial_predicates'] = measure(core, args.task, initial_height)
        action_index = 0
        original_step = core.step

        def observed_step(incoming):
            np.testing.assert_allclose(incoming, arrays['actions_actually_received_by_simulator'][action_index],
                                       rtol=0, atol=2e-7)
            return original_step(incoming)

        core.step = observed_step
        for step, action in enumerate(actions):
            action_index = step
            if step in query_lookup:
                error = float(np.max(np.abs(observation_arrays(obs)['state'] - arrays['query_states'][query_lookup[step]])))
                query_errors.append(dict(step=step, state_max_error=error))
            obs, _, terminated, truncated, info = env.step(action_dict(action))
            rows.append(dict(step=step + 1, success=bool(info['success']),
                             **measure(core, args.task, initial_height)))
            if (step + 1) % 200 == 0:
                print('DIAGNOSTIC_REPLAY', args.task, seed, step + 1, len(actions), flush=True)
            if terminated or truncated:
                break
        final_state = core.sim.get_state().flatten().copy()
        np.savez_compressed(args.output / 'final_states.npz', generated=final_state, reference=arrays['final_state'])
        final_error = float(np.max(np.abs(final_state - arrays['final_state'])))
        query_error = max(x['state_max_error'] for x in query_errors)
        save(args.output / 'predicates.json', rows)
        save(args.output / 'query_errors.json', query_errors)
        match = (len(rows) == len(actions) and len(query_errors) == len(query_lookup)
                 and final_error < 1e-8 and query_error < 1e-6
                 and bool(any(x['success'] for x in rows)) == result['success'])
        report.update(steps=len(rows), query_states_compared=len(query_errors), final_state_max_error=final_error,
                      max_query_state_error=query_error, trajectory_matches_reference=bool(match),
                      first_success_step=next((x['step'] for x in rows if x['success']), None))
        if match and args.task == 'PickPlaceCounterToCabinet':
            grasp = [x['target_grasp_predicate'] for x in rows]
            inside = [x['object_inside_cabinet'] for x in rows]
            report['failure_evidence'] = dict(target_grasp_steps=sum(grasp), longest_grasp_steps=longest_true(grasp),
                first_target_grasp_step=next((x['step'] for x in rows if x['target_grasp_predicate']), None),
                object_inside_cabinet_steps=sum(inside), max_object_lift_m=max(x['object_lift_m'] for x in rows),
                min_eef_object_center_distance_m=min(x['eef_object_center_distance_m'] for x in rows),
                failure_stage='no_target_grasp_observed' if not any(grasp) else
                              'grasp_observed_no_cabinet_entry' if not any(inside) else
                              'cabinet_entry_observed_inspect_release_or_retention')
        elif match:
            distances = [x['target_xy_distance_m'] for x in rows]
            report['failure_evidence'] = dict(min_target_xy_distance_m=min(distances),
                final_target_xy_distance_m=distances[-1],
                base_mode_steps=int(np.sum(actions[:, 4] >= .5)),
                failure_stage='never_within_position_threshold' if min(distances) > .20 else
                              'position_threshold_reached_inspect_orientation')
        report.update(status='complete' if match else 'state_mismatch', seconds=time.monotonic() - started,
                      verified_utc=datetime.now(timezone.utc).isoformat())
        save(args.output / 'summary.json', report)
        assert match, 'Replayed trajectory differs; no causal interpretation permitted'
        print(json.dumps(report), flush=True)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', seconds=time.monotonic() - started)
        save(args.output / 'summary.json', report)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
