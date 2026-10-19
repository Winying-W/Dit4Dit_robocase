"""GT action replay with explicit restoration of the collection startup sequence."""
import argparse
import copy
from contextlib import nullcontext
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import robosuite
import robocasa.utils.lerobot_utils as LU
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from scripts.robocasa365.replay_utils import apply_collection_startup

CAMERAS = ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand']
STATE_KEYS = ['robot0_base_pos', 'robot0_base_quat', 'robot0_base_to_eef_pos',
              'robot0_base_to_eef_quat', 'robot0_gripper_qpos']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, nargs='+', default=[0, 3, 5])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-video', action='store_true')
    p.add_argument('--startup', choices=['auto', 'none'], default='auto',
                   help='Restore omitted collection zero-step from simulation timestamps; none reproduces old baseline')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    meta = LU.get_env_metadata(args.dataset)
    report = dict(task=meta['env_name'], episodes=[], startup_mode=args.startup, seed=args.seed,
                  method='Fresh environment per episode; initial XML/state restore, collection startup, then GT actions only',
                  controller_configs=meta['env_kwargs']['controller_configs'])
    for ep in args.episodes:
        states = LU.get_episode_states(args.dataset, ep)
        actions = LU.get_episode_actions(args.dataset, ep)
        df = pd.read_parquet(next((args.dataset/'data').glob(f'*/episode_{ep:06d}.parquet')))
        raw = np.stack(df.action)
        recorded = np.stack(df['observation.state'])
        assert np.all(raw[:, :4] == 0), f'episode {ep} is mobile'
        assert np.array_equal(actions, raw[:, [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]])
        assert len(states) == len(actions)
        kw = copy.deepcopy(meta['env_kwargs'])
        kw.update(env_name=meta['env_name'], has_renderer=False,
                  has_offscreen_renderer=not args.no_video, use_camera_obs=False, seed=args.seed)
        env = robosuite.make(**kw)
        try:
            reset_to(env, dict(states=states[0], model=LU.get_episode_model_xml(args.dataset, ep),
                               ep_meta=json.dumps(LU.get_episode_meta(args.dataset, ep))))
            initial_full_state_error = float(np.max(np.abs(env.sim.get_state().flatten()-states[0])))
            obs = env._get_observations(force_update=True)
            rebuilt = np.concatenate([obs[k] for k in STATE_KEYS])
            initial_state_error = float(np.max(np.abs(rebuilt-recorded[0])))
            initial_images = {}
            if not args.no_video:
                for cam in CAMERAS:
                    cap = cv2.VideoCapture(str(next((args.dataset/'videos').glob(f'*/observation.images.{cam}/episode_{ep:06d}.mp4'))))
                    ok, bgr = cap.read()
                    cap.release()
                    assert ok
                    gt = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rendered = env.sim.render(height=gt.shape[0], width=gt.shape[1], camera_name=cam)[::-1]
                    initial_images[cam] = dict(mae=float(np.abs(rendered.astype(float)-gt).mean()),
                                               flipped_mae=float(np.abs(rendered[::-1].astype(float)-gt).mean()))
                    imageio.imwrite(args.output/f'replay_{ep:06d}_{cam}_comparison.png', np.concatenate([gt, rendered], axis=1))
            startup = apply_collection_startup(env, states, mode=args.startup)
            errors = []
            success = False
            context = nullcontext(None) if args.no_video else imageio.get_writer(str(args.output/f'replay_{ep:06d}.mp4'), fps=10)
            with context as writer:
                for t, action in enumerate(actions):
                    env.step(action)
                    success = success or bool(env._check_success())
                    if t+1 < len(states):
                        diff = env.sim.get_state().flatten()-states[t+1]
                        assert np.isfinite(diff).all()
                        errors.append(dict(step=t+1, state_l2=float(np.linalg.norm(diff)), time_abs=float(abs(diff[0])),
                                           qpos_linf=float(np.max(np.abs(diff[1:1+env.sim.model.nq]))),
                                           qvel_linf=float(np.max(np.abs(diff[1+env.sim.model.nq:])))))
                    if writer is not None and t % 2 == 0:
                        writer.append_data(np.concatenate([env.sim.render(height=256, width=256, camera_name=c)[::-1] for c in CAMERAS], axis=1))
                    if (t+1) % 200 == 0:
                        print(f'episode={ep} step={t+1}/{len(actions)} success={success}', flush=True)
            item = dict(episode=ep, steps=len(actions), initial_state_max_error=initial_state_error,
                        initial_full_state_max_error=initial_full_state_error, initial_images=initial_images, startup=startup,
                        time_error_max=max(x['time_abs'] for x in errors),
                        qpos_error_max=max(x['qpos_linf'] for x in errors), qvel_error_max=max(x['qvel_linf'] for x in errors),
                        sim_state_l2_max=max(x['state_l2'] for x in errors), sim_state_l2_final=errors[-1]['state_l2'],
                        success=success, success_time=getattr(env, 'success_time', None),
                        base_action_abs_max=np.abs(raw[:, :4]).max(0).tolist(), control_mode_values=np.unique(raw[:, 4]).tolist())
            (args.output/f'errors_{ep:06d}.json').write_text(json.dumps(errors)+'\n')
            report['episodes'].append(item)
            report['timing_passed'] = all(x['time_error_max'] < 1e-8 for x in report['episodes'])
            report['passed'] = report['timing_passed'] and all(x['initial_state_max_error'] < 1e-4 and x['success'] for x in report['episodes'])
            (args.output/'replay.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(item), flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
