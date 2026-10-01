"""Run the author's single-env evaluation and persist episode-level evidence.

Uses the upstream policy adapter, camera preprocessing, action chunking and
720-step horizon. Single env avoids vector autoreset losing terminal success.
"""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from examples.Robocasa_tabletop.eval_files.simulation_env import (
    SimulationConfig, MultiStepConfig, VideoConfig, _create_single_env,
)
from examples.Robocasa_tabletop.eval_files.model2robocasa_interface import PolicyWarper


def atomic_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--task-index', type=int, required=True)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--max-steps', type=int, default=720)
    p.add_argument('--action-steps', type=int, default=12)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--port', type=int, default=16398)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--videos', type=int, default=1)
    a = p.parse_args(argv)
    tasks = json.loads((Path(__file__).parent / 'tasks.json').read_text())
    assert 0 <= a.task_index < len(tasks)
    assert a.episodes > 0 and 0 < a.action_steps <= 16 and a.max_steps > 0
    name = tasks[a.task_index]
    a.output.mkdir(parents=True, exist_ok=True)
    task_dir = a.output / f'{a.task_index:02d}'
    task_dir.mkdir(exist_ok=True)
    if (task_dir/'episodes.jsonl').exists():
        raise FileExistsError(f'Refusing to mix previous episodes in {task_dir}')
    cfg = SimulationConfig(env_name=name, multistep=MultiStepConfig(
        n_action_steps=a.action_steps, max_episode_steps=a.max_steps))
    model = None if a.preflight else PolicyWarper(
        policy_ckpt_path=os.environ['CKPT'], host='127.0.0.1', port=a.port,
        image_size=[224, 224], n_action_steps=a.action_steps)
    # No vector autoreset: capture terminal info before the next explicit reset.
    env = _create_single_env(cfg, 0, seed=a.seed)
    results = []
    started = time.monotonic()
    try:
        for ep in range(a.episodes):
            seed = a.seed + ep
            obs, _ = env.reset(seed=seed)
            if model:
                model.reset(None)
            frames = []
            success = False
            author_success = False
            calls = 0
            t0 = time.monotonic()
            while True:
                # Preserve the author's exact image alias and batch shape.
                batch = {k: np.expand_dims(v, 0) if not isinstance(v, str) else (v,)
                         for k, v in obs.items()}
                if 'video.ego_view_bg_crop_pad_res256_freq20' in batch:
                    batch['video.ego_view'] = batch.pop('video.ego_view_bg_crop_pad_res256_freq20')
                if ep < a.videos:
                    frames.append(batch['video.ego_view'][0, -1].copy())
                if a.preflight:
                    action = {f'action.{k}': np.repeat(obs[f'state.{k}'][-1:,:], a.action_steps, axis=0)
                              for k in ['left_arm','right_arm','left_hand','right_hand','waist']}
                else:
                    pred = model.step(batch)['actions']
                    action = {k: v[0] for k,v in pred.items()}
                    assert all(np.isfinite(v).all() and v.shape[0] == a.action_steps
                               for v in action.values()), 'Invalid policy action'
                obs, reward, done, truncated, info = env.step(action)
                calls += 1
                success |= bool(reward > 0)  # upstream wrapper aggregates all per-step rewards
                # Exact sampling used by original single-env vector evaluator;
                # its terminal info gets replaced by reset info in Gym 0.29.
                if not (done or truncated):
                    author_success |= bool(np.asarray(info['success'])[0])
                if a.preflight or done or truncated:
                    break
            if frames:
                import imageio.v2 as imageio
                imageio.mimsave(task_dir/f'episode_{ep:03d}.mp4', frames, fps=20/a.action_steps)
            row = dict(episode=ep, seed=seed, success=success,
                       author_success=author_success, action_calls=calls,
                       environment_steps=len(env.reward), seconds=time.monotonic()-t0)
            results.append(row)
            with (task_dir/'episodes.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            report = dict(task=name, task_index=a.task_index, episodes=len(results),
                          expected_episodes=a.episodes, successes=sum(r['success'] for r in results),
                          success_rate=sum(r['success'] for r in results)/len(results),
                          author_successes=sum(r['author_success'] for r in results),
                          max_episode_steps=a.max_steps, n_action_steps=a.action_steps,
                          preflight=a.preflight, complete=len(results)==a.episodes,
                          seconds=time.monotonic()-started, results=results)
            atomic_json(task_dir/'summary.json', report)
            print(json.dumps(dict(task_index=a.task_index, **row)), flush=True)
    finally:
        env.close()
        if model:
            model.client.close()

if __name__ == '__main__':
    main()
