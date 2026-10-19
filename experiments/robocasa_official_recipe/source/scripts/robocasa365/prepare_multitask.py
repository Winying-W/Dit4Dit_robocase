"""Audit target task rows and freeze episode splits and training-only statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def deterministic_split(episodes, task, seed, validation_fraction=.1):
    salt = int.from_bytes(hashlib.sha256(task.encode()).digest()[:4], 'little')
    order = np.random.default_rng(np.random.SeedSequence([seed, salt])).permutation(sorted(episodes)).tolist()
    nval = max(2, int(round(len(order) * validation_fraction)))
    if nval >= len(order):
        raise ValueError('Insufficient episodes for disjoint train/validation sets')
    return sorted(order[nval:]), sorted(order[:nval])


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    from robocasa.utils.dataset_registry import TARGET_TASKS
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20260924)
    p.add_argument('--task-set', choices=['composite_seen','atomic_seen'], default='composite_seen')
    p.add_argument('--tasks', nargs='+', help='Explicitly declared subset; omitted means the full official task set')
    p.add_argument('--final-trials', type=int, default=100)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output/'manifest.json').exists():
        raise FileExistsError('Immutable manifest exists; use a new output directory')
    source = json.loads(a.datasets.read_text())
    assert source['status'] == 'complete'
    selected = a.tasks or TARGET_TASKS[a.task_set]
    assert len(selected) == len(set(selected)) and set(selected).issubset(TARGET_TASKS[a.task_set])
    assert source['task_set'] == a.task_set
    assert {r['task'] for r in source['tasks']} == set(selected)
    if a.final_trials < 20:
        raise ValueError('Final evaluation needs at least 20 trials per task')
    rows = []
    train_arrays = {'action': [], 'observation.state': []}
    expected = dict(base_motion=(0,4), control_mode=(4,5), end_effector_position=(5,8),
                    end_effector_rotation=(8,11), gripper_close=(11,12))
    for item in sorted(source['tasks'], key=lambda x:x['task']):
        task, path = item['task'], Path(item['path'])
        info = json.loads((path/'meta/info.json').read_text())
        modality = json.loads((path/'meta/modality.json').read_text())
        for key, bounds in expected.items():
            desc = modality['action'][key]
            assert (desc['start'], desc['end']) == bounds, (task, key, desc)
        meta = json.loads((path/'extras/dataset_meta.json').read_text())['env_args']
        arm = meta['env_kwargs']['controller_configs']['body_parts']['right']
        assert arm['type'] == 'OSC_POSE' and arm['input_type'] == 'delta' and arm['input_ref_frame'] == 'base'
        np.testing.assert_allclose(arm['output_max'], [.05,.05,.05,.5,.5,.5], rtol=0, atol=0)
        assert info['fps'] == 20, (task, info['fps'])
        files = {int(f.stem.split('_')[-1]):f for f in sorted((path/'data').glob('*/episode_*.parquet'))}
        assert len(files) == info['total_episodes']
        train, val = deterministic_split(files, task, a.seed)
        train_set = set(train)
        per_episode = []
        for ep, file in files.items():
            df = pd.read_parquet(file, columns=['action', 'observation.state'])
            arrays = {key:np.stack(df[key]).astype(np.float32) for key in train_arrays}
            action, state = arrays['action'], arrays['observation.state']
            assert action.shape == (len(df),12) and state.shape == (len(df),16), (task,ep)
            assert np.isfinite(action).all() and np.isfinite(state).all(), (task,ep)
            assert np.max(np.abs(action)) <= 1 + 1e-6, (task,ep)
            assert set(np.unique(action[:,4])).issubset({-1.,1.}), (task,ep,'control mode')
            assert set(np.unique(action[:,11])).issubset({-1.,1.}), (task,ep,'gripper')
            for cam in ('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand'):
                video_key = 'observation.images.' + cam
                video = path / info['video_path'].format(episode_chunk=ep//info['chunks_size'], video_key=video_key, episode_index=ep)
                assert video.is_file() and video.stat().st_size > 0, video
            extras = path/'extras'/f'episode_{ep:06d}'
            assert all((extras/name).is_file() for name in ['states.npz','model.xml.gz','ep_meta.json']), extras
            if ep in train_set:
                for key in arrays:
                    train_arrays[key].append(arrays[key])
            per_episode.append(dict(episode=ep, frames=len(df), split='train' if ep in train_set else 'validation',
                nonzero_base_frames=int(np.any(action[:,:4] != 0, axis=1).sum()),
                base_mode_frames=int((action[:,4] == 1).sum()),
                parquet_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        row = dict(task=task, path=str(path.resolve()), horizon=item['horizon'],
                   train_episodes=train, validation_episodes=val, episodes=per_episode,
                   controller_input='delta', controller_reference='base', fps=info['fps'])
        rows.append(row)
        save(a.output/'audit_progress.json', dict(tasks=rows))
        print('MULTITASK_AUDITED', task, len(train), len(val), flush=True)
    statistics = {}
    for key in train_arrays:
        x = np.concatenate(train_arrays[key]); train_arrays[key].clear()
        statistics[key] = {name:fn(x, axis=0).tolist() for name,fn in
            [('min',np.min),('max',np.max),('mean',np.mean),('std',np.std)]}
        statistics[key].update(q01=np.quantile(x,.01,axis=0).tolist(), q99=np.quantile(x,.99,axis=0).tolist())
        del x
    save(a.output/'normalization.json', statistics)
    save(a.output/'manifest.json', dict(format_version=1, task_set=a.task_set, split='target', seed=a.seed,
        task_scope='full_official_task_set' if set(selected)==set(TARGET_TASKS[a.task_set]) else 'declared_task_subset',
        tasks=rows, train_episodes=sum(len(r['train_episodes']) for r in rows),
        validation_episodes=sum(len(r['validation_episodes']) for r in rows),
        normalization_sha256=hashlib.sha256((a.output/'normalization.json').read_bytes()).hexdigest(),
        statistics_source='All and only training episodes; shared continuous-action min/max across tasks; raw binary mode/gripper; unnormalized state',
        evaluation=dict(initialization='fresh_gym_target', seeds=list(range(1000,1000+a.final_trials)),
                        trials_per_task=a.final_trials, planned_trials=len(rows)*a.final_trials, selection_seeds=list(range(100,120)))))


if __name__ == '__main__':
    main()
