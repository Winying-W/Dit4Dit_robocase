"""Audit phase pools against raw training labels and the real dataset adapter.

This is a CPU data preflight, not a model-update or closed-loop acceptance.
The reference switch implementation expands each transition backwards instead
of reusing the builder's cumulative-window implementation.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.robocasa365.phase_sampling import PhaseBalancedTasks, sha


def audit(manifest_path, pools_path, output, draws=16000):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from scripts.robocasa365.multitask_data import prepare_multitask_dataset

    if torch.cuda.is_initialized():
        raise RuntimeError('Data audit must not initialize CUDA')
    pa.set_cpu_count(2)
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads(pools_path.read_text())
    assert metadata['initial_frames'] == 16 and metadata['execution_prefix'] == 8
    assert metadata['manifest_sha256'] == sha(manifest_path)
    pool_file = pools_path.parent / metadata['pools_file']
    assert metadata['pools_sha256'] == sha(pool_file)
    summaries = []
    with np.load(pool_file, allow_pickle=False) as pools:
        for row, recorded in zip(manifest['tasks'], metadata['tasks']):
            name = row['task']
            assert name == recorded['task']
            root = Path(row['path'])
            info = json.loads((root / 'meta/info.json').read_text())
            episode_path = root / 'meta/episodes.jsonl'
            assert sha(episode_path) == recorded['episode_metadata_sha256']
            episodes = [json.loads(line) for line in episode_path.read_text().splitlines() if line]
            records = {entry['episode']: entry for entry in row['episodes']}
            train, validation = set(row['train_episodes']), set(row['validation_episodes'])
            assert train.isdisjoint(validation)
            assert train | validation == set(records)
            assert {entry['episode_index'] for entry in episodes} == set(records)
            offsets, total = {}, 0
            expected_ranges = []
            for episode in episodes:
                ep, n = int(episode['episode_index']), int(episode['length'])
                assert records[ep]['frames'] == n
                offsets[ep] = total
                expected_ranges.append(dict(episode=ep, start=total, end=total+n,
                                            split='train' if ep in train else 'validation'))
                total += n
            assert expected_ranges == recorded['episode_ranges']
            initial = np.zeros(total, dtype=bool)
            switching = np.zeros(total, dtype=bool)
            training = np.zeros(total, dtype=bool)
            modes, grippers = 0, 0
            for ep in sorted(train):
                entry = records[ep]
                assert entry['split'] == 'train'
                n, offset = entry['frames'], offsets[ep]
                path = root / info['data_path'].format(
                    episode_chunk=ep // info['chunks_size'], episode_index=ep)
                assert sha(path) == entry['parquet_sha256']
                table = pq.read_table(path, columns=['action'])
                action = np.asarray(table['action'].to_pylist())
                assert action.shape == (n, 12) and np.isfinite(action).all()
                assert np.isin(action[:, [4, 11]], [-1, 1]).all()
                training[offset:offset+n] = True
                initial[offset:offset+min(16, n)] = True
                for column in [4, 11]:
                    transitions = np.nonzero(action[1:, column] != action[:-1, column])[0] + 1
                    if column == 4:
                        modes += len(transitions)
                    else:
                        grippers += len(transitions)
                    for transition in transitions:
                        switching[offset+max(0, int(transition)-7):offset+int(transition)] = True
            for ep in validation:
                assert records[ep]['split'] == 'validation'
            np.testing.assert_array_equal(pools['initial__'+name], np.flatnonzero(initial))
            np.testing.assert_array_equal(pools['switch__'+name], np.flatnonzero(switching))
            assert not (initial & ~training).any() and not (switching & ~training).any()
            assert hashlib.sha256(np.flatnonzero(training).astype(np.int64).tobytes()).hexdigest() == recorded['training_indices_sha256']
            summaries.append(dict(task=name, training_episodes=len(train), validation_episodes=len(validation),
                training_frames=int(training.sum()), initial_frames=int(initial.sum()),
                switch_frames=int(switching.sum()), mode_transitions=modes, gripper_transitions=grippers))
            print('RAW_PHASE_AUDIT', name, 'passed', flush=True)
    assert len(summaries) == 16 and sum(row['training_episodes'] for row in summaries) == 7271

    prepared = output / 'prepared'
    prepared.mkdir()
    dataset, _, _, split = prepare_multitask_dataset(manifest_path, prepared)
    left = PhaseBalancedTasks(dataset, pools_path, manifest_path, 'paired_uniform')
    right = PhaseBalancedTasks(dataset, pools_path, manifest_path, 'initial_switch')
    a, b, reference = [np.random.default_rng(20260928) for _ in range(3)]
    choices = dict(ordinary=0, initial=0, switch=0)
    fallback, tasks = 0, np.zeros(16, dtype=np.int64)
    left_rows, right_rows = [], []
    for _ in range(draws):
        l, r = left.sample_index(a), right.sample_index(b)
        task = int(reference.integers(16))
        child = np.random.default_rng(int(reference.integers(np.iinfo(np.int64).max)))
        p = child.random()
        kind = 'ordinary' if p < .5 else 'initial' if p < .75 else 'switch'
        pool = right.indices[task] if kind == 'ordinary' else right.pools[task][kind]
        if not len(pool):
            fallback += 1
            pool = right.indices[task]
        expected = [task, int(pool[int(child.integers(len(pool)))])]
        assert r == expected and l[0] == r[0] == task
        choices[kind] += 1
        tasks[task] += 1
        left_rows.append(l)
        right_rows.append(r)
    assert a.bit_generator.state == b.bit_generator.state == reference.bit_generator.state
    for sampler, rng in [(left, a), (right, b)]:
        state = copy.deepcopy(rng.bit_generator.state)
        expected = [sampler.sample_index(rng) for _ in range(256)]
        resumed = np.random.default_rng(0)
        resumed.bit_generator.state = state
        assert expected == [sampler.sample_index(resumed) for _ in range(256)]
        assert rng.bit_generator.state == resumed.bit_generator.state
    assert a.bit_generator.state == b.bit_generator.state
    np.savez_compressed(output / 'paired_draws.npz', uniform=np.asarray(left_rows),
                        initial_switch=np.asarray(right_rows))
    print('REAL_ADAPTER_PAIRED_SAMPLING passed', flush=True)

    samples = []
    for task, adapter in enumerate(dataset.datasets):
        row = manifest['tasks'][task]
        candidates = [('initial', right.pools[task]['initial'][0]),
                      ('switch', right.pools[task]['switch'][len(right.pools[task]['switch'])//2]),
                      ('episode_tail', right.indices[task][-1])]
        for kind, local_index in candidates:
            ep, step = map(int, adapter.dataset.all_steps[int(local_index)])
            assert ep in row['train_episodes'] and ep not in row['validation_episodes']
            batch = right[[task, int(local_index)]]
            root = Path(row['path'])
            info = json.loads((root / 'meta/info.json').read_text())
            path = root / info['data_path'].format(episode_chunk=ep//info['chunks_size'], episode_index=ep)
            raw = pq.read_table(path, columns=['action', 'observation.state'])
            actions = np.asarray(raw['action'].to_pylist(), dtype=np.float32)
            states = np.asarray(raw['observation.state'].to_pylist(), dtype=np.float32)
            valid = min(16, len(actions)-step)
            assert batch['state'].shape == (1, 64) and batch['action'].shape == (16, 32)
            assert batch['state_mask'].shape == (1, 64) and batch['state_mask'].dtype == np.bool_
            assert batch['action_mask'].shape == (16, 32) and batch['action_mask'].dtype == np.bool_
            assert batch['state_mask'][:, :16].all() and not batch['state_mask'][:, 16:].any()
            expected_mask = np.zeros((16, 32), dtype=bool)
            expected_mask[:valid, :12] = True
            np.testing.assert_array_equal(batch['action_mask'], expected_mask)
            np.testing.assert_array_equal(batch['state'][:, :16], states[step:step+1])
            assert np.isfinite(batch['state']).all() and np.isfinite(batch['action']).all()
            assert (batch['state'][:, 16:] == 0).all() and (batch['action'][:, 12:] == 0).all()
            decoded = adapter.decode_actions(batch['action'][:valid], env_order=False)
            np.testing.assert_allclose(decoded, actions[step:step+valid], rtol=1e-5, atol=2e-6)
            np.testing.assert_array_equal(batch['action'][:valid, [4, 11]], actions[step:step+valid, [4, 11]])
            assert isinstance(batch['lang'], str) and batch['lang'].strip()
            assert len(batch['image']) == 9
            for image in batch['image']:
                assert tuple(image.shape) == (3, 128, 384)
                assert image.dtype == torch.float32 and torch.isfinite(image).all()
                assert float(image.min()) >= 0 and float(image.max()) <= 1
            samples.append(dict(task=row['task'], kind=kind, index=int(local_index), episode=ep, frame=step,
                valid_action_frames=valid, image_shape=[9, 3, 128, 384], state_shape=[1, 64],
                action_shape=[16, 32], action_mask_shape=[16, 32], roundtrip_max_error=float(np.max(np.abs(decoded-actions[step:step+valid])))))
        print('REAL_SAMPLES', row['task'], 'passed', flush=True)
    assert not torch.cuda.is_initialized()
    result = dict(status='passed', completed_utc=datetime.now(timezone.utc).isoformat(),
        verifier_sha256=sha(Path(__file__)), manifest_sha256=sha(manifest_path),
        phase_pool_manifest_sha256=sha(pools_path), phase_pool_sha256=sha(pool_file),
        sampler_source_sha256=metadata['source_sha256'], tasks=summaries,
        training_parquets_independently_checked=7271, validation_parquets_used=0,
        training_frames=split['train_frames'], paired_draws=draws,
        category_choices=choices, category_fractions={k:v/draws for k,v in choices.items()},
        fallback_draws=fallback, task_counts=tasks.tolist(),
        task_sequence_and_parent_rng_equal=True, exact_sampler_resume=True,
        paired_draws_sha256=sha(output / 'paired_draws.npz'), real_samples=samples,
        real_model_forward=False, real_model_backward=False, cuda_checked=False,
        new_optimizer_updates=0, new_policy_trials=0,
        scope='Full16 training-only pool reconstruction and live adapter sampling. Commands are dataset labels, not physical contact annotations. Model updates, CUDA memory and checkpoint resume remain separate prequeue checks.')
    (output / 'acceptance.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--pools', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.manifest.resolve(), args.pools.resolve(), args.output.resolve())
    print(json.dumps({key:result[key] for key in ['status', 'training_parquets_independently_checked',
        'paired_draws', 'category_choices', 'fallback_draws']}), flush=True)
