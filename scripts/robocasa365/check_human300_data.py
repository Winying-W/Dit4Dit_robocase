"""Audit selected episodes and real RGB/action samples before human300 training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.robocasa365.multitask_data import prepare_multitask_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assert manifest['episode_selection']['filter_key'] == '100_demos'
    assert manifest['episode_selection']['filter_key_seed'] == 0
    if not args.preflight:
        assert manifest['task_set'] == 'pretrain_human300' and len(manifest['tasks']) == 300
    args.output.mkdir(parents=True, exist_ok=True)
    dataset, _, validation, split = prepare_multitask_dataset(args.manifest, args.output)
    reports = []
    for task_id, (adapter, indices, row) in enumerate(zip(dataset.datasets, dataset.indices, manifest['tasks'])):
        train = set(row['train_episodes'])
        heldout = set(row['validation_episodes'])
        excluded = set(row['excluded_episodes'])
        assert len(train) == 95 and len(heldout) == 5
        observed = {int(adapter.dataset.all_steps[i][0]) for i in indices}
        assert observed == train and not observed.intersection(heldout | excluded)
        assert len(indices) == sum(row['episode_lengths'][str(ep)] for ep in train)
        samples = []
        for index in [int(indices[0]), int(indices[len(indices)//2]), int(indices[-1])]:
            sample = adapter[index]
            ep, timestep = map(int, adapter.dataset.all_steps[index])
            path = next((Path(row['path'])/'data').glob(f'*/episode_{ep:06d}.parquet'))
            raw = np.stack(pd.read_parquet(path, columns=['action']).action)[timestep:timestep+16]
            decoded = adapter.decode_actions(sample['action'], env_order=False)[:len(raw)]
            error = float(np.abs(decoded-raw).max())
            assert error < 1e-6
            assert len(sample['image']) == 9
            assert all(tuple(image.shape) == (3, 224, 672) for image in sample['image'])
            assert sample['state'].shape == (1, 64) and sample['action'].shape == (16, 32)
            assert np.isfinite(sample['state']).all() and np.isfinite(sample['action']).all()
            assert int(sample['action_mask'].sum()) == len(raw)*12
            assert not sample['action_mask'][:, 12:].any()
            assert isinstance(sample['lang'], str) and sample['lang']
            samples.append(dict(index=index, episode=ep, timestep=timestep,
                                valid_action_frames=len(raw), roundtrip_max_error=error))
        reports.append(dict(task=row['task'], train_frames=len(indices), excluded_episodes=len(excluded), samples=samples))
        (args.output/'progress.json').write_text(json.dumps(dict(complete=False, checked=len(reports), total=len(manifest['tasks']))))
        print('HUMAN300_TASK_VERIFIED', task_id+1, row['task'], flush=True)
    assert sum(row['train_frames'] for row in reports) == manifest['train_frames']
    report = dict(complete=True, diagnostic=args.preflight,
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  train_episodes=split['train_episodes'], validation_episodes=split['validation_episodes'],
                  validation_windows=len(validation), tasks=reports,
                  scope='Exact training-index coverage and three real RGB/action windows per task; no model/GPU or closed-loop acceptance')
    (args.output/'acceptance.json').write_text(json.dumps(report, indent=2)+'\n')
    print('HUMAN300_DATA_ACCEPTED', len(reports), flush=True)


if __name__ == '__main__':
    main()
