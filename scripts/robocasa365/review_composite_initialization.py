"""Independently summarize paired Composite initialization predictions.

Only the first eight commands are evaluated, matching the execution prefix.
Noise repetitions and chunk positions are never counted as independent demos.
This is an offline initialization diagnostic, not a policy success rate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(arrays, rows):
    predicted = np.clip(arrays['decoded'][rows, :8].astype(np.float64), -1, 1)
    target = arrays['target'][rows, :8].astype(np.float64)
    valid = arrays['valid'][rows, :8].astype(bool)
    result = {}
    for name, columns in [('base', slice(0, 4)), ('position', slice(5, 8)),
                          ('rotation', slice(8, 11)), ('arm', slice(5, 11))]:
        error = (predicted[..., columns] - target[..., columns])[valid[..., columns]]
        result[name] = dict(count=len(error), mae=float(np.abs(error).mean()),
                            rmse=float(np.sqrt(np.square(error).mean())))
    for name, column in [('mode', 4), ('gripper', 11)]:
        selected = valid[..., column]
        p, t = predicted[..., column][selected] >= .5, target[..., column][selected] >= .5
        tp, tn = int(np.sum(p & t)), int(np.sum(~p & ~t))
        fp, fn = int(np.sum(p & ~t)), int(np.sum(~p & t))
        result[name] = dict(count=len(t), true_positive=tp, true_negative=tn,
            false_positive=fp, false_negative=fn, accuracy=float(np.mean(p == t)),
            positive_recall=tp / (tp + fn) if tp + fn else None,
            negative_recall=tn / (tn + fp) if tn + fp else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    identity_path, report_path = args.audit / 'identity.json', args.audit / 'report.json'
    identity, original = (json.loads(p.read_text()) for p in (identity_path, report_path))
    assert original['status'] == 'complete' and original['identity_sha256'] == sha(identity_path)
    observations = identity['observations']
    assert len(observations) == 64 and identity['unique_episodes'] == 32
    tasks = sorted({x['task'] for x in observations})
    assert len(tasks) == 16 and identity['inference_steps'] == 4
    initializations = ['released_gr1', 'atomic18_step050000']
    arrays = {}
    for name in initializations:
        with np.load(args.audit / (name + '.npz'), allow_pickle=False) as saved:
            value = {key: saved[key] for key in saved.files}
        assert value['normalized'].shape == (128, 16, 32)
        assert value['decoded'].shape == value['target'].shape == value['valid'].shape == (128, 16, 12)
        assert value['valid'].dtype == bool
        assert all(np.isfinite(value[key]).all() for key in ['normalized', 'decoded', 'target'])
        np.testing.assert_array_equal(value['observation'], np.repeat(np.arange(64), 2))
        np.testing.assert_array_equal(value['noise_seed'], [seed + 1000 * i for i in range(64) for seed in (123, 124)])
        arrays[name] = value
    for key in ['target', 'valid', 'observation', 'noise_seed']:
        np.testing.assert_array_equal(arrays[initializations[0]][key], arrays[initializations[1]][key])
    aggregates, by_task = [], []
    for split in ['train', 'validation']:
        indices = [i for i, x in enumerate(observations) if x['split'] == split]
        assert len(indices) == 32 and len({(observations[i]['task'], observations[i]['episode']) for i in indices}) == 16
        for name in initializations:
            value = arrays[name]
            rows = np.isin(value['observation'], indices)
            aggregates.append(dict(split=split, initialization=name, distinct_episodes=16,
                distinct_windows=32, prediction_chunks=int(rows.sum()), metrics=summarize(value, rows)))
        for task in tasks:
            indices = [i for i, x in enumerate(observations) if x['split'] == split and x['task'] == task]
            assert len(indices) == 2 and {observations[i]['stage'] for i in indices} == {'initial', 'middle'}
            row = dict(task=task, split=split, distinct_episodes=1, distinct_windows=2, noise_repeats=2)
            for name in initializations:
                value = arrays[name]
                metrics = summarize(value, np.isin(value['observation'], indices))
                row[name] = metrics
                reference = next(x for x in original['records'] if x['initialization'] == name and x['task'] == task and x['split'] == split)
                for group in ['arm', 'position', 'rotation']:
                    for metric in ['mae', 'rmse']:
                        np.testing.assert_allclose(metrics[group][metric], reference['metrics'][group + '_command'][metric], rtol=1e-6, atol=1e-8)
                for group, reference_key in [('mode', 'base_mode'), ('gripper', 'gripper')]:
                    for count in ['count', 'true_positive', 'true_negative', 'false_positive', 'false_negative']:
                        assert metrics[group][count] == reference[reference_key][count]
            by_task.append(row)
    comparisons = {}
    for split in ['train', 'validation']:
        rows = [x for x in by_task if x['split'] == split]
        comparisons[split] = {group + '_mae_lower_tasks': sum(x[initializations[1]][group]['mae'] < x[initializations[0]][group]['mae'] for x in rows) for group in ['base', 'arm']}
        comparisons[split]['gripper_accuracy_higher_tasks'] = sum(x[initializations[1]]['gripper']['accuracy'] > x[initializations[0]]['gripper']['accuracy'] for x in rows)
    report = dict(status='complete',reviewed_utc=datetime.now(timezone.utc).isoformat(),
        aggregate_metrics=aggregates, per_task=by_task, paired_task_comparisons=comparisons,
        execution_prefix=8, distinct_episodes=32, distinct_windows=64, sampled_chunks=256,
        source_report_metrics_reproduced=True, optimizer_updates=0, new_policy_trials=0,
        source_sha256=sha(__file__), evidence_sha256={str(p):sha(p) for p in
            [identity_path, report_path] + [args.audit / (name + '.npz') for name in initializations]},
        limitations=['Only one train and one held-out demo per task; initial/middle windows only.',
            'CPU predictions with shared frozen features; no CUDA or closed-loop benefit established.',
            'Noise repetitions and chunk commands are correlated and are not new demonstrations.',
            'This comparison cannot replace the missing original16-demo step2000 video A/B.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:report[k] for k in ['status','aggregate_metrics','paired_task_comparisons']}))


if __name__ == '__main__':
    main()
