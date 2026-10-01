"""CPU prediction-fit diagnostic on fixed train/validation observations.

The frozen backbone is evaluated once per observation and reused across action
checkpoints and integration-step counts. This is not a closed-loop benchmark.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def binary_metrics(prediction, target, valid):
    prediction, target, valid = (np.asarray(x) for x in (prediction, target, valid))
    assert prediction.shape == target.shape == valid.shape and valid.dtype == bool
    assert np.isfinite(prediction).all() and np.isfinite(target).all()
    p, t = prediction[valid] >= .5, target[valid] >= .5
    tp, tn = int(np.sum(p & t)), int(np.sum(~p & ~t))
    fp, fn = int(np.sum(p & ~t)), int(np.sum(~p & t))
    recall_pos = tp / (tp + fn) if tp + fn else None
    recall_neg = tn / (tn + fp) if tn + fp else None
    recalls = [x for x in [recall_pos, recall_neg] if x is not None]
    return dict(count=len(t), true_positive=tp, true_negative=tn, false_positive=fp, false_negative=fn,
                accuracy=(tp + tn) / len(t) if len(t) else None,
                positive_recall=recall_pos, negative_recall=recall_neg,
                balanced_accuracy=float(np.mean(recalls)) if len(recalls) == 2 else None,
                target_positive_fraction=float(np.mean(t)) if len(t) else None,
                predicted_positive_fraction=float(np.mean(p)) if len(t) else None)


def transition_windows(actions, task):
    """Select command transitions, not physical grasp/release or task success."""
    actions = np.asarray(actions)
    assert actions.ndim == 2 and actions.shape[1] == 12 and np.isfinite(actions).all()
    if task not in ['NavigateKitchen', 'PickPlaceCounterToCabinet']:
        raise ValueError('Transition selection is defined only for navigation/cabinet')
    windows = [dict(stage='initial', step=0, transition_step=None)]
    index = 4 if task == 'NavigateKitchen' else 11
    active = actions[:, index] >= .5
    candidates = []
    for step in range(4, len(actions) - 3):
        before, after = active[step - 4:step], active[step:step + 4]
        if task == 'NavigateKitchen':
            found = not before.any() and after.all()
        elif task == 'PickPlaceCounterToCabinet':
            found = before.all() and not after.any()
        else:
            raise ValueError('Transition selection is defined only for navigation/cabinet')
        if found:
            candidates.append(step)
    if candidates:
        transition = candidates[0] if task == 'NavigateKitchen' else candidates[-1]
        # Four commands precede the transition, so the executed first eight
        # commands contain both labels. End padding remains masked by adapter.
        windows.append(dict(stage='base_switch_command' if task == 'NavigateKitchen' else
                            'last_gripper_open_command', step=transition - 4,
                            transition_step=transition))
    return windows


def main():
    import torch
    from omegaconf import OmegaConf
    from DiT4DiT.model.framework.base_framework import baseframework
    from DiT4DiT.dataloader.robocasa365_datasets import Robocasa365DatasetAdapter
    from scripts.robocasa365.multitask_data import apply_statistics
    from scripts.robocasa365.prediction_metrics import action_metrics

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--tasks', nargs='+', default=['NavigateKitchen', 'PickPlaceCounterToCabinet'])
    parser.add_argument('--episodes-per-split', type=int, default=2)
    parser.add_argument('--selection-seed', type=int, default=77)
    parser.add_argument('--noise-seeds', type=int, nargs='+', default=[123, 124])
    parser.add_argument('--inference-steps', type=int, nargs='+', default=[4, 16])
    parser.add_argument('--window-mode', choices=['fractions', 'transitions'], default='fractions')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_available(), 'CPU diagnostic must not occupy evaluation GPUs'
    assert args.episodes_per_split > 0 and min(args.inference_steps) > 0
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    started = time.monotonic()
    manifest = json.loads(args.manifest.read_text())
    payloads = [torch.load(path, map_location='cpu', weights_only=False, mmap=True) for path in args.checkpoints]
    assert all(p['phase'] == 'action' and p['base_checkpoint'] == payloads[0]['base_checkpoint'] for p in payloads)
    assert all(p['normalization_sha256'] == payloads[0]['normalization_sha256'] and p['split'] == payloads[0]['split'] for p in payloads)
    assert payloads[0]['split']['manifest_sha256'] == sha(args.manifest)
    assert all(len(p['trained_state']) == 247 and all(k.startswith('action_model.') for k in p['trained_state']) for p in payloads)
    directory = args.checkpoints[0].parent
    assert sha(directory / 'normalization.json') == payloads[0]['normalization_sha256']
    stats = json.loads((directory / 'normalization.json').read_text())
    cfg = OmegaConf.load(directory / 'data_config.yaml')
    datasets, examples, specs, targets, masks, selections = {}, [], [], [], [], []
    rng = np.random.default_rng(args.selection_seed)
    for task in args.tasks:
        row = next(x for x in manifest['tasks'] if x['task'] == task)
        local_cfg = OmegaConf.merge(cfg, dict(dataset_path=row['path']))
        ds = Robocasa365DatasetAdapter(row['path'], local_cfg)
        apply_statistics(ds, row['path'], stats)
        datasets[task] = ds
        by_episode, step_lookup = {}, {}
        for index, (episode, step) in enumerate(ds.dataset.all_steps):
            step_lookup[int(episode), int(step)] = index
            length = ds.dataset.trajectory_lengths[ds.dataset.get_trajectory_index(episode)]
            if int(step) + 16 <= length:
                by_episode.setdefault(int(episode), []).append(index)
        assert set(row['train_episodes']).isdisjoint(row['validation_episodes'])
        for split in ['train', 'validation']:
            selected = sorted(int(x) for x in rng.choice(row[split + '_episodes'], size=args.episodes_per_split, replace=False))
            for episode in selected:
                indices = by_episode[episode]
                if args.window_mode == 'fractions':
                    windows = [dict(stage=f'fraction_{fraction}', fraction=fraction,
                        step=int(ds.dataset.all_steps[indices[min(int(len(indices) * fraction), len(indices) - 1)]][1]),
                        transition_step=None) for fraction in [.1, .5]]
                else:
                    import pyarrow.parquet as pq
                    data_root = Path(row['path'])
                    info = json.loads((data_root / 'meta/info.json').read_text())
                    path = data_root / info['data_path'].format(episode_chunk=episode // info['chunks_size'],
                                                              episode_index=episode)
                    actions = np.asarray(pq.read_table(path, columns=['action'])['action'].to_pylist())
                    windows = transition_windows(actions, task)
                selections.append(dict(task=task, split=split, episode=episode, windows=windows,
                    transition_found=any(x['transition_step'] is not None for x in windows)
                    if args.window_mode == 'transitions' else None))
                for window in windows:
                    index = step_lookup[episode, window['step']]
                    ep, step = ds.dataset.all_steps[index]
                    raw = ds.dataset.get_step_data(ep, step)
                    example = ds[index]
                    example['image'] = example['image'][:1]
                    target = np.concatenate([np.asarray(raw[key]).copy() for key in ds.robot.action_keys], axis=-1)
                    np.testing.assert_allclose(ds.decode_actions(example['action'], env_order=False), target, atol=1e-6, rtol=1e-5)
                    examples.append(example)
                    targets.append(target)
                    masks.append(example['action_mask'][:, :12])
                    specs.append(dict(task=task, split=split, episode=episode, step=int(step), dataset_index=int(index),
                                      **{key:value for key,value in window.items() if key != 'step'}, language=example['lang']))
    identity = dict(created_utc=datetime.now(timezone.utc).isoformat(), device='cpu', source_sha256=sha(__file__),
                    base_checkpoint=payloads[0]['base_checkpoint'], manifest_sha256=sha(args.manifest),
                    normalization_sha256=payloads[0]['normalization_sha256'], checkpoints=[dict(path=str(p.resolve()),
                    step=payload['global_step'], sha256=sha(p)) for p, payload in zip(args.checkpoints, payloads)],
                    selection_seed=args.selection_seed, window_mode=args.window_mode, selections=selections,
                    observations=specs, noise_seeds=args.noise_seeds,
                    inference_steps=args.inference_steps, prediction_dtype='CPU backbone bfloat16 autocast, Action DiT float32',
                    scope='Fixed small train/validation observation diagnostic. No optimizer updates or closed-loop trials; '
                          'CPU random draws and precision are not numerically identical to CUDA evaluation.')
    (args.output / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    print('PROBE_MODEL_LOADING', len(examples), flush=True)
    model = baseframework.from_pretrained(payloads[0]['base_checkpoint'])
    model.requires_grad_(False)
    model.eval()
    model.config.datasets.vla_data = cfg
    model.config.framework.cosmos25.training = 'action'
    model.config.trainer.repeated_diffusion_steps = 1
    assert all(p.device.type == 'cpu' for p in model.parameters())
    model.action_model.float()
    features = []
    with torch.inference_mode():
        for i, ex in enumerate(examples):
            torch.manual_seed(50000 + i)
            inputs = model.backbone_interface.build_cosmos_inputs(images=[ex['image']], instructions=[ex['lang']])
            with torch.autocast('cpu', dtype=torch.bfloat16):
                outputs = model.backbone_interface(**inputs, output_attentions=False, output_hidden_states=True, return_dict=True)
            hidden = outputs.hidden_states[-1].detach().float()
            assert torch.isfinite(hidden).all()
            features.append(hidden)
            print('PROBE_FEATURE', i + 1, len(examples), specs[i]['task'], specs[i]['split'], flush=True)
            (args.output / 'progress.json').write_text(json.dumps(dict(stage='encoding', observations_completed=i + 1,
                observations_planned=len(examples), elapsed_seconds=time.monotonic() - started)) + '\n')
        records = []
        for checkpoint_id, payload in enumerate(payloads):
            weights = {k.removeprefix('action_model.'): v for k, v in payload['trained_state'].items()}
            model.action_model.load_state_dict(weights, strict=True)
            for inference_steps in args.inference_steps:
                model.action_model.num_inference_timesteps = inference_steps
                values = []
                for i, (hidden, ex) in enumerate(zip(features, examples)):
                    state = torch.as_tensor(ex['state'][None], dtype=torch.float32)
                    for noise_seed in args.noise_seeds:
                        draw_seed = noise_seed + 1000 * i
                        torch.manual_seed(draw_seed)
                        normalized = model.action_model.predict_action(hidden, state)[0].float().numpy()
                        assert normalized.shape == (16, 32) and np.isfinite(normalized).all()
                        decoded = datasets[specs[i]['task']].decode_actions(normalized, env_order=False)
                        values.append(dict(observation=i, noise_seed=draw_seed, decoded=decoded, normalized=normalized))
                label = f"step_{payload['global_step']:06d}_iterations_{inference_steps}"
                np.savez_compressed(args.output / (label + '.npz'),
                    normalized=np.stack([v['normalized'] for v in values]), decoded=np.stack([v['decoded'] for v in values]),
                    target=np.stack([targets[v['observation']] for v in values]),
                    valid=np.stack([masks[v['observation']] for v in values]),
                    observation=np.asarray([v['observation'] for v in values]), noise_seed=np.asarray([v['noise_seed'] for v in values]))
                for task in args.tasks:
                    for split in ['train', 'validation']:
                        stages = sorted({s['stage'] for s in specs if s['task'] == task and s['split'] == split})
                        for stage in stages:
                            group = [v for v in values if specs[v['observation']]['task'] == task and
                                     specs[v['observation']]['split'] == split and specs[v['observation']]['stage'] == stage]
                            predicted = np.stack([v['decoded'] for v in group])
                            target = np.stack([targets[v['observation']] for v in group])
                            valid = np.stack([masks[v['observation']] for v in group])
                            records.append(dict(checkpoint_step=payload['global_step'], inference_steps=inference_steps,
                                task=task, split=split, stage=stage, distinct_windows=len({v['observation'] for v in group}),
                                distinct_episodes=len({specs[v['observation']]['episode'] for v in group}),
                                noise_repeats=len(args.noise_seeds), predictions=len(group), metrics=action_metrics(predicted, target, valid),
                                base_mode=binary_metrics(predicted[..., 4], target[..., 4], valid[..., 4]),
                                gripper=binary_metrics(predicted[..., 11], target[..., 11], valid[..., 11]), artifact=label + '.npz'))
                (args.output / 'partial_metrics.json').write_text(json.dumps(records, indent=2) + '\n')
                print('PROBE_CHECKPOINT_COMPLETE', payload['global_step'], inference_steps, flush=True)
    result = dict(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(), records=records,
                  seconds=time.monotonic() - started, observations=len(specs), predictions=sum(r['predictions'] for r in records),
                  identity_sha256=sha(args.output / 'identity.json'), optimizer_updates=0, policy_trials_added=0, scope=identity['scope'])
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
