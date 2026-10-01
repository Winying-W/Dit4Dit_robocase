"""Read all trained weights and optimizer moments from a saved DDP checkpoint.

Runs on CPU and leaves the training process and checkpoint untouched. This checks
serialization/integrity; actual CUDA resume is a separate executed short gate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def verify_initialization(payload, config, expected_initialization):
    """Keep released-base and explicit Atomic-to-Composite lineage distinct."""
    lineage = payload['warm_start']
    assert lineage == config.get('warm_start'), 'Checkpoint/config initialization differs'
    if expected_initialization is None:
        assert lineage is None, 'Warm-start audit requires the explicit run plan'
        return None
    assert isinstance(lineage, dict), 'Expected warm-start lineage is absent'
    for field in ('protocol', 'checkpoint', 'checkpoint_sha256', 'source_run'):
        assert lineage[field] == expected_initialization[field], field
    assert lineage['global_step'] == expected_initialization['source_global_step']
    assert expected_initialization['local_initial_step'] == 0
    assert all(lineage.get(field) is True for field in
               ('optimizer_reset', 'scheduler_reset', 'sampler_reset', 'local_steps_start_at_zero'))
    assert lineage['transferred_tensors'] == len(payload['trained_state']) == 247
    assert payload['training_schedule']['initialization_checkpoint_sha256'] == lineage['checkpoint_sha256']
    return dict(protocol=lineage['protocol'], checkpoint_sha256=lineage['checkpoint_sha256'],
                source_global_step=lineage['global_step'], local_initial_step=0,
                optimizer_scheduler_sampler_reset=True)


def audit(checkpoint, manifest, expected_step, expected_initialization=None):
    original = file_identity(checkpoint)
    directory = checkpoint.parent
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert payload['format_version'] == 2 and payload['phase'] == 'action'
    assert payload['global_step'] == payload['phase_step'] == expected_step
    config = json.loads((directory / 'run_config.json').read_text())
    shapes = json.loads((directory / 'action_trainable.json').read_text())
    split = json.loads((directory / 'split.json').read_text())
    assert payload['split'] == split
    assert split['manifest_sha256'] == sha256(manifest)
    assert payload['base_checkpoint'] == config['base_checkpoint']
    assert payload['training_schedule'] == config['training_schedule']
    assert payload['normalization_sha256'] == sha256(directory / 'normalization.json')
    initialization = verify_initialization(payload, config, expected_initialization)

    weights = payload['trained_state']
    assert len(weights) == 247 and set(weights) == set(shapes)
    for name, tensor in weights.items():
        assert name.startswith('action_model.')
        assert list(tensor.shape) == shapes[name], name
        assert torch.isfinite(tensor).all(), name

    optimizer = payload['optimizer']
    identifiers = [key for group in optimizer['param_groups'] for key in group['params']]
    assert len(identifiers) == len(set(identifiers)) == len(weights)
    assert set(optimizer['state']) == set(identifiers)
    # Trainer constructs the optimizer in named_parameters / trained_state order.
    for (name, tensor), key in zip(weights.items(), identifiers):
        state = optimizer['state'][key]
        assert float(state['step']) == expected_step, name
        for moment in ['exp_avg', 'exp_avg_sq']:
            value = state[moment]
            assert value.shape == tensor.shape and torch.isfinite(value).all(), (name, moment)
        assert torch.all(state['exp_avg_sq'] >= 0), name
    assert payload['scheduler']['last_epoch'] == expected_step
    assert payload['scheduler']['_last_lr'] == [group['lr'] for group in optimizer['param_groups']]

    world = payload['training_schedule']['world_size']
    assert len(payload['distributed_rng']) == world == 4
    streams = []
    for state in payload['distributed_rng']:
        sampler = np.random.default_rng()
        sampler.bit_generator.state = state['sampler']
        random.Random().setstate(state['python'])
        np.random.RandomState().set_state(state['numpy'])
        torch.Generator(device='cpu').set_state(state['torch'])
        cuda = state['cuda']
        assert cuda.dtype == torch.uint8 and cuda.ndim == 1 and cuda.numel() > 0
        streams.append(json.dumps(state['sampler'], sort_keys=True))
    assert len(set(streams)) == world

    digest = sha256(checkpoint)
    assert file_identity(checkpoint) == original, 'Checkpoint path changed while being inspected; retry the newer saved step'
    return dict(status='complete', verified_utc=datetime.now(timezone.utc).isoformat(),
                checkpoint=str(checkpoint.resolve()), global_step=expected_step,
                checkpoint_bytes=original[1], checkpoint_sha256=digest,
                trained_tensors=len(weights), trained_parameters=sum(tensor.numel() for tensor in weights.values()),
                all_trained_weights_finite=True, all_adam_moments_finite=True,
                all_optimizer_steps_match=True, scheduler_step_matches=True,
                rng_states=world, cpu_rng_states_deserialized=True,
                independent_sampler_states=True, cuda_rng_buffers_present=True,
                split_and_normalization_match=True,
                initialization=initialization,
                source_sha256=sha256(Path(__file__)),
                scope='Read-only CPU audit of every trained tensor and Adam moment in the saved checkpoint. '
                      'CUDA RNG buffers were checked structurally, not restored on a GPU here. '
                      'This is not a new training update, full-model inference, or success-rate evaluation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-plan', type=Path,
                        help='Explicit planned initialization, required for warm-started runs')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.output.exists():
        raise FileExistsError('Preserve existing audit evidence; choose a new output path')
    initialization = json.loads(args.run_plan.read_text()).get('warm_start') if args.run_plan else None
    result = audit(args.checkpoint, args.manifest, args.step, initialization)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
