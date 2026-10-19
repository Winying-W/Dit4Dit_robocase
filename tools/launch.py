"""Portable single-node launcher. Keeps the declared four-rank/global-128 recipe."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]


def parser(experiment):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--cosmos-model', type=Path, required=True)
    p.add_argument('--qwen-model', type=Path, required=experiment == 'concat_random_head')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, required=True, help='Final schedule length, unchanged when resuming')
    p.add_argument('--microbatch', type=int, default=1, help='Must divide 32; accumulation is 32/microbatch')
    p.add_argument('--gpus', default='0,1,2,3', help='Four CUDA device IDs or UUIDs visible to this process')
    p.add_argument('--stop-after', type=int, help='Stop at this global step, retaining the full --steps schedule')
    p.add_argument('--resume', action='store_true', help='Resume output/action_latest.pt including optimizer/RNG')
    p.add_argument('--checkpoint-interval', type=int, default=1000)
    p.add_argument('--validation-interval', type=int, default=1000)
    p.add_argument('--dry-run', action='store_true', help='Validate arguments/assets and print command; do not train')
    p.add_argument('--cpu-runtime-preflight', action='store_true', help='Four-rank actual trainer import/Gloo check; no model loading')
    return p


def build_command(a, experiment):
    if a.steps < 2:
        raise ValueError('--steps must be >= 2')
    if a.microbatch < 1 or 32 % a.microbatch:
        raise ValueError('--microbatch must be a positive divisor of 32')
    devices = a.gpus.split(',')
    if len(devices) != 4 or len(set(devices)) != 4 or any(not x.strip() for x in devices):
        raise ValueError('--gpus must specify exactly four distinct devices')
    if a.stop_after is not None and not 1 <= a.stop_after <= a.steps:
        raise ValueError('--stop-after must be between 1 and --steps')
    if min(a.checkpoint_interval, a.validation_interval) < 1:
        raise ValueError('Checkpoint/validation intervals must be positive')
    if experiment == 'robocasa_official_recipe' and a.qwen_model:
        raise ValueError('The baseline must not attach Qwen')
    if experiment == 'concat_random_head' and not a.qwen_model:
        raise ValueError('Concat requires --qwen-model')
    snapshots = sorted({min(100, a.steps), a.steps, *range(5000, a.steps + 1, 5000)})
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=4',
           '-m', 'scripts.robocasa365.train_distributed', '--manifest', str(a.manifest.resolve()),
           '--checkpoint', str(a.checkpoint.resolve()), '--output', str(a.output.resolve()),
           '--steps', str(a.steps), '--microbatch', str(a.microbatch), '--accumulation', str(32 // a.microbatch),
           '--warmup-steps', str(max(1, round(a.steps * .025))), '--seed', '42',
           '--checkpoint-interval', str(a.checkpoint_interval), '--validation-interval', str(a.validation_interval),
           '--action-head-init', 'random' if experiment == 'concat_random_head' else 'gr1',
           '--snapshot-steps', *map(str, snapshots)]
    if a.qwen_model:
        cmd += ['--qwen-model', str(a.qwen_model.resolve())]
    if a.resume:
        cmd += ['--resume', str(a.output.resolve() / 'action_latest.pt')]
    if a.stop_after is not None:
        cmd += ['--stop-after', str(a.stop_after)]
    if a.cpu_runtime_preflight:
        cmd += ['--cpu-runtime-preflight']
    return cmd


def validate_assets(a):
    required = [a.manifest, a.manifest.parent/'normalization.json', a.checkpoint,
                a.checkpoint.resolve().parents[1]/'config.yaml',
                a.checkpoint.resolve().parents[1]/'dataset_statistics.json', a.cosmos_model/'model_index.json']
    for part in ['transformer', 'vae', 'text_encoder', 'tokenizer', 'scheduler']:
        if not (a.cosmos_model/part).is_dir():
            raise FileNotFoundError(a.cosmos_model/part)
    if a.qwen_model:
        required += [a.qwen_model/'config.json', a.qwen_model/'preprocessor_config.json', a.qwen_model/'tokenizer_config.json']
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(p)
    for component in ['transformer', 'vae', 'text_encoder']:
        directory = a.cosmos_model/component
        if not (directory/'config.json').is_file():
            raise FileNotFoundError(directory/'config.json')
        if not any(f.stat().st_size for pattern in ['*.safetensors', '*.bin'] for f in directory.glob(pattern)):
            raise FileNotFoundError(f'Missing Cosmos component weights: {directory}')
    if a.qwen_model:
        qconfig = json.loads((a.qwen_model/'config.json').read_text())
        if qconfig.get('model_type') != 'qwen3_vl' or qconfig.get('text_config', {}).get('hidden_size') != 2560:
            raise ValueError('This experiment requires Qwen3-VL-4B (hidden_size=2560)')
        index = a.qwen_model/'model.safetensors.index.json'
        shards = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
        for shard in shards:
            if not (a.qwen_model/shard).is_file() or not (a.qwen_model/shard).stat().st_size:
                raise FileNotFoundError(a.qwen_model/shard)
    m = json.loads(a.manifest.read_text())
    if len(m['tasks']) != 18 or m['task_set'] != 'atomic_seen':
        raise ValueError('Expected the Atomic18 manifest')
    if hashlib.sha256((a.manifest.parent/'normalization.json').read_bytes()).hexdigest() != m['normalization_sha256']:
        raise ValueError('Normalization hash mismatch')
    for row in m['tasks']:
        path = Path(row['path'])
        if not path.is_absolute():
            raise ValueError('Run tools/prepare_data.py to materialize absolute data paths first')
        for rel in ['meta/info.json', 'meta/modality.json', 'meta/episodes.jsonl', 'meta/tasks.jsonl', 'meta/stats.json']:
            if not (path/rel).is_file():
                raise FileNotFoundError(path/rel)
        if not set(row['train_episodes']).isdisjoint(row['validation_episodes']):
            raise ValueError(f"Overlapping train/validation episodes: {row['task']}")
    if a.resume and not (a.output/'action_latest.pt').is_file():
        raise FileNotFoundError(a.output/'action_latest.pt')


def main(experiment):
    a = parser(experiment).parse_args()
    cmd = build_command(a, experiment)
    if not a.cpu_runtime_preflight:
        validate_assets(a)
    source = ROOT/'experiments'/experiment/'source'
    env = os.environ.copy()
    env.update(PYTHONPATH=str(source), COSMOS_MODEL_DIR=str(a.cosmos_model.resolve()),
               NO_ALBUMENTATIONS_UPDATE='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS=env.get('OMP_NUM_THREADS', '1'),
               CUDA_VISIBLE_DEVICES='' if a.cpu_runtime_preflight else a.gpus)
    print(shlex.join(cmd), flush=True)
    if a.dry_run:
        return
    if not a.cpu_runtime_preflight:
        # Check GPU visibility before allocating any model weights or spawning ranks.
        subprocess.run([sys.executable, '-c', 'import torch; assert torch.cuda.device_count()==4, "Exactly four visible CUDA GPUs required"; print([(i,torch.cuda.get_device_name(i),torch.cuda.get_device_properties(i).total_memory) for i in range(4)])'], env=env, check=True)
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output/'launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not a.cpu_runtime_preflight:
            sys.path.insert(0, str(source))
            from scripts.robocasa365.recovery import prepare_attempt
            if not a.resume and any((a.output/name).exists() for name in ['action_latest.pt', 'metrics.jsonl', 'run_config.json']):
                raise ValueError('Existing training output: use --resume or a fresh output directory')
            if a.resume:
                prepare_attempt(a.output, uuid.uuid4().hex)
        subprocess.run(cmd, cwd=source, env=env, check=True)
