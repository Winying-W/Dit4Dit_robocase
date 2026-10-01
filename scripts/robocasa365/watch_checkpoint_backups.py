"""Copy immutable milestone checkpoints to a separate mounted filesystem.

No GPU control, source changes or overwriting of differing backups. Completion
requires SHA256 readback and CPU inspection of every trained tensor/Adam moment.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import uuid

from scripts.robocasa365.audit_distributed_checkpoint import audit, file_identity, sha256


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def copy_verified(source, target, expected_sha256=None):
    original = file_identity(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.copying-' + uuid.uuid4().hex)
    copied = False
    try:
        if target.exists():
            source_sha = sha256(source)
            target_sha = sha256(target)
            if source_sha != target_sha:
                raise ValueError(f'Existing backup differs; refusing overwrite: {target}')
        else:
            digest = hashlib.sha256()
            with source.open('rb') as src, temporary.open('xb') as dst:
                for chunk in iter(lambda: src.read(8 * 1024 * 1024), b''):
                    dst.write(chunk)
                    digest.update(chunk)
                dst.flush()
                os.fsync(dst.fileno())
            source_sha = digest.hexdigest()
            target_sha = sha256(temporary)
            if target_sha != source_sha or sha256(source) != source_sha:
                raise ValueError(f'Source changed or copy readback differs: {source}')
        if expected_sha256 is not None and source_sha != expected_sha256:
            raise ValueError(f'Source identity differs from the accepted artifact: {source}')
        if file_identity(source) != original:
            raise ValueError(f'Source changed while being copied: {source}')
        if temporary.exists():
            # Atomic publish without replacing a file created by another writer.
            os.link(temporary, target)
            copied = True
        return dict(source=str(source), target=str(target), bytes=original[1],
                    sha256=source_sha, readback_sha256=target_sha, new_copy=copied)
    finally:
        temporary.unlink(missing_ok=True)


def stage_recovery_metadata(run, destination, base_backup_manifest):
    plan = read(run / 'plan.json')
    previous = read(base_backup_manifest)
    candidates = [row for row in previous['files'] if row['sha256'] == plan['base_sha256']]
    if len(candidates) != 1:
        raise ValueError('Expected one verified GR1 base in the existing backup archive')
    base = Path(previous['backup_root']) / candidates[0]['path']
    if sha256(base) != plan['base_sha256']:
        raise ValueError('Existing GR1 base backup changed')
    files = []
    for name in ('plan.json', 'source_hashes.json', 'prequeue_acceptance.json',
                 'prepared/manifest.json', 'prepared/normalization.json'):
        files.append(copy_verified(run / name, destination / 'run' / name))
    for name, expected in sorted(read(run / 'source_hashes.json').items()):
        files.append(copy_verified(run / 'source' / name,
                                   destination / 'run/source' / name, expected))
    reference = dict(path=str(base), sha256=plan['base_sha256'], bytes=base.stat().st_size,
                     backup_manifest=str(base_backup_manifest),
                     backup_manifest_sha256=sha256(base_backup_manifest))
    save(destination / 'BASE_REFERENCE.json', reference)
    result = dict(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                  files=files, base_reference=reference,
                  scope='Run configuration and source copied; GR1 already stored and rehashed '
                        'in the referenced EFS archive. Dataset media/runtime not copied.')
    save(destination / 'RECOVERY_METADATA.json', result)
    return result


def backup_step(run, training_output, destination, step):
    receipt_path = destination / f'checkpoint_{step:06d}.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt['status'] != 'complete' or receipt['step'] != step:
            raise ValueError('Existing checkpoint backup receipt is invalid')
        # Completed immutable files are fully verified when published. Avoid
        # hashing gigabytes on every poll; reject deletion/size changes here.
        for row in receipt['files']:
            if Path(row['target']).stat().st_size != row['bytes']:
                raise ValueError('Completed backup file size changed')
        return receipt
    checkpoint = training_output / f'action_step_{step:06d}.pt'
    latest_path = training_output / 'latest.json'
    if not checkpoint.is_file() or not latest_path.is_file():
        return dict(status='waiting_for_checkpoint', step=step)
    latest = read(latest_path)
    if latest.get('global_step', -1) < step or latest.get('checkpoint_readback_passed') is not True:
        return dict(status='waiting_for_checkpoint_publication', step=step)
    plan = read(run / 'plan.json')
    files = []
    for name in ('run_config.json', 'split.json', 'dataset_manifest.json', 'data_config.yaml',
                 'normalization.json', 'trainer_source.json', 'action_trainable.json'):
        files.append(copy_verified(training_output / name, destination / 'training' / name))
    files.append(copy_verified(checkpoint, destination / 'training' / checkpoint.name))
    verified = audit(destination / 'training' / checkpoint.name,
                     destination / 'run/prepared/manifest.json', step, plan.get('warm_start'))
    if verified['checkpoint_sha256'] != files[-1]['sha256']:
        raise ValueError('CPU readback checkpoint differs from the copied artifact')
    result = dict(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                  step=step, files=files, cpu_readback=verified,
                  scope='Verified backup and CPU deserialization; no CUDA restore or new training.')
    save(receipt_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'training-output', 'destination', 'base-backup-manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--steps', type=int, nargs='+', default=[10000, 25000, 50000])
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--max-hours', type=float, default=240)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60 or args.max_hours <= 0 or any(x < 1 for x in args.steps):
        parser.error('Invalid polling interval, duration or checkpoint steps')
    if len(set(args.steps)) != len(args.steps):
        parser.error('Duplicate checkpoint steps')
    for name in ('run', 'training_output', 'destination', 'base_backup_manifest', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    args.destination.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.run.stat().st_dev == args.destination.stat().st_dev:
        raise ValueError('Backup requires a separate mounted filesystem')
    import torch
    torch.set_num_threads(2)
    os.nice(10)
    stop = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda number, _frame: stop.append(number))
    with (args.destination / 'backup.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        save(args.output / 'status.json', dict(status='staging_recovery_metadata', pid=os.getpid(),
             heartbeat_utc=datetime.now(timezone.utc).isoformat(), destination=str(args.destination)))
        try:
            metadata = stage_recovery_metadata(args.run, args.destination, args.base_backup_manifest)
            save(args.output / 'recovery_metadata.json', metadata)
        except Exception as exc:
            save(args.output / 'status.json', dict(status='metadata_backup_failed', pid=os.getpid(),
                 heartbeat_utc=datetime.now(timezone.utc).isoformat(), error=f'{type(exc).__name__}: {exc}'))
            raise
        while True:
            rows = []
            for step in sorted(args.steps):
                try:
                    result = backup_step(args.run, args.training_output, args.destination, step)
                    if result['status'] == 'complete':
                        save(args.output / f'checkpoint_{step:06d}.json', result)
                    rows.append(dict(step=step, status=result['status']))
                except Exception as exc:
                    rows.append(dict(step=step, status='backup_failed', error=f'{type(exc).__name__}: {exc}'))
            status = 'complete' if all(row['status'] == 'complete' for row in rows) else 'watching'
            if any(row['status'] == 'backup_failed' for row in rows):
                status = 'backup_failed'
            if stop:
                status = 'observer_stopped'
            elif time.monotonic() - started >= args.max_hours * 3600:
                status = 'observer_time_limit'
            elif args.once and status == 'watching':
                status = 'one_pass_waiting'
            save(args.output / 'status.json', dict(status=status, pid=os.getpid(),
                 heartbeat_utc=datetime.now(timezone.utc).isoformat(), checkpoints=rows,
                 destination=str(args.destination), source_device=args.run.stat().st_dev,
                 backup_device=args.destination.stat().st_dev, gpu_control=False,
                 scope='Checkpoint backups only; no claims of CUDA recovery or policy success.'))
            if status != 'watching':
                if status == 'backup_failed':
                    raise RuntimeError('Checkpoint backup failed; preserve evidence and inspect before restarting')
                break
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
