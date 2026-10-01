"""Watch an existing run on CPU and review changed evaluation reports.

This process cannot submit, stop, modify or restart a GPU job. Its heartbeat and
immutable review records make its actual progress inspectable across turns.
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

from scripts.robocasa365.review_evaluation import read, require, review
from scripts.robocasa365.scene_protocol import protocol_spec


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def validation_history(output):
    path = output / 'metrics.jsonl'
    if not path.exists():
        return []
    rows = []
    with path.open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A trailing live record may still be being written.
            if 'validation_action_fm_loss' in row:
                rows.append(dict(step=row['global_step'], action_fm_loss=row['validation_action_fm_loss']))
    return rows


def validation_windows(training_output, manifest_path):
    """Describe the trainer's actual fixed windows, never infer from task count."""
    path = training_output / 'split.json'
    if not path.is_file():
        return dict(status='waiting_for_training_split', count=None, source=str(path))
    raw = path.read_bytes()
    split = json.loads(raw)
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    require(split['manifest_sha256'] == manifest_sha, 'Training split manifest identity mismatch')
    require(split['task_set'] == manifest['task_set'], 'Training split task set mismatch')
    require([row['task'] for row in split['tasks']] == [row['task'] for row in manifest['tasks']],
            'Training split task order mismatch')
    windows, per_task = [], []
    for index, (actual, planned) in enumerate(zip(split['tasks'], manifest['tasks'])):
        for field in ('train_episodes', 'validation_episodes'):
            require(actual[field] == planned[field], f'Training split {field} changed')
        require(set(actual['train_episodes']).isdisjoint(actual['validation_episodes']),
                'Training and validation episodes overlap')
        indices = actual['validation_indices']
        require(indices and all(type(value) is int and value >= 0 for value in indices),
                'Invalid fixed validation indices')
        require(len(indices) == len(set(indices)), 'Duplicate fixed validation window')
        windows.extend([[index, value] for value in indices])
        per_task.append(dict(task=actual['task'], windows=len(indices),
                             held_out_episodes=len(actual['validation_episodes'])))
    require(windows == split['validation_indices'], 'Flattened validation windows mismatch')
    require(bool(windows), 'Empty fixed validation windows')
    return dict(status='verified_training_split', count=len(windows), source=str(path),
                sha256=hashlib.sha256(raw).hexdigest(), manifest_sha256=manifest_sha,
                tasks=per_task,
                scope='Offline loss windows recorded by the trainer; not closed-loop trials '
                      'or a count of all held-out demonstrations.')


def scan(run, training_output, output):
    plan = read(run / 'plan.json')
    manifest = run / 'prepared/manifest.json'
    expected_protocol = plan.get('scene_protocol', protocol_spec())
    window_metadata = validation_windows(training_output, manifest)
    gates = [(f'development_{step}', step, 'development') for step in plan['development_checkpoints']]
    gates.append((f"evaluation_{plan['planned_final_trials']}", plan['final_checkpoint'], 'final'))
    records = []
    for name, step, purpose in gates:
        source = run / name / 'report.json'
        item = dict(gate=name, checkpoint_step=step, purpose=purpose, status='waiting_for_evaluation')
        if source.is_file():
            signature = hashlib.sha256(source.read_bytes()).hexdigest()
            destination = output / name
            destination.mkdir(exist_ok=True)
            report_path = destination / f'review_{signature}.json'
            try:
                if not report_path.exists():
                    result = review(source.parent, manifest, step, purpose)
                    # Source may advance while being reviewed: name the captured version.
                    report_path = destination / f"review_{result['report_sha256']}.json"
                    if not report_path.exists():
                        save(report_path, result)
                    print(json.dumps(dict(event='evaluation_review', gate=name, status=result['status'],
                                          trials=result['completed_trials'], successes=result['successes'])), flush=True)
                result = read(report_path)
                require(result.get('scene_protocol', protocol_spec()) == expected_protocol,
                        'Reviewed scene protocol differs from the run plan')
                item.update(status=result['status'], evidence=str(report_path),
                            completed_trials=result['completed_trials'], successes=result['successes'],
                            pooled_success_rate=result['pooled_success_rate'],
                            recommended_next_action=result['recommended_next_action'])
            except Exception as exc:
                failure = dict(status='review_failed', observed_utc=datetime.now(timezone.utc).isoformat(),
                               source=str(source), source_sha256=signature, error=f'{type(exc).__name__}: {exc}',
                               next_action='Inspect this review failure before interpreting benchmark results.')
                error_path = destination / f'error_{signature}.json'
                if not error_path.exists():
                    save(error_path, failure)
                    print(json.dumps(failure), flush=True)
                item.update(status='review_failed', evidence=str(error_path), error=failure['error'])
        records.append(item)
    stage = (run / 'stage.txt').read_text().strip() if (run / 'stage.txt').exists() else None
    latest = read(training_output / 'latest.json') if (training_output / 'latest.json').exists() else None
    live_path = run / 'train_live/status.json'
    live = read(live_path) if live_path.exists() else {}
    return dict(heartbeat_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), stage=stage,
                live_step=live.get('global_step', live.get('global_steps')),
                live_status_age_seconds=time.time() - live_path.stat().st_mtime if live_path.exists() else None,
                last_saved_checkpoint=latest, gates=records,
                fixed_validation_windows=window_metadata['count'], validation_window_metadata=window_metadata,
                scene_protocol=expected_protocol, validation_history=validation_history(training_output),
                full_behavioral_diagnosis_completed=False, gpu_jobs_submitted=0,
                scope='Local CPU observer and saved-action review. No GPU control, new policy trials or '
                      'semantic replay. Heartbeat age must be checked; local process survival is not guaranteed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--training-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--max-hours', type=float, default=120)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60 or args.max_hours <= 0:
        parser.error('Polling must be between 1 and 60 seconds; max-hours must be positive')
    args.run, args.training_output, args.output = (p.resolve() for p in (args.run, args.training_output, args.output))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'observer.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stop = []
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda number, _frame: stop.append(number))
        started = time.monotonic()
        while True:
            try:
                state = scan(args.run, args.training_output, args.output)
                status = ('complete' if all(row['status'] == 'complete' for row in state['gates']) else
                          'run_failed' if (state['stage'] or '').startswith('failed') else 'watching')
                if args.once:
                    status = 'one_pass_complete'
                if stop:
                    status = 'observer_stopped'
                elif time.monotonic() - started >= args.max_hours * 3600:
                    status = 'observer_time_limit'
                state['observer_status'] = status
                save(args.output / 'status.json', state)
                if status != 'watching':
                    print(json.dumps(dict(event='observer_exit', status=status)), flush=True)
                    break
            except Exception as exc:
                save(args.output / 'status.json', dict(observer_status='read_error', pid=os.getpid(),
                     heartbeat_utc=datetime.now(timezone.utc).isoformat(), error=f'{type(exc).__name__}: {exc}'))
                if args.once or stop or time.monotonic() - started >= args.max_hours * 3600:
                    raise
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
