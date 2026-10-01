"""Submit the accepted Composite candidate once, after Atomic evaluation releases GPUs.

The CPU watcher cannot stop existing jobs. Cloud observation errors are retries,
but submission errors are never retried automatically. A durable intent prevents
duplicate submission after a timeout or process restart.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import yaml


ACTIVE = 'Queue,Staging,Running,Killing,Initialized'
ALL_STATUSES = ACTIVE + ',Success,Failed,Killed'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


class Cloud:
    def __init__(self):
        self.env = os.environ.copy()
        self.env['PATH'] = str(Path.home() / '.volc/bin') + os.pathsep + self.env['PATH']

    def call(self, args, timeout=30):
        response = subprocess.run(['volc', 'ml_task', *args], env=self.env,
                                  capture_output=True, text=True, timeout=timeout)
        if response.returncode:
            raise RuntimeError(f'Cloud CLI exit {response.returncode}: {response.stderr[-1500:]}')
        output = response.stdout.strip()
        # volc 1.2.57 prints this notice before JSON for an empty list.
        # Accept only the observed empty response; do not hide other CLI text.
        if (args and args[0] == 'list' and
                output.splitlines() == ['没有匹配条件的任务', '', '[]']):
            return []
        return json.loads(output)

    def get(self, task_id):
        rows = self.call(['get', '--id', task_id, '--output', 'json',
                          '--format', 'Id,Name,Status,ExitCode'])
        if len(rows) != 1 or rows[0]['Id'] != task_id:
            raise ValueError('Cloud dependency identity is missing or ambiguous')
        return rows[0]

    def find(self, name, active_only=False):
        result = []
        for offset in range(0, 10000, 100):
            rows = self.call(['list', '--name', name, '--status', ACTIVE if active_only else ALL_STATUSES,
                              '--limit', '100', '--offset', str(offset), '--output', 'json',
                              '--format', 'Id,Name,Status'])
            result.extend(rows)
            if len(rows) < 100:
                if len({row['Id'] for row in result}) != len(result):
                    raise ValueError('Cloud pagination returned duplicate task identities')
                return result
        raise ValueError('Cloud list exceeded bounded pagination; do not assume capacity is free')

    def submit(self, config):
        return self.call(['submit', '--conf', str(config), '--output', 'json'], timeout=120)


def candidate(run, verify_weights=False):
    """Verify existing accepted evidence; never rewrite the frozen candidate."""
    acceptance = read(run / 'prequeue_acceptance.json')
    plan = read(run / 'plan.json')
    config = yaml.safe_load((run / 'task.yaml').read_text())
    if acceptance['status'] != 'complete' or not acceptance['checks']['warm_start']['passed']:
        raise ValueError('Composite prequeue is incomplete')
    for name, digest in acceptance['evidence_sha256'].items():
        if sha(run / name) != digest:
            raise ValueError(f'Accepted evidence changed: {name}')
    sources = read(run / 'source_hashes.json')
    if sha(run / 'source_hashes.json') != acceptance['source_hashes_sha256']:
        raise ValueError('Source manifest changed')
    for name, digest in sources.items():
        if sha(run / 'source' / name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    roles = config['TaskRoleSpecs']
    if (len(roles) != 1 or roles[0]['RoleReplicas'] != 1 or
            roles[0]['ResourceSpec']['GPUNum'] != 4 or plan['gpu_count'] != 4):
        raise ValueError('Expected exactly one four-GPU allocation')
    if (plan['task_set'] != 'composite_seen' or plan['task_count'] != 16 or
            plan['planned_updates'] != 50000 or plan['planned_final_trials'] != 1600):
        raise ValueError('Unexpected Composite scope')
    if config['Entrypoint'] != f"bash {run}/source/{plan['entrypoint_script']}":
        raise ValueError('Cloud entrypoint does not select the accepted frozen source')
    if any(s['Type'] == 'Nas' for s in config['Storages']):
        raise ValueError('Current candidate must not depend on the retired NAS')
    if plan['scene_protocol'] != acceptance['scene_protocol']:
        raise ValueError('Scene protocol changed')
    warm = plan['warm_start']
    if warm['source_global_step'] != 50000 or warm['local_initial_step'] != 0:
        raise ValueError('Unexpected action initialization')
    if verify_weights and sha(Path(warm['checkpoint'])) != warm['checkpoint_sha256']:
        raise ValueError('Atomic initialization checkpoint changed')
    return dict(task_name=config['TaskName'],task_yaml_sha256=sha(run / 'task.yaml'),
                prequeue_sha256=sha(run / 'prequeue_acceptance.json'),
                source_manifest_sha256=sha(run / 'source_hashes.json'),
                source_files=len(sources),gpu_count=4,checkpoint_sha256=warm['checkpoint_sha256'])


def reviews_complete(run, observer):
    """Require complete development/final reviews of the current exact reports."""
    plan = read(run / 'plan.json')
    if (plan['task_set'], plan['task_count'], plan['final_checkpoint'], plan['planned_final_trials']) != ('atomic_seen', 18, 50000, 1800):
        raise ValueError('Unexpected Atomic dependency scope')
    manifest = run / 'prepared/manifest.json'
    task_names = {row['task'] for row in read(manifest)['tasks']}
    if len(task_names) != 18:
        raise ValueError('Expected complete Atomic18 task set')
    gates = [(f'development_{step}', step, 'development', 360) for step in plan['development_checkpoints']]
    gates += [('evaluation_1800', 50000, 'final', 1800)]
    records = []
    for name, step, purpose, count in gates:
        report_path = run / name / 'report.json'
        if not report_path.is_file():
            return None
        digest = sha(report_path)
        report = read(report_path)
        review_path = observer / name / f'review_{digest}.json'
        if report['status'] != 'complete' or not review_path.is_file():
            return None
        review = read(review_path)
        if review['status'] != 'complete':
            return None
        if (review['report_sha256'] != digest or review['manifest_sha256'] != sha(manifest) or
                review['identity_sha256'] != sha(run / name / 'identity.json') or
                review['checkpoint_step'] != step or review['purpose'] != purpose or
                review['task_set'] != 'atomic_seen' or not review['saved_action_chain_passed'] or
                review['planned_trials'] != count or review['completed_trials'] != count or
                report['completed_trials'] != count or report['successes'] != review['successes'] or
                {row['task'] for row in review['tasks']} != task_names or len(review['tasks']) != 18 or
                not all(row['complete'] and row['action_chain_passed'] and
                        row['completed_trials'] == count // 18 for row in review['tasks'])):
            raise ValueError(f'Incomplete or mismatched independent audit: {name}')
        records.append(dict(gate=name,review=str(review_path),sha256=sha(review_path),
                            completed_trials=count,successes=review['successes']))
    if (run / 'stage.txt').read_text().strip() != 'complete':
        return None
    return records


def advance(cloud, *, dependency_task, dependency_run, observer, target, output, accepted, submit):
    """One observation/decision; caller must hold target's submission lock."""
    receipt = target / 'submission.json'
    intent = target / 'submission_intent.json'
    if receipt.exists():
        record = read(receipt)
        if record['candidate'] != accepted:
            raise ValueError('Existing submission selects a different candidate')
        task = cloud.get(record['Id'])
        if task['Name'] != accepted['task_name']:
            raise ValueError('Submitted cloud task name does not match the candidate')
        return dict(status='already_submitted',target=task)
    # Resolve a previous uncertain call by name, but never issue another submit.
    if intent.exists():
        matches = [r for r in cloud.find(accepted['task_name']) if r['Name'] == accepted['task_name']]
        return dict(status='submission_requires_reconciliation',matches=matches,intent=str(intent))
    dependency = cloud.get(dependency_task)
    if dependency['Status'] in ('Failed', 'Killed', 'Canceled', 'Cancelled', 'Stopped'):
        return dict(status='dependency_failed',dependency=dependency)
    if dependency['Status'] != 'Success':
        return dict(status='waiting_for_dependency',dependency=dependency)
    if dependency.get('ExitCode') != 0:
        return dict(status='dependency_failed',dependency=dependency)
    reviews = reviews_complete(dependency_run, observer)
    if reviews is None:
        return dict(status='waiting_for_complete_reviews',dependency=dependency)
    matches = [r for r in cloud.find(accepted['task_name']) if r['Name'] == accepted['task_name']]
    if matches:
        return dict(status='existing_target_requires_reconciliation',matches=matches,dependency=dependency)
    active = cloud.find('dit4dit', active_only=True)
    if active:
        return dict(status='waiting_for_project_capacity',active_tasks=active,dependency=dependency)
    fresh = candidate(target, verify_weights=True)
    if fresh != accepted:
        raise ValueError('Candidate changed after the watcher was armed')
    ready = dict(status='ready',dependency=dependency,reviews=reviews,candidate=fresh)
    if not submit:
        return ready
    # Weight verification takes time: refresh capacity immediately before submit.
    active = cloud.find('dit4dit', active_only=True)
    if active:
        return dict(status='waiting_for_project_capacity',active_tasks=active,dependency=dependency)
    # Persist BEFORE the only mutating API call. Any timeout/crash leaves this marker.
    record = dict(created_utc=utc(),dependency_task=dependency_task,candidate=fresh,
                  reviews=reviews,config=str(target / 'task.yaml'),retry_allowed=False)
    with intent.open('x') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    try:
        result = cloud.submit(target / 'task.yaml')
        if not isinstance(result, dict) or not result.get('Id'):
            raise ValueError('Submission returned no task ID')
        save(receipt,dict(Id=result['Id'],submitted_utc=utc(),candidate=fresh,
                          dependency_task=dependency_task,submission_intent=str(intent)))
        save(output / 'submission_response.json',result)
        task = cloud.get(result['Id'])
        if task['Name'] != fresh['task_name']:
            raise ValueError('Submitted cloud task name does not match the candidate')
        return dict(status='submitted',target=task,candidate=fresh)
    except Exception as exc:
        # The server may have accepted the job even if the client timed out.
        return dict(status='submission_requires_reconciliation',intent=str(intent),
                    error=f'{type(exc).__name__}: {exc}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependency-task', required=True)
    parser.add_argument('--dependency-run', type=Path, required=True)
    parser.add_argument('--observer', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--submit-on-ready', action='store_true')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--max-hours', type=float, default=24)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60 or args.max_hours <= 0:
        parser.error('Polling must be 1–60 seconds and time limit positive')
    for name in ('dependency_run','observer','target','output'):
        setattr(args,name,getattr(args,name).resolve())
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.target / 'submission.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        accepted = candidate(args.target)
        save(args.output / 'armed_candidate.json',dict(verified_utc=utc(),**accepted))
        stop = []
        for sig in (signal.SIGINT,signal.SIGTERM):
            signal.signal(sig,lambda number,_frame: stop.append(number))
        start = time.monotonic()
        cloud = Cloud()
        previous = None
        while not stop:
            try:
                state = advance(cloud,dependency_task=args.dependency_task,dependency_run=args.dependency_run,
                                observer=args.observer,target=args.target,output=args.output,
                                accepted=accepted,submit=args.submit_on_ready)
            except (subprocess.SubprocessError,RuntimeError,json.JSONDecodeError) as exc:
                state = dict(status='observation_retry',error=f'{type(exc).__name__}: {exc}')
            except Exception as exc:
                state = dict(status='validation_failed',error=f'{type(exc).__name__}: {exc}')
            state.update(heartbeat_utc=utc(),pid=os.getpid(),submit_on_ready=args.submit_on_ready)
            save(args.output / 'status.json',state)
            if state['status'] != previous:
                print(json.dumps(state),flush=True)
                previous = state['status']
            waiting = state['status'].startswith('waiting_') or state['status'] == 'observation_retry'
            if args.once or not waiting:
                return
            if time.monotonic() - start >= args.max_hours * 3600:
                save(args.output / 'status.json',dict(state,status='watcher_time_limit',heartbeat_utc=utc()))
                return
            time.sleep(args.poll_seconds)
        save(args.output / 'status.json',dict(status='watcher_stopped',pid=os.getpid(),heartbeat_utc=utc()))


if __name__ == '__main__':
    main()
