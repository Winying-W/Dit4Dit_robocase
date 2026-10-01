"""Audit the declared sampling A/B and compare all saved development scene pairs.

This CPU observer cannot start, stop, or alter training or policy evaluations.
It uses the immutable experiment plan, not a post-hoc task or seed selection.
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

from scripts.robocasa365.compare_development_checkpoints import compare
from scripts.robocasa365.review_evaluation import require


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def declaration(run):
    accepted = read(run / 'prequeue_acceptance.json')
    require(accepted['status'] == 'complete', 'Experiment prequeue was not accepted')
    for name in ['plan.json', 'cloud_source_hashes.json']:
        require(sha(run / name) == accepted['evidence_sha256'][name], 'Frozen experiment changed: ' + name)
    plan = read(run / 'plan.json')
    require(plan['branches'] == ['paired_uniform', 'initial_switch'], 'Unexpected sampling branches')
    require(plan['additional_updates_per_branch'] == 2000, 'Unexpected continuation budget')
    require(sha(plan['manifest']) == plan['manifest_sha256'], 'Training manifest changed')
    manifest = read(plan['manifest'])
    names = [row['task'] for row in manifest['tasks']]
    require(manifest['task_set'] == 'composite_seen' and manifest['task_scope'] == 'full_official_task_set'
            and len(names) == len(set(names)) == 16, 'Expected full Composite16 training manifest')
    requested = plan['development_tasks']
    tasks = [name for name in names if name in requested]
    seeds = plan['development_seeds']
    require(len(requested) == len(set(requested)) == len(tasks) == 3, 'Invalid declared three-task screen')
    require(seeds == manifest['evaluation']['selection_seeds'] and len(seeds) == len(set(seeds)) == 20,
            'Unexpected development seeds')
    require(set(seeds).isdisjoint(manifest['evaluation']['seeds']), 'Final seed contamination')
    require(plan['development_trials_per_branch'] == len(tasks) * len(seeds) == 60, 'Unexpected screen size')
    return plan, tasks


def ready_inputs(run, plan, tasks):
    signatures = []
    for branch in plan['branches']:
        folder = run / branch / 'development_2000'
        report_path = folder / 'report.json'
        if not report_path.is_file():
            return None
        report = read(report_path)
        require(report['purpose'] == 'development', 'Final evaluation cannot select a sampling strategy')
        if report['status'] != 'complete':
            return None
        require(report['task_scope'] == 'declared_task_subset' and report['planned_tasks'] == tasks,
                'Report task scope differs from the declared screen')
        require(report['seeds'] == plan['development_seeds'], 'Report seed set differs')
        require(report['completed_trials'] == report['planned_trials'] == 60, 'Incomplete screen denominator')
        signatures.append(dict(branch=branch, report=str(report_path), report_sha256=sha(report_path),
                               identity_sha256=sha(folder / 'identity.json')))
    return signatures


def checkpoint_and_inference_contract(run, plan):
    # Import torch only when both complete reports are available.
    from scripts.robocasa365.audit_sampling_checkpoint import audit
    import torch
    torch.set_num_threads(2)
    audits, inference = [], []
    expected_inference = ['inference_timesteps', 'image_size', 'architecture', 'phase',
                          'execute_horizon', 'protocol']
    for branch in plan['branches']:
        folder = run / branch / 'development_2000'
        identity = read(folder / 'identity.json')
        checkpoint = run.parents[1] / 'artifacts/training' / run.name / branch / 'action_step_002000.pt'
        require(Path(identity['checkpoint']).resolve() == checkpoint.resolve(), 'Branch checkpoint path differs')
        weights = identity['policy_artifacts']['incremental_weights']
        require(Path(weights['path']).resolve() == checkpoint.resolve(), 'Incremental weights path differs')
        checked = audit(checkpoint, run, branch, 2000)
        require(checked['checkpoint_sha256'] == weights['sha256']
                and checked['checkpoint_bytes'] == weights['bytes'], 'Evaluated weights differ from audited checkpoint')
        require(identity['policy_artifacts']['base_weights']['sha256'] == plan['base_sha256'], 'Base model differs')
        require(identity['scene_protocol']['name'] == plan['scene_protocol'], 'Scene protocol differs from plan')
        report = read(folder / 'report.json')
        attempts = {Path(trial['artifact']).parent for task in report['tasks'] for trial in task['trials']}
        require(bool(attempts), 'No actual policy attempts')
        for attempt in sorted(attempts):
            verification = read(attempt / 'checkpoint_verification.json')
            require(verification['warm_start']['checkpoint_sha256'] == plan['warm_start_sha256'],
                    'Evaluation initialization lineage differs')
            inference.append({key: verification[key] for key in expected_inference})
        audits.append(checked)
    require(all(item == inference[0] for item in inference), 'Policy inference configuration differs')
    return audits, inference[0]


def compare_run(run, plan, tasks, signatures):
    checkpoints, inference = checkpoint_and_inference_contract(run, plan)
    directories = [run / branch / 'development_2000' for branch in plan['branches']]
    result = compare(*directories, plan['manifest'], 2000, 2000,
                     requested_tasks=tasks, same_step_branches=True)
    require(result['planned_pairs'] == 60 and result['task_scope'] == 'declared_task_subset',
            'Comparison denominator or scope differs')
    require(ready_inputs(run, plan, tasks) == signatures, 'Evaluation files changed during comparison')
    result.update(branches=plan['branches'], checkpoint_contracts=checkpoints,
                  common_inference_configuration=inference, input_signatures=signatures,
                  experiment_plan_sha256=sha(run / 'plan.json'), source_composite_step=10000,
                  additional_training_steps_per_branch=2000, full16_success_rate=None,
                  new_policy_trials=0,
                  scope='All60 predeclared three-task development pairs. Independent checkpoint/action audits '
                        'and saved initial XML/state matching. Not the full16-task benchmark, physical milestone '
                        'replay, or a video-unfreezing experiment; no new policy trials.')
    return result


def validate_result(result, signatures, plan):
    require(result['status'] in ['complete', 'scene_mismatch'], 'Unfinished comparison')
    require(result['input_signatures'] == signatures, 'Stored comparison inputs changed')
    require(result['branches'] == plan['branches'] and result['left_step'] == result['right_step'] == 2000,
            'Stored comparison branch identity differs')
    require(result['purpose'] == 'development' and result['task_scope'] == 'declared_task_subset'
            and result['planned_pairs'] == 60, 'Stored comparison scope differs')
    require(result['matched_pairs'] + result['mismatched_pairs'] == 60, 'Incomplete paired coverage')
    require(result['full16_success_rate'] is None, 'Three tasks cannot report a full16 success rate')
    if result['status'] == 'complete':
        require(result['matched_pairs'] == 60, 'Incomplete scene matching')
    else:
        for field in ['paired_success_rate_delta', 'paired_delta_bootstrap_95', 'gained', 'lost']:
            require(result[field] is None, 'Mismatched scenes cannot establish a paired improvement')


def scan(run, output):
    plan, tasks = declaration(run)
    signatures = ready_inputs(run, plan, tasks)
    state = dict(status='waiting_for_complete_reports', checked_utc=datetime.now(timezone.utc).isoformat(),
                 pid=os.getpid(), planned_pairs=60, task_scope='declared_task_subset',
                 gpu_control=False, new_policy_trials=0)
    if signatures is None:
        return state
    path = output / 'comparison.json'
    result = read(path) if path.is_file() else compare_run(run, plan, tasks, signatures)
    validate_result(result, signatures, plan)
    require(result['experiment_plan_sha256'] == sha(run / 'plan.json'), 'Compared experiment plan changed')
    if not path.exists():
        save(path, result)
    state.update(status=result['status'], evidence=str(path), input_signatures=signatures,
                 matched_pairs=result['matched_pairs'], mismatched_pairs=result['mismatched_pairs'],
                 paired_success_rate_delta=result['paired_success_rate_delta'])
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-hashes', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--max-hours', type=float, default=72)
    args = parser.parse_args()
    require(1 <= args.poll_seconds <= 60 and args.max_hours > 0, 'Invalid polling interval or duration')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU observer requires hidden CUDA devices')
    source = Path(__file__).resolve().parents[2]
    source_hashes = read(args.source_hashes)
    for name, digest in source_hashes.items():
        require(sha(source / name) == digest, 'Frozen observer source changed: ' + name)
    args.run = args.run.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'observer.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stopped = []
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, lambda number, _frame: stopped.append(number))
        started = time.monotonic()
        while True:
            try:
                state = scan(args.run, args.output)
            except Exception as error:
                state = dict(status='review_required', checked_utc=datetime.now(timezone.utc).isoformat(),
                             pid=os.getpid(), error=f'{type(error).__name__}: {error}',
                             gpu_control=False, new_policy_trials=0)
                save(args.output / 'status.json', state)
                raise
            if stopped:
                state['status'] = 'observer_stopped'
            elif time.monotonic() - started >= args.max_hours * 3600:
                state['status'] = 'observer_time_limit'
            save(args.output / 'status.json', state)
            if args.once or state['status'] != 'waiting_for_complete_reports':
                print(json.dumps(state), flush=True)
                return
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
