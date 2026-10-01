"""CPU review of completed policy trials; never launches training or inference.

Recount the declared benchmark and independently audit the saved action chain.
An incomplete evaluation has no aggregate success rate. Behavioral causes need
the selected videos/replays and controlled experiments, not a loss threshold.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from scripts.robocasa365.audit_saved_actions import audit_trial
from scripts.robocasa365.scene_protocol import protocol_spec, validate_protocol_records


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    d = 1 + z * z / count
    center = (p + z * z / (2 * count)) / d
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / d
    return [max(0., center - radius), min(1., center + radius)]


def verify_attempt(folder, task, identity, step, purpose_seeds):
    verification = read(folder / 'checkpoint_verification.json')
    evaluation = read(folder / 'evaluation.json')
    validate_protocol_records(identity, verification, evaluation)
    require(verification['policy_artifacts'] == identity['policy_artifacts'], 'Policy artifacts differ')
    require(verification['global_step'] == step, 'Unexpected checkpoint step')
    require(verification['task'] == task['task'] == evaluation['task'], 'Unexpected task')
    require(verification['checkpoint'] == identity['checkpoint'], 'Unexpected checkpoint path')
    require(verification['normalization_sha256'] == identity['policy_artifacts']['normalization']['sha256'],
            'Normalization identity mismatch')
    require(verification['architecture']['framework'] == 'DiT4DiT', 'Unexpected framework')
    require(verification['max_steps_override'] is None, 'Truncated smoke is not a benchmark')
    require(verification['demo_episodes'] is None, 'Demo initialization is not a fresh benchmark')
    require(set(verification['seeds']).issubset(purpose_seeds), 'Unexpected evaluation seeds')
    require(evaluation['initialization'] == 'fresh_gym_target' and evaluation['split'] == 'target',
            'Unexpected initial distribution')
    require(evaluation['horizon'] == evaluation['official_horizon'] == task['horizon'],
            'Incomplete official horizon')
    require(evaluation['execution_horizon'] == verification['execute_horizon'] == identity['execute_horizon'],
            'Execution horizon differs')
    return evaluation


def review(directory, manifest_path, expected_step, purpose, requested_tasks=None):
    directory, manifest_path = Path(directory).resolve(), Path(manifest_path).resolve()
    report_bytes = (directory / 'report.json').read_bytes()
    report = json.loads(report_bytes)
    manifest, identity = read(manifest_path), read(directory / 'identity.json')
    require(purpose in ('development', 'final'), 'Unexpected purpose')
    seeds = manifest['evaluation']['selection_seeds' if purpose == 'development' else 'seeds']
    require(set(manifest['evaluation']['selection_seeds']).isdisjoint(manifest['evaluation']['seeds']),
            'Development and final seeds overlap')
    names = [row['task'] for row in manifest['tasks']]
    tasks = {row['task']: row for row in manifest['tasks']}
    require(len(names) == len(tasks) and len(set(seeds)) == len(seeds), 'Duplicate task or seed')
    require(len(names) == {'atomic_seen': 18, 'composite_seen': 16}[manifest['task_set']],
            'Expected complete official task set')
    require(manifest['task_scope']=='full_official_task_set','Expected the unchanged full training manifest')
    task_scope='full_official_task_set'
    if requested_tasks is not None:
        require(purpose=='development','Task selection is restricted to development review')
        require(bool(requested_tasks) and len(set(requested_tasks))==len(requested_tasks)
                and set(requested_tasks)<=set(names),'Invalid explicit development task list')
        selected=[name for name in names if name in set(requested_tasks)]
        if len(selected)<len(names):task_scope='declared_task_subset'
        names=selected;tasks={name:tasks[name] for name in names}
    require(identity['task_scope']==report['task_scope']==task_scope,'Subset is not a full benchmark')
    require(manifest['task_set'] == identity['task_set'] == report['task_set'], 'Task set mismatch')
    require(report['purpose'] == identity['purpose'] == purpose, 'Development/final mismatch')
    require(report['planned_tasks'] == identity['tasks'] == names, 'Task list mismatch')
    require(report['seeds'] == identity['seeds'] == seeds, 'Seed list mismatch')
    require(identity['manifest_sha256'] == digest(manifest_path), 'Manifest identity mismatch')
    normal = identity['policy_artifacts']['normalization']
    require(digest(normal['path']) == normal['sha256'], 'Normalization file changed')
    statistics = read(normal['path'])
    rows, attempts, trial_keys = [], {}, set()
    for task_row in report['tasks']:
        name = task_row['task']
        require(name in tasks and name not in {row['task'] for row in rows}, 'Duplicate or unknown task')
        audited, failures, successes, endings = [], [], [], Counter()
        for trial in task_row['trials']:
            key = (name, trial['seed'])
            require(trial['seed'] in seeds and key not in trial_keys, 'Duplicate or unexpected trial seed')
            trial_keys.add(key)
            artifact = Path(trial['artifact']).resolve()
            require(artifact.is_relative_to(directory / name), 'Trial belongs to another task or benchmark')
            saved = read(artifact / 'result.json')
            require({k: v for k, v in trial.items() if k != 'artifact'} == saved, 'Trial summary changed')
            require(type(saved['success']) is bool, 'Success must be an explicit boolean')
            require(saved['demo_episode'] is None and saved['action_chain_runtime_passed'] is True,
                    'Demo trial or failed runtime action audit')
            require(saved['controller_input_type'] == 'delta', 'Unexpected action type')
            require(0 < saved['steps'] <= tasks[name]['horizon'], 'Invalid episode length')
            require(saved['success'] or saved['steps'] == tasks[name]['horizon'] or
                    saved['terminated'] or saved['truncated'], 'Unfinished trial is not a policy failure')
            folder = artifact.parent
            if folder not in attempts:
                attempts[folder] = verify_attempt(folder, tasks[name], identity, expected_step, set(seeds))
            evaluation = attempts[folder]
            require(saved in evaluation['episodes'], 'Trial absent from simulator report')
            command_audit = audit_trial(artifact, statistics, evaluation['execution_horizon'])
            require(command_audit['passed'], 'Saved action chain mismatch')
            audited.append(command_audit)
            case = dict(seed=saved['seed'], artifact=str(artifact), video=str(artifact / 'rollout.mp4'),
                        steps=saved['steps'], success=saved['success'], action_audit=command_audit)
            if saved['success']:
                successes.append(case)
            else:
                failures.append(case)
                endings['official_horizon_exhausted' if saved['steps'] == tasks[name]['horizon']
                        else 'environment_ended_early'] += 1
        present = {trial['seed'] for trial in task_row['trials']}
        complete = present == set(seeds)
        require(task_row['complete'] == complete, 'Task completeness mismatch')
        require(task_row['planned_trials'] == len(seeds) and task_row['completed_trials'] == len(present),
                'Task denominator mismatch')
        require(task_row['successes'] == len(successes), 'Task numerator mismatch')
        require(task_row['missing_seeds'] == [seed for seed in seeds if seed not in present], 'Missing seeds mismatch')
        expected_rate = len(successes) / len(seeds) if complete else None
        require(task_row['success_rate'] == expected_rate, 'Task success rate mismatch')
        steps = sum(item['steps'] for item in audited)
        rows.append(dict(task=name, complete=complete, completed_trials=len(present), planned_trials=len(seeds),
                         successes=len(successes), success_rate=expected_rate,
                         wilson_95=wilson(len(successes), len(seeds)) if complete else None,
                         audited_steps=steps, action_chain_passed=bool(audited),
                         arm_clipped_fraction=sum(x['steps'] * x['executed_arm_clipped_component_fraction']
                                                  for x in audited) / steps if steps else None,
                         gripper_closed_fraction=sum(x['steps'] * x['executed_gripper_closed_fraction']
                                                     for x in audited) / steps if steps else None,
                         base_mode_fraction=sum(x['steps'] * x['executed_base_mode_fraction']
                                                for x in audited) / steps if steps else None,
                         failure_endings=dict(endings),
                         selected_failures=sorted(failures, key=lambda x: x['seed'])[:3],
                         selected_successes=sorted(successes, key=lambda x: x['seed'])[:1],
                         infrastructure_attempts=[x for x in task_row['attempts'] if x['status'] != 'complete']))
    complete = len(rows) == len(names) and all(row['complete'] for row in rows)
    successes = sum(row['successes'] for row in rows)
    completed = sum(row['completed_trials'] for row in rows)
    require(report['status'] == ('complete' if complete else 'incomplete'), 'Benchmark completeness mismatch')
    require(report['planned_trials'] == len(names) * len(seeds), 'Planned denominator mismatch')
    require(report['completed_trials'] == completed and report['successes'] == successes, 'Aggregate counts mismatch')
    rate = successes / completed if complete else None
    require(report['pooled_success_rate'] == rate, 'Pooled success rate mismatch')
    macro = sum(row['success_rate'] for row in rows) / len(names) if complete else None
    require((macro is None and report['equal_task_macro_success_rate'] is None) or
            (macro is not None and math.isclose(macro, report['equal_task_macro_success_rate'], abs_tol=1e-12)),
            'Macro success rate mismatch')
    if not complete:
        next_action = 'Wait for missing trials or repair recorded infrastructure failures; no full-benchmark claim.'
    elif purpose == 'final':
        next_action = ('Archive the final result. If this result informs further tuning, use a new held-out '
                       'scenario set for the next version; preserve this version and its failures.')
    elif successes == 0:
        next_action = ('Prioritize failure video/state and train/eval observation checks plus a small training-scene '
                       'overfit diagnostic. The saved action conversion passed; zero success does not identify '
                       'insufficient training or a particular model module as the cause.')
    else:
        next_action = ('Inspect per-task failure cases and compare the next development checkpoint on the same '
                       'seeds. Verify scene identity before paired conclusions; choose data/continuation/video '
                       'ablation using development evidence, not final seeds.')
    return dict(status='complete' if complete else 'incomplete', reviewed_utc=datetime.now(timezone.utc).isoformat(),
                checkpoint_step=expected_step, purpose=purpose, task_set=manifest['task_set'],task_scope=task_scope,
                scene_protocol=identity.get('scene_protocol', protocol_spec()),
                report=str(directory / 'report.json'), report_sha256=hashlib.sha256(report_bytes).hexdigest(),
                identity_sha256=digest(directory / 'identity.json'), manifest_sha256=digest(manifest_path),
                normalization_sha256=normal['sha256'], planned_trials=len(names) * len(seeds),
                completed_trials=completed, successes=successes, pooled_success_rate=rate,
                equal_task_macro_success_rate=macro, pooled_wilson_95=wilson(successes, completed) if complete else None,
                seeds=seeds, tasks=rows, saved_action_chain_passed=bool(completed), recommended_next_action=next_action,
                behavioral_root_cause='not established by this artifact review',
                scope='CPU recount and independent saved-action audit only. No new policy trials, semantic '
                      'milestone replay, camera-equivalence proof, training update or GPU checkpoint restore. '
                      'Clipping/gripper/base fractions are diagnostic indicators, not causal labels.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--purpose', choices=['development', 'final'], required=True)
    parser.add_argument('--tasks',nargs='+',help='Explicit development subset; keep the full training manifest unchanged')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.evaluation, args.manifest, args.step, args.purpose,args.tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ['status', 'completed_trials', 'successes', 'pooled_success_rate']}))


if __name__ == '__main__':
    main()
