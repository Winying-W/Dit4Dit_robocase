"""Observe an existing cloud run without submitting, stopping, or restarting it.

Training, checkpoint handoff, and final evaluation expose different status fields.
Keep live updates, saved updates, development trials, and final trials separate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def summarize_report(path):
    content = path.read_bytes()
    report = json.loads(content)
    fields = ['status', 'task_set', 'task_scope', 'purpose', 'planned_trials',
              'completed_trials', 'successes', 'pooled_success_rate',
              'equal_task_macro_success_rate', 'pooled_wilson_95']
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(content).hexdigest(),
                **{name: report.get(name) for name in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--training-output', type=Path, required=True)
    parser.add_argument('--update-progress', action='store_true',
                        help='Update local experiment/goal progress records; never goal lifecycle status')
    parser.add_argument('--classification', choices=['progress', 'verified_wait'], default='verified_wait')
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.training_output.resolve()
    root = Path(__file__).resolve().parents[2]
    plan = read_json(run / 'plan.json')
    environment = os.environ.copy()
    environment['PATH'] = str(Path.home() / '.volc/bin') + os.pathsep + environment['PATH']
    # Observation failures propagate; they never mean that the GPU job stopped.
    response = subprocess.run(
        ['volc', 'ml_task', 'get', '--id', args.task_id, '--output', 'json',
         '--format', 'Id,Status,Elapsed,ExitCode'],
        env=environment, capture_output=True, text=True, check=True, timeout=30)
    records = json.loads(response.stdout)
    assert len(records) == 1 and records[0]['Id'] == args.task_id
    cloud = records[0]
    now = datetime.now(timezone.utc)
    status_path = run / 'train_live/status.json'
    training = read_json(status_path) or {}
    checkpoint = read_json(output / 'latest.json') or {}
    live_step = training.get('global_step', training.get('global_steps'))
    saved_step = checkpoint.get('global_step')
    stage_path = run / 'stage.txt'
    stage = stage_path.read_text().strip() if stage_path.is_file() else None
    numerical_fields = ['action_loss', 'gradient_norm', 'seconds']
    available = {name: training[name] for name in numerical_fields if name in training}
    health = dict(checked_fields=available,
                  nonfinite_fields=[name for name, value in available.items() if not math.isfinite(value)],
                  absent_fields=[name for name in numerical_fields if name not in training])
    development = [summarize_report(path) for path in sorted(run.glob('development_*/report.json'))]
    final = [summarize_report(path) for path in sorted(run.glob('evaluation_*/report.json'))
             if path.parent.name != 'evaluation_smoke']
    expected_final = [report for report in final
                      if report['purpose'] == 'final' and report['task_set'] == plan['task_set']
                      and report['task_scope'] == 'full_official_task_set'
                      and report['planned_trials'] == plan['planned_final_trials']]
    assert len(expected_final) <= 1, 'Review multiple final reports explicitly rather than combine benchmarks'
    final_completed = expected_final[0]['completed_trials'] if expected_final else 0
    observation = dict(verified_utc=now.isoformat(), cloud=cloud, stage=stage,
                       live_training=training, live_step=live_step,
                       last_saved_checkpoint=checkpoint, saved_step=saved_step,
                       status_age_seconds=now.timestamp()-status_path.stat().st_mtime if status_path.is_file() else None,
                       numerical_health=health, development_reports=development, final_reports=final,
                       final_policy_trials_completed=final_completed,
                       source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       scope='Read-only cloud observation. Local progress records are optional. '
                             'Missing loss fields during checkpoint handoff/completion are not NaNs. '
                             'Development and final trials remain separate; no success rate is inferred from loss.')
    evidence = run / 'observations' / ('status_' + now.strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    save(evidence, observation)

    if args.update_progress:
        requirement = {
            'atomic_seen': 'Expanded4-GPU Atomic18 diagnostic requested subsequently',
            'composite_seen': 'Full16 Composite training and final closed-loop evaluation',
        }[plan['task_set']]
        if cloud['Status'] == 'Running':
            state = ('development_evaluation_running' if (stage or '').startswith('development') else
                     'final_evaluation_running' if (stage or '').startswith('final_evaluation') else
                     'training_running')
        else:
            state = cloud['Status']
        goal_path = root / 'runs/robocasa365_goal_progress_20260924.json'
        goal = read_json(goal_path)
        if goal is not None:
            goal['previous_turn_classification'] = goal.get('current_turn_classification')
            goal.update(verified_utc=now.isoformat(), current_turn_classification=args.classification)
            for item in goal['requirements']:
                if item['requirement'] == requirement:
                    item.update(status=state, verified_updates=live_step, verified_saved_updates=saved_step,
                                monitor_evidence=str(evidence.relative_to(root)),
                                final_trials_completed=final_completed)
            save(goal_path, goal)
        progress_path = root / 'runs/robocasa365_multitask_20260924/progress.json'
        progress = read_json(progress_path)
        if progress is not None:
            progress['last_progress_utc'] = now.isoformat()
            key = 'atomic_diagnostic' if plan['task_set'] == 'atomic_seen' else 'composite_training'
            progress.setdefault(key, {}).update(status=cloud['Status'], completed_updates=live_step,
                durable_checkpoint_updates=saved_step, completed_policy_trials=final_completed,
                development_trial_counts={Path(row['path']).parent.name: row['completed_trials'] for row in development},
                monitor_evidence=str(evidence.relative_to(root)))
            save(progress_path, progress)
    print(json.dumps(dict(cloud=cloud, stage=stage, live_step=live_step, saved_step=saved_step,
                          numerical_health=health, development_reports=development,
                          final_reports=final, final_policy_trials_completed=final_completed,
                          observation=str(evidence))), flush=True)


if __name__ == '__main__':
    main()
