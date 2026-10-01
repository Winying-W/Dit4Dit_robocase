"""Build a conservative evidence report from replay diagnostic artifacts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT/'runs'


def read(relative):
    return json.loads((RUNS/relative).read_text())


def main():
    matrix = read('robocasa365_replay_warmup_20260923/summary.json')
    comparisons = read('robocasa365_replay_contract_20260923/fresh_repeat_comparison.json')
    contract = read('robocasa365_replay_contract_20260923/contract.json')
    table = []
    for ep in (0, 3, 5, 10, 11):
        baseline = next(r for r in matrix if r['episode'] == ep and r['variant'] == 'official')
        fixed = next(r for r in matrix if r['episode'] == ep and r['variant'] == 'warmup')
        rows = [json.loads(line) for line in (RUNS/f'robocasa365_replay_warmup_20260923/episode_{ep:06d}/warmup/steps.jsonl').read_text().splitlines()]
        table.append(dict(episode=ep, steps=fixed['steps'], official_fresh_success=baseline['success'],
                          startup_repaired_success=fixed['success'], first_error_before=baseline['first_step']['state_l2'],
                          first_error_after=fixed['first_step']['state_l2'],
                          max_abs_time_error_after=max(abs(r['time_error']) for r in rows if 'time_error' in r)))
    verification = RUNS/'robocasa365_replay_fixed_20260923/verification.json'
    final_check = json.loads(verification.read_text()) if verification.exists() else {'status': 'pending'}
    report = dict(status='verified_interface_fixes_with_residual_drift' if final_check['status']=='passed' else 'verification_pending',
                  task='StirVegetables / Composite seen / target human',
                  model_involved=False, policy_success_rate=None,
                  confirmed_fixes=['Restore one unrecorded zero-action collection step when raw simulation timestamps identify it',
                                   'Create a fresh environment per episode with explicit seed'],
                  gt_replay_comparison=table,
                  official_fresh_successes=sum(r['official_fresh_success'] for r in table),
                  startup_repaired_successes=sum(r['startup_repaired_success'] for r in table),
                  diagnostic_episodes=5,
                  fresh_environment_repeatability=comparisons,
                  recorded_state_predicate_diagnostics=contract,
                  final_entrypoint_verification=final_check,
                  residual_gt_action_failures=[r['episode'] for r in table if not r['startup_repaired_success']],
                  limitation='Residual open-loop contact-sensitive drift remains; no 100% replay, closed-loop policy, or statistically representative success claim.',
                  upstream_context='https://github.com/robocasa/robocasa/issues/209#issuecomment-4896068511')
    target = RUNS/'robocasa365_replay_fixed_20260923/diagnosis_report.json'
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','official_fresh_successes','startup_repaired_successes','residual_gt_action_failures')},indent=2))


if __name__ == '__main__':
    main()
