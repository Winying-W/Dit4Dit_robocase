"""Compare complete development checkpoints only after checking every saved scene.

Any mismatched scene suppresses the full-benchmark paired estimate. No trials
are silently dropped. This command never runs a policy or changes training.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from scripts.robocasa365.review_evaluation import review, require
from scripts.robocasa365.scene_identity import compare_scene_xml
from scripts.robocasa365.scene_protocol import protocol_spec


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_initial_scenes(left, right):
    left_xml, right_xml = left / 'initial_model.xml', right / 'initial_model.xml'
    xml = compare_scene_xml(left_xml.read_text(), right_xml.read_text())
    states = []
    for folder in [left, right]:
        with np.load(folder / 'trajectory.npz', allow_pickle=False) as arrays:
            states.append(arrays['initial_state'].copy())
    finite = all(np.issubdtype(x.dtype, np.floating) and x.ndim == 1 and x.size > 0
                 and np.isfinite(x).all() for x in states)
    same_shape = states[0].shape == states[1].shape
    error = float(np.max(np.abs(states[0] - states[1]))) if finite and same_shape else None
    return dict(comparable=bool(xml['equivalent'] and error is not None and error <= 1e-10),
                xml=xml, initial_state_finite=bool(finite), initial_state_same_shape=same_shape,
                initial_state_max_error=error, initial_state_tolerance=1e-10,
                initial_xml_sha256=[sha(left_xml), sha(right_xml)],
                initial_state_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in states])


def paired_summary(rows, tasks, seeds):
    expected = {(task, seed) for task in tasks for seed in seeds}
    keys = [(row['task'], row['seed']) for row in rows]
    require(len(tasks) == len(set(tasks)) and len(seeds) == len(set(seeds)), 'Duplicate declaration')
    require(expected and len(keys) == len(set(keys)) and set(keys) == expected,
            'Missing, duplicate, or unexpected scene pairs')
    require(all(type(row['left_success']) is bool and type(row['right_success']) is bool
                and type(row['scene']['comparable']) is bool for row in rows), 'Non-boolean outcome')
    by_key = dict(zip(keys, rows))
    matched = sum(row['scene']['comparable'] for row in rows)
    complete = matched == len(expected)
    result = dict(status='complete' if complete else 'scene_mismatch', planned_pairs=len(expected),
                  matched_pairs=matched, mismatched_pairs=len(expected) - matched,
                  gained=None, lost=None, paired_success_rate_delta=None,
                  paired_delta_bootstrap_95=None, paired_task_deltas=None)
    if not complete:
        result['reason'] = 'All declared pairs are required; no subset or paired improvement estimate is reported.'
        return result
    changes = np.asarray([[int(by_key[task, seed]['right_success']) - int(by_key[task, seed]['left_success'])
                           for seed in seeds] for task in tasks], dtype=np.float64)
    # Resample each seed across all tasks together, preserving dependence
    # induced by using the same seed set for multiple task environments.
    rng = np.random.default_rng(20260927)
    indices = rng.integers(len(seeds), size=(10000, len(seeds)))
    bootstrap = changes.mean(axis=0)[indices].mean(axis=1)
    result.update(gained=int((changes > 0).sum()), lost=int((changes < 0).sum()),
                  paired_success_rate_delta=float(changes.mean()),
                  paired_delta_bootstrap_95=np.quantile(bootstrap, [.025, .975]).tolist(),
                  paired_task_deltas=[dict(task=task, delta=float(changes[index].mean()))
                                      for index, task in enumerate(tasks)],
                  uncertainty=dict(method='Percentile paired bootstrap of seed blocks across all fixed tasks',
                      samples=10000, random_seed=20260927, independent_seed_blocks=len(seeds),
                      scope='Exploratory development comparison on these fixed tasks and scenes. '
                            'Small seed counts and degenerate intervals do not establish equivalence or generalization. '
                            'Policy sampling randomness is not claimed to match across checkpoints.'))
    return result


def compare_data_configs(left, right, allow_statistics_path_relocation=False):
    artifacts = [identity['policy_artifacts'] for identity in [left, right]]
    if artifacts[0]['data_config']['sha256'] == artifacts[1]['data_config']['sha256']:
        return dict(equivalent=True, identical_bytes=True, statistics_path_relocation=False)
    require(allow_statistics_path_relocation, 'Policy input/base artifact differs: data_config')
    configs, statistics = [], []
    for item in artifacts:
        descriptor = item['data_config']
        path = Path(descriptor['path'])
        require(sha(path) == descriptor['sha256'], 'Data config changed')
        config = yaml.safe_load(path.read_text())
        require(isinstance(config, dict) and isinstance(config.get('statistics_path'), str),
                'Expected an explicit normalization path in data config')
        normal_path = Path(config.pop('statistics_path')).resolve()
        require(normal_path == Path(item['normalization']['path']).resolve(),
                'Data config points to a different normalization artifact')
        require(sha(normal_path) == item['normalization']['sha256'], 'Data config statistics changed')
        configs.append(config)
        statistics.append(item['normalization']['sha256'])
    require(statistics[0] == statistics[1], 'Data config normalization differs')
    require(configs[0] == configs[1], 'Data config differs beyond the statistics path')
    return dict(equivalent=True, identical_bytes=False, statistics_path_relocation=True,
                config_sha256=[item['data_config']['sha256'] for item in artifacts],
                normalization_sha256=statistics[0],
                rule='Only statistics_path may relocate to the separately verified identical normalization bytes.')


def compare(left, right, manifest_path, left_step, right_step, *,
            requested_tasks=None, same_step_branches=False):
    left, right, manifest_path = map(lambda p: Path(p).resolve(), [left, right, manifest_path])
    if same_step_branches:
        require(0 < left_step == right_step and left != right,
                'Branch comparison requires equal positive steps and distinct evaluations')
    else:
        require(0 < left_step < right_step, 'Expected strictly increasing positive checkpoint steps')
    manifest = read(manifest_path)
    tasks = [task['task'] for task in manifest['tasks']]
    scope = 'full_official_task_set'
    if requested_tasks is not None:
        require(bool(requested_tasks) and len(set(requested_tasks)) == len(requested_tasks)
                and set(requested_tasks) <= set(tasks), 'Invalid explicit development task list')
        selected = [name for name in tasks if name in set(requested_tasks)]
        if len(selected) < len(tasks):
            scope = 'declared_task_subset'
        tasks = selected
    seeds = manifest['evaluation']['selection_seeds']
    require(len(seeds) >= 20, 'At least twenty development seeds per task are required')
    require(set(seeds).isdisjoint(manifest['evaluation']['seeds']), 'Final seed contamination')
    audits, reports, identities = [], [], []
    for directory, step in [(left, left_step), (right, right_step)]:
        identity_bytes = (directory / 'identity.json').read_bytes()
        identity = json.loads(identity_bytes)
        require(identity['purpose'] == 'development', 'Final evaluation cannot select a training strategy')
        audit = review(directory, manifest_path, step, 'development', requested_tasks)
        require(audit['status'] == 'complete' and audit['saved_action_chain_passed'],
                'Complete independently audited development evaluations are required')
        require(audit['report_sha256'] == sha(directory / 'report.json'), 'Report changed during audit')
        require(audit['identity_sha256'] == hashlib.sha256(identity_bytes).hexdigest(),
                'Identity changed during audit')
        audits.append(audit)
        reports.append(read(directory / 'report.json'))
        identities.append(identity)
    for field in ['tasks', 'seeds', 'manifest_sha256', 'execute_horizon', 'task_set', 'task_scope']:
        require(identities[0][field] == identities[1][field], f'Protocol differs: {field}')
    require(identities[0].get('scene_protocol', protocol_spec()) ==
            identities[1].get('scene_protocol', protocol_spec()), 'Scene protocol differs')
    for field in ['normalization', 'base_weights', 'base_config', 'base_statistics']:
        require(identities[0]['policy_artifacts'][field]['sha256'] ==
                identities[1]['policy_artifacts'][field]['sha256'], f'Policy input/base artifact differs: {field}')
    data_configs = compare_data_configs(*identities, allow_statistics_path_relocation=same_step_branches)
    indexes = [{(task['task'], trial['seed']): trial for task in report['tasks'] for trial in task['trials']}
               for report in reports]
    rows = []
    for task in tasks:
        for seed in seeds:
            trials = [index[task, seed] for index in indexes]
            folders = [Path(trial['artifact']) for trial in trials]
            rows.append(dict(task=task, seed=seed, left_success=trials[0]['success'],
                             right_success=trials[1]['success'], artifacts=list(map(str, folders)),
                             scene=compare_initial_scenes(*folders)))
    for directory, audit in zip([left, right], audits):
        require(audit['report_sha256'] == sha(directory / 'report.json') and
                audit['identity_sha256'] == sha(directory / 'identity.json'), 'Evaluation changed during comparison')
    return dict(**paired_summary(rows, tasks, seeds), verified_utc=datetime.now(timezone.utc).isoformat(),
                left_step=left_step, right_step=right_step, purpose='development', task_set=manifest['task_set'],
                task_scope=scope, compared_tasks=tasks, same_step_branches=same_step_branches,
                data_config_equivalence=data_configs,
                source_reports=[dict(path=str(directory / 'report.json'), sha256=audit['report_sha256'])
                                for directory, audit in zip([left, right], audits)],
                independent_reviews=audits, scene_pairs=rows,
                comparison_source_sha256=sha(Path(__file__)),
                scope='Full declared development set with fresh saved-action re-audits and per-trial initial scene/state comparison. '
                      'No policy trials or training added; no final benchmark or video-unfreezing A/B claim.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['left', 'right', 'manifest', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--left-step', type=int, required=True)
    parser.add_argument('--right-step', type=int, required=True)
    parser.add_argument('--tasks', nargs='+', help='Explicit development subset of the full training manifest')
    parser.add_argument('--same-step-branches', action='store_true',
                        help='Compare distinct branches with equal local steps; does not establish matched training')
    args = parser.parse_args()
    result = compare(args.left, args.right, args.manifest, args.left_step, args.right_step,
                     requested_tasks=args.tasks, same_step_branches=args.same_step_branches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ['status', 'planned_pairs', 'matched_pairs',
          'mismatched_pairs', 'gained', 'lost', 'paired_success_rate_delta']}), flush=True)
