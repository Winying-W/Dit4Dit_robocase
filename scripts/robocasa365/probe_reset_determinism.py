"""Isolated fresh-process test of counter reset-region ordering.

The optional patch exists only in a worker process. It never writes installed
RoboCasa files or changes an existing evaluation process. No policy is loaded.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


from scripts.robocasa365.scene_protocol import COUNTER_SHA256, OFFICIAL, STABLE, apply_scene_protocol


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def configure_counter(variant):
    receipt = apply_scene_protocol(STABLE if variant == 'stable' else OFFICIAL)
    if receipt['source_sha256'] != COUNTER_SHA256:
        raise RuntimeError('RoboCasa counter source changed; review before applying experiment')
    return dict(path=receipt['source_path'],sha256=receipt['source_sha256'],variant=variant,
                function_source_sha256=receipt['effective_function_sha256'],
                installed_file_unchanged=receipt['installed_file_unchanged'],protocol=receipt['protocol'])


def worker(args):
    import gymnasium as gym
    import numpy as np
    import robocasa  # Register Gym environments.
    from scripts.robocasa365.eval_protocol import observation_arrays

    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status='running', task=args.task, seed=args.seed, variant=args.variant,
                  pid=os.getpid(), python_hash_seed=os.environ.get('PYTHONHASHSEED'),
                  source_sha256=sha(__file__), policy_trials_added=0,
                  rendering='disabled identically in both variants; images not compared')
    save(args.output / 'result.json', report)
    env = None
    try:
        report['counter'] = configure_counter(args.variant)
        # Keep official Gym/reset behavior but skip all GL contexts in this CPU
        # experiment. The wrapper supplies its documented black placeholders;
        # those images are deliberately excluded from repeatability evidence.
        import robocasa.utils.env_utils as env_utils
        original_make = env_utils.robosuite.make

        def cpu_make(*positional, **kwargs):
            kwargs.update(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
            return original_make(*positional, **kwargs)

        env_utils.robosuite.make = cpu_make
        np.random.seed(args.seed)
        random.seed(args.seed)
        env = gym.make('robocasa/' + args.task, split='target', seed=args.seed,
                       camera_widths=256, camera_heights=256, enable_render=False)
        obs, _ = env.reset(seed=args.seed)
        core = env.unwrapped.env
        assert not core.has_offscreen_renderer and not core.has_renderer and not core.use_camera_obs
        (args.output / 'model.xml').write_text(core.sim.model.get_xml())
        save(args.output / 'episode_meta.json', core.get_ep_meta())
        np.savez_compressed(args.output / 'initial_state.npz',
                            physical=core.sim.get_state().flatten().copy(),
                            policy_state=observation_arrays(obs)['state'])
        report['images_compared'] = False
        report['counter']['installed_file_unchanged'] = sha(report['counter']['path']) == COUNTER_SHA256
        assert report['counter']['installed_file_unchanged']
        report.update(status='complete', seconds=time.monotonic() - started)
        save(args.output / 'result.json', report)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', seconds=time.monotonic() - started)
        save(args.output / 'result.json', report)
        raise
    finally:
        if env is not None:
            env.close()


def run_experiment(args):
    import numpy as np
    from scripts.robocasa365.scene_identity import compare_scene_xml

    assert args.repeats >= 2 and 1 <= args.workers <= 2
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [(case.rsplit(':', 1)[0], int(case.rsplit(':', 1)[1])) for case in args.cases]
    identity = dict(created_utc=datetime.now(timezone.utc).isoformat(), cases=cases,
                    repeats=args.repeats, workers=args.workers, source_sha256=sha(__file__),
                    counter_source_sha256=COUNTER_SHA256, fixed_python_hash_seed='0',
                    scope='Fresh-process reset repeatability with rendering disabled in both variants. '
                          'Physical state, policy state, XML and metadata are compared; images are not. '
                          'No policy trials, training or GT replay. '
                          'The stable variant changes only process-local counter candidate deduplication. '
                          'Repeatability on selected cases does not prove all tasks or seeds deterministic.')
    save(args.output / 'identity.json', identity)
    jobs = []
    for task, seed in cases:
        for variant in ['original', 'stable']:
            for repeat in range(args.repeats):
                directory = args.output / task / f'seed_{seed}' / variant / f'repeat_{repeat}'
                jobs.append(dict(task=task, seed=seed, variant=variant, repeat=repeat, directory=str(directory)))

    def run(job):
        directory = Path(job['directory'])
        directory.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(PYTHONHASHSEED='0', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                           OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', LIBGL_ALWAYS_SOFTWARE='1')
        command = [sys.executable, '-u', '-m', 'scripts.robocasa365.probe_reset_determinism', '--worker',
                   '--task', job['task'], '--seed', str(job['seed']), '--variant', job['variant'],
                   '--output', str(directory)]
        with directory.with_suffix('.log').open('w') as stream:
            result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=240)
        return dict(**job, returncode=result.returncode)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as exc:
                row = dict(**futures[future], error=f'{type(exc).__name__}: {exc}', returncode=None)
            results.append(row)
            save(args.output / 'progress.json', dict(completed=len(results), planned=len(jobs), jobs=results))
            print('RESET_PROBE', len(results), len(jobs), row, flush=True)
    comparisons = []
    for task, seed in cases:
        for variant in ['original', 'stable']:
            group = sorted([r for r in results if r['task'] == task and r['seed'] == seed and r['variant'] == variant],
                           key=lambda r: r['repeat'])
            if any(r['returncode'] != 0 for r in group):
                comparisons.append(dict(task=task, seed=seed, variant=variant, status='execution_failed'))
                continue
            reference = Path(group[0]['directory'])
            with np.load(reference / 'initial_state.npz', allow_pickle=False) as content:
                states = {key: content[key].copy() for key in content.files}
            checks = []
            for row in group[1:]:
                directory = Path(row['directory'])
                check = dict(repeat=row['repeat'], xml=compare_scene_xml(
                    (reference / 'model.xml').read_text(), (directory / 'model.xml').read_text()),
                    metadata_equal=json.loads((reference / 'episode_meta.json').read_text()) ==
                                   json.loads((directory / 'episode_meta.json').read_text()), states={})
                with np.load(directory / 'initial_state.npz', allow_pickle=False) as generated:
                    for key, original in states.items():
                        value = generated[key]
                        same_shape = original.shape == value.shape
                        check['states'][key] = dict(shape_equal=same_shape, exact=bool(np.array_equal(original, value)),
                            max_error=float(np.max(np.abs(original - value))) if same_shape else None)
                check['all_exact'] = (check['xml']['equivalent'] and check['metadata_equal'] and
                                      all(v['exact'] for v in check['states'].values()))
                checks.append(check)
            comparisons.append(dict(task=task, seed=seed, variant=variant, status='complete',
                                    all_repeats_exact=all(c['all_exact'] for c in checks), checks=checks))
    report = dict(status='complete' if all(r['returncode'] == 0 for r in results) else 'execution_failed',
                  completed_utc=datetime.now(timezone.utc).isoformat(), comparisons=comparisons,
                  resets=len(results), policy_trials_added=0, scope=identity['scope'])
    save(args.output / 'report.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--task')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--variant', choices=['original', 'stable'])
    parser.add_argument('--cases', nargs='+', default=['NavigateKitchen:110', 'OpenStandMixerHead:107',
                        'PickPlaceSinkToCounter:101', 'TurnOnElectricKettle:106', 'TurnOnSinkFaucet:101'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.worker:
        assert args.task and args.seed is not None and args.variant
        worker(args)
    else:
        run_experiment(args)


if __name__ == '__main__':
    main()
