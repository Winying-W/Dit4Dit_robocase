"""Download an official target task set or declared subset without replacing data."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

from robocasa.utils.dataset_registry import TARGET_TASKS, COMPOSITE_TASK_DATASETS, ATOMIC_TASK_DATASETS


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--downloader', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--task-set', choices=['composite_seen','atomic_seen'], default='composite_seen')
    p.add_argument('--tasks', nargs='+', help='Explicit subset of the declared official target task set')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('mirror', a.downloader)
    mirror = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mirror)
    tasks = a.tasks or TARGET_TASKS[a.task_set]
    assert len(tasks) == len(set(tasks)) and set(tasks).issubset(TARGET_TASKS[a.task_set])
    category = 'atomic' if a.task_set == 'atomic_seen' else 'composite'
    registry = ATOMIC_TASK_DATASETS if category == 'atomic' else COMPOSITE_TASK_DATASETS
    def resolve(task):
        remote, size = mirror.resolve_task(task, 'target', category, 'master')
        official = registry[task]['target']['human_path']
        assert remote == official.removeprefix('v1.0/') + '/lerobot.tar', (task, remote, official)
        return dict(task=task, remote=remote, bytes=size, path=str(a.dest / Path(remote).parent),
                    horizon=registry[task]['horizon'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(resolve, tasks))
    write(a.output/'download_manifest.json', dict(source=mirror.MODELSCOPE_DATASET,
        revision='master', resolved_at=time.time(), tasks=manifest,
        total_bytes=sum(row['bytes'] for row in manifest)))
    for row in manifest:
        print(json.dumps(row), flush=True)
    if a.dry_run:
        return
    a.dest.mkdir(parents=True, exist_ok=True)
    print('STORAGE', shutil.disk_usage(a.dest), flush=True)
    def fetch(row):
        target = Path(row['path'])
        if target.exists():
            report = mirror.verify(target)
            if not report['ok']:
                raise RuntimeError(f'Existing incomplete dataset requires explicit repair: {report}')
            operation = 'verified_existing'
        else:
            tar = a.dest.parent / '.composite_seen_tars' / (row['task'] + '.tar')
            mirror.download(row['remote'], tar, row['bytes'], 'master')
            # The target must still be absent; never let the legacy extractor replace data.
            if target.exists():
                raise FileExistsError(target)
            mirror.extract(tar, target)
            report = mirror.verify(target)
            if not report['ok']:
                raise RuntimeError(report)
            tar.unlink()
            operation = 'downloaded'
        info = json.loads((target/'meta/info.json').read_text())
        assert report['n_episodes_data'] == info['total_episodes'], (row, info)
        hashes = {name:hashlib.sha256((target/'meta'/name).read_bytes()).hexdigest()
                  for name in ['info.json', 'modality.json', 'episodes.jsonl', 'tasks.jsonl']}
        return dict(**row, operation=operation, verification=report, metadata_sha256=hashes)
    completed = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(fetch, row) for row in manifest]):
            completed.append(future.result())
            write(a.output/'download_progress.json', dict(status='running', tasks=completed))
            print('DATASET_READY', completed[-1]['task'], flush=True)
    write(a.output/'datasets.json', dict(status='complete', task_set=a.task_set,
        scope='full_official_task_set' if set(tasks)==set(TARGET_TASKS[a.task_set]) else 'declared_task_subset',
        split='target', tasks=sorted(completed, key=lambda row: row['task'])))


if __name__ == '__main__':
    main()
