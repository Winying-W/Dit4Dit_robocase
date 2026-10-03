"""Freeze official target50 tasks for evaluating the human300-only policy."""
import argparse
import ast
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

from scripts.robocasa365.prepare_multitask import save


def read_registry(path):
    names = {'ATOMIC_TASK_DATASETS', 'COMPOSITE_TASK_DATASETS', 'TARGET_TASKS'}
    records = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                records[name] = eval(compile(ast.Expression(node.value), str(path), 'eval'),
                                     {'__builtins__': {}, 'dict': dict, 'OrderedDict': OrderedDict})
    assert set(records) == names
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--training-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Evaluation manifest is immutable')
    registry = read_registry(args.registry)
    tasks = {**registry['ATOMIC_TASK_DATASETS'], **registry['COMPOSITE_TASK_DATASETS']}
    groups = registry['TARGET_TASKS']
    assert {k:len(v) for k,v in groups.items()} == dict(atomic_seen=18, composite_seen=16, composite_unseen=16)
    source = json.loads((args.root/'download_manifest.json').read_text())
    rows = []
    for group, names in groups.items():
        for name in names:
            archives = [r for r in source['files'] if r['Path'].startswith('target/')
                        and r['Path'].endswith('/lerobot.tar') and Path(r['Path']).parts[2] == name]
            assert len(archives) == 1, name
            archive = archives[0]
            dataset = args.root/'extracted'/Path(archive['Path']).parent
            marker = json.loads((dataset/'EXTRACTION_VERIFIED.json').read_text())
            assert marker['archive_sha256'] == archive['Sha256']
            metadata = json.loads((dataset/'extras/dataset_meta.json').read_text())['env_args']
            assert metadata['env_name'] == name and metadata['env_kwargs']['obj_instance_split'] == 'target'
            assert metadata['env_kwargs']['robots'] == 'PandaOmron'
            rows.append(dict(task=name, group=group, path=str(dataset.resolve()), horizon=tasks[name]['horizon'],
                             archive_sha256=archive['Sha256']))
    trained = json.loads(args.training_manifest.read_text())
    assert trained['task_set'] == 'pretrain_human300' and len(trained['tasks']) == 300
    result = dict(task_set='target50', split='target', task_scope='full_official_task_set', tasks=rows,
                  registry_sha256=hashlib.sha256(args.registry.read_bytes()).hexdigest(),
                  training_manifest_sha256=hashlib.sha256(args.training_manifest.read_bytes()).hexdigest(),
                  statistics_source='unchanged human300 checkpoint normalization',
                  evaluation=dict(seeds=list(range(100,150)), selection_seeds=list(range(10000,10020))),
                  protocol_note='Frozen local evaluation seeds; pi0.5 seed equivalence has not been established. No target finetuning.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, result)
    print('TARGET50_MANIFEST_READY', len(rows), flush=True)


if __name__ == '__main__':
    main()
