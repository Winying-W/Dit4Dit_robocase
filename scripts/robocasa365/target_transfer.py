"""Keep target-only evaluation separate from human300 training and statistics."""
import hashlib
import json
from pathlib import Path


def target_task_set(task, registry):
    """Resolve all three official target groups and reject ambiguous membership."""
    groups = ('atomic_seen', 'composite_seen', 'composite_unseen')
    matches = [group for group in groups if task in registry.get(group, ())]
    if len(matches) != 1:
        raise ValueError(f'Expected one official target group for {task}, got {matches}')
    return matches[0]


def validate_transfer_manifest(manifest, training_manifest, training_split):
    training_manifest = Path(training_manifest)
    digest = hashlib.sha256(training_manifest.read_bytes()).hexdigest()
    trained = json.loads(training_manifest.read_text())
    if trained['task_set'] != 'pretrain_human300':
        raise ValueError('Transfer evaluation requires human300 training')
    if manifest.get('training_manifest_sha256') != digest or training_split['manifest_sha256'] != digest:
        raise ValueError('Transfer manifest is bound to a different training dataset')
    if manifest.get('task_set') != 'target50' or manifest.get('split') != 'target':
        raise ValueError('Transfer evaluation requires the declared target50 manifest')
    tasks = manifest['tasks']
    if len(tasks) != 50 or len({row['task'] for row in tasks}) != 50:
        raise ValueError('Expected exactly 50 unique target tasks')
    if manifest.get('statistics_source') != 'unchanged human300 checkpoint normalization':
        raise ValueError('Target statistics must not replace training normalization')


def transfer_probe_split(adapter, dataset, task):
    """One target demo checks the action interface only; never supplies training data."""
    metadata = json.loads((Path(dataset)/'extras/dataset_meta.json').read_text())['env_args']
    if metadata['env_name'] != task or metadata['env_kwargs']['obj_instance_split'] != 'target':
        raise ValueError('Target task/dataset metadata mismatch')
    episode = int(adapter.dataset.all_steps[0][0])
    length = int(adapter.dataset.trajectory_lengths[adapter.dataset.get_trajectory_index(episode)])
    candidates = [i for i, (ep, t) in enumerate(adapter.dataset.all_steps)
                  if int(ep) == episode and int(t)+16 < length]
    if not candidates:
        raise ValueError('No complete target action window for interface verification')
    return dict(train_episodes=[], validation_episodes=[episode],
                validation_indices=[candidates[len(candidates)//3], candidates[2*len(candidates)//3]],
                scope='Target demonstration interface diagnostic only; no training or checkpoint selection')
