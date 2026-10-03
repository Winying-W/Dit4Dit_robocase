"""Task-balanced sampling with immutable, shared training-only normalization."""
import hashlib
import json
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from DiT4DiT.dataloader.robocasa365_datasets import Robocasa365DatasetAdapter
from DiT4DiT.dataloader.gr00t_lerobot.schema import DatasetStatisticalValues


def apply_statistics(ds, path, stats):
    modality = json.loads((Path(path)/'meta/modality.json').read_text())
    for group, column in [('action','action'), ('state','observation.state')]:
        for name, desc in modality[group].items():
            values = {key:value[desc['start']:desc['end']] for key,value in stats[column].items()}
            getattr(ds.dataset.metadata.statistics, group)[name] = DatasetStatisticalValues.model_validate(values)
    ds.dataset.transforms.set_metadata(ds.dataset.metadata)


class BalancedTasks:
    def __init__(self, datasets, indices):
        self.datasets, self.indices = datasets, indices

    def sample_index(self, rng):
        task = int(rng.integers(len(self.datasets)))
        return [task, int(rng.choice(self.indices[task]))]

    def __getitem__(self, index):
        task, local_index = index
        return self.datasets[int(task)][int(local_index)]


def prepare_multitask_dataset(manifest_path, output):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    stats_path = manifest_path.parent/'normalization.json'
    assert hashlib.sha256(stats_path.read_bytes()).hexdigest() == manifest['normalization_sha256']
    stats = json.loads(stats_path.read_text())
    (output/'normalization.json').write_bytes(stats_path.read_bytes())
    cfg = OmegaConf.create(dict(dataset_py='robocasa365_datasets', action_horizon=16,
        max_state_dim=64, max_action_dim=32, image_size=[128,128], video_delta_indices=list(range(17)),
        action_video_freq_ratio=2, video_backend='decord', lerobot_version='v2.0',
        statistics_path=str((output/'normalization.json').resolve())))
    cfg = OmegaConf.merge(cfg, manifest.get('data_config', {}))
    datasets, training, validation, task_splits = [], [], [], []
    for task_id, row in enumerate(manifest['tasks']):
        path = Path(row['path'])
        local_cfg = OmegaConf.merge(cfg, {'dataset_path':str(path)})
        ds = Robocasa365DatasetAdapter(path, local_cfg)
        apply_statistics(ds, path, stats)
        train_set, val_set = set(row['train_episodes']), set(row['validation_episodes'])
        excluded_set = set(row.get("excluded_episodes", []))
        assert excluded_set.isdisjoint(train_set | val_set)
        assert train_set.isdisjoint(val_set)
        train_idx, by_val_ep = [], {}
        observed_episodes = set()
        for idx, (ep, t) in enumerate(ds.dataset.all_steps):
            ep, t = int(ep), int(t); observed_episodes.add(ep)
            if ep in train_set:
                train_idx.append(idx)
            elif ep in val_set:
                length = ds.dataset.trajectory_lengths[ds.dataset.get_trajectory_index(ep)]
                if t + 16 < length:
                    by_val_ep.setdefault(ep, []).append(idx)
        assert observed_episodes == train_set | val_set | excluded_set, row['task']
        assert set(by_val_ep) == val_set, row['task']
        # Two deterministic windows from each of four held-out episodes per task.
        # All held-out episodes remain excluded from training/statistics.
        val_idx = []
        for ep in sorted(val_set)[:4]:
            indices = by_val_ep[ep]
            val_idx.extend([indices[len(indices)//3], indices[2*len(indices)//3]])
        task_splits.append(dict(task=row['task'], path=str(path), train_episodes=row['train_episodes'],
            validation_episodes=row['validation_episodes'], excluded_episodes=sorted(excluded_set), validation_indices=val_idx,
            train_frames=len(train_idx)))
        validation.extend([[task_id, idx] for idx in val_idx])
        training.append(np.asarray(train_idx, dtype=np.int64)); datasets.append(ds)
    split = dict(task_set=manifest['task_set'], tasks=task_splits, sampling='uniform task then uniform training frame',
        validation_indices=validation, seed=manifest['seed'],
        train_frames=sum(len(indices) for indices in training),
        train_episodes=manifest['train_episodes'], validation_episodes=manifest['validation_episodes'],
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        statistics_source=manifest['statistics_source'])
    (output/'split.json').write_text(json.dumps(split, indent=2)+'\n')
    (output/'dataset_manifest.json').write_bytes(manifest_path.read_bytes())
    OmegaConf.save(cfg, output/'data_config.yaml')
    return BalancedTasks(datasets, training), None, validation, split
