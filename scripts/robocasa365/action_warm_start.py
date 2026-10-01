"""Guarded Atomic18 -> Composite16 action initialization, distinct from resume.

The original16-demo step2000 comparison is a separate experiment. This path
transfers an Atomic action branch with matching adapter semantics and effective
normalization; optimizer, scheduler, sampler and local update count stay fresh.
"""
import hashlib
import json
from pathlib import Path


CONFIG_KEYS=('dataset_py','action_horizon','max_state_dim','max_action_dim','image_size',
             'video_delta_indices','action_video_freq_ratio','lerobot_version')
SEMANTIC_SOURCES=('DiT4DiT/dataloader/robocasa365_datasets.py',
    'DiT4DiT/dataloader/gr00t_lerobot/transform/state_action.py','scripts/robocasa365/multitask_data.py')
PROTOCOL='atomic_to_composite_action_warmstart_v1'


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):value.update(data)
    return value.hexdigest()


def validate_contract(source_config,target_config,source_stats,target_stats,source_split,target_split):
    """Compare the statistics actually used by the shared adapter, not unused quantiles."""
    if source_config.get('dataset_py')!='robocasa365_datasets':
        raise ValueError('Warm start requires the RoboCasa365 adapter')
    for key in CONFIG_KEYS:
        if source_config.get(key)!=target_config.get(key):
            raise ValueError(f'Warm-start input contract differs: {key}')
    for key in ('min','max'):
        if source_stats['action'][key]!=target_stats['action'][key]:
            raise ValueError(f'Effective action normalization differs: {key}')
    if source_split['task_set']!='atomic_seen' or target_split['task_set']!='composite_seen':
        raise ValueError('This warm-start protocol is Atomic18 to Composite16 only')
    source_tasks={row['task'] for row in source_split['tasks']}
    target_tasks={row['task'] for row in target_split['tasks']}
    if len(source_tasks)!=18 or len(target_tasks)!=16 or not source_tasks.isdisjoint(target_tasks):
        raise ValueError('Expected all18 Atomic and all16 disjoint Composite tasks')
    return dict(config_keys=list(CONFIG_KEYS),effective_continuous_action_normalization='min_max',
        state_normalization='raw16D padded to64; no additional normalization',
        source_tasks=sorted(source_tasks),target_tasks=sorted(target_tasks))


def load_action_warm_start(model, checkpoint, *, source_run, base_checkpoint,
                          target_config, target_stats, target_split):
    import torch
    from omegaconf import OmegaConf

    checkpoint=Path(checkpoint).resolve();source_run=Path(source_run).resolve()
    root=Path(__file__).resolve().parents[2]
    manifest_path=source_run/'source_hashes.json';source_hashes=json.loads(manifest_path.read_text())
    # The source checkpoint's frozen adapter implementation must match the one
    # that will actually encode target training samples.
    for name in SEMANTIC_SOURCES:
        expected=source_hashes[name]
        if sha(source_run/'source'/name)!=expected or sha(root/name)!=expected:
            raise ValueError(f'Warm-start adapter implementation differs: {name}')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if payload['phase']!='action' or payload['global_step']<1:
        raise ValueError('Warm start requires an action-training checkpoint')
    audit_path=source_run/'checkpoint_audits'/f"step_{payload['global_step']:06d}.json"
    audit=json.loads(audit_path.read_text());checkpoint_sha=sha(checkpoint)
    if (audit['status']!='complete' or audit['global_step']!=payload['global_step'] or
            audit['checkpoint_sha256']!=checkpoint_sha):
        raise ValueError('Warm-start checkpoint is not the independently audited source weight')
    if Path(payload['base_checkpoint']).resolve()!=Path(base_checkpoint).resolve():
        raise ValueError('Warm-start released base differs')
    source_stats_path=checkpoint.parent/'normalization.json'
    if sha(source_stats_path)!=payload['normalization_sha256']:
        raise ValueError('Warm-start normalization artifact does not match checkpoint')
    source_stats=json.loads(source_stats_path.read_text())
    source_config=OmegaConf.to_container(OmegaConf.load(checkpoint.parent/'data_config.yaml'),resolve=True)
    target_config=OmegaConf.to_container(target_config,resolve=True)
    contract=validate_contract(source_config,target_config,source_stats,target_stats,payload['split'],target_split)
    expected={name:param for name,param in model.named_parameters() if name.startswith('action_model.')}
    weights=payload['trained_state']
    if set(weights)!=set(expected):raise ValueError('Warm start must contain the entire action branch and no other weights')
    for name,param in expected.items():
        value=weights[name]
        if value.shape!=param.shape or not torch.isfinite(value).all():
            raise ValueError(f'Invalid warm-start tensor: {name}')
    # Checkpoint reading and assignment must not consume the new run's RNG.
    before=torch.get_rng_state().clone()
    result=model.load_state_dict(weights,strict=False)
    assert not result.unexpected_keys and torch.equal(before,torch.get_rng_state())
    assert all(torch.equal(param.detach().cpu(),weights[name]) for name,param in expected.items())
    lineage=dict(protocol=PROTOCOL,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,
        global_step=payload['global_step'],source_run=str(source_run),source_manifest_sha256=sha(manifest_path),
        source_checkpoint_audit_sha256=sha(audit_path),
        source_normalization_sha256=payload['normalization_sha256'],compatibility=contract,
        optimizer_reset=True,scheduler_reset=True,sampler_reset=True,local_steps_start_at_zero=True,
        transferred_tensors=len(expected),transferred_parameters=sum(p.numel() for p in expected.values()),
        video_vae_text_weights_transferred=False,original16_demo_step2000_ab=False)
    del payload
    return lineage
