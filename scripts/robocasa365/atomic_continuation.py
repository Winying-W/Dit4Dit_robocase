"""Initialize a new single-GPU Atomic18 A/B from the audited four-GPU50k policy.

This is a new experiment with fresh optimizer/sampler state, not DDP checkpoint
resume and not the historical16-demo step2000 comparison.
"""
import json
from pathlib import Path

from scripts.robocasa365.action_warm_start import CONFIG_KEYS, SEMANTIC_SOURCES, sha


PROTOCOL='same_atomic18_action_warmstart_v1'


def load_atomic_action_start(model, checkpoint, *, source_run, base_checkpoint,
                             target_config, target_stats, target_split):
    import torch
    from omegaconf import OmegaConf

    checkpoint=Path(checkpoint).resolve();source_run=Path(source_run).resolve()
    root=Path(__file__).resolve().parents[2]
    source_hashes=json.loads((source_run/'source_hashes.json').read_text())
    for name in SEMANTIC_SOURCES:
        expected=source_hashes[name]
        if sha(source_run/'source'/name)!=expected or sha(root/name)!=expected:
            raise ValueError(f'Atomic continuation adapter implementation differs: {name}')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if saved.get('phase')!='action' or saved.get('global_step')!=50000:
        raise ValueError('This experiment requires the audited Atomic18 action step50000')
    audit_path=source_run/'checkpoint_audits/step_050000.json'
    audit=json.loads(audit_path.read_text());checkpoint_sha=sha(checkpoint)
    if audit.get('status')!='complete' or audit.get('global_step')!=50000 or audit.get('checkpoint_sha256')!=checkpoint_sha:
        raise ValueError('Atomic continuation source is not the audited50k weight')
    if Path(saved['base_checkpoint']).resolve()!=Path(base_checkpoint).resolve():
        raise ValueError('Atomic continuation released base differs')
    source_stats_path=checkpoint.parent/'normalization.json'
    if sha(source_stats_path)!=saved['normalization_sha256']:
        raise ValueError('Atomic continuation source statistics do not match checkpoint')
    if json.loads(source_stats_path.read_text())!=target_stats:
        raise ValueError('Same-task continuation requires identical complete normalization')
    if target_split!=saved['split']:
        raise ValueError('Same-task continuation requires the original Atomic18 split and manifest')
    tasks=[row['task'] for row in target_split['tasks']]
    if target_split['task_set']!='atomic_seen' or len(tasks)!=len(set(tasks)) or len(tasks)!=18:
        raise ValueError('Same-task continuation requires all18 Atomic tasks')
    source_config=OmegaConf.to_container(OmegaConf.load(checkpoint.parent/'data_config.yaml'),resolve=True)
    target_config=OmegaConf.to_container(target_config,resolve=True)
    for key in CONFIG_KEYS:
        if source_config.get(key)!=target_config.get(key):
            raise ValueError(f'Atomic continuation input contract differs: {key}')
    expected={name:p for name,p in model.named_parameters() if name.startswith('action_model.')}
    weights=saved['trained_state']
    if set(weights)!=set(expected):
        raise ValueError('Atomic continuation must transfer the complete action branch only')
    for name,p in expected.items():
        if weights[name].shape!=p.shape or not torch.isfinite(weights[name]).all():
            raise ValueError(f'Invalid Atomic continuation tensor: {name}')
    rng=torch.get_rng_state().clone()
    result=model.load_state_dict(weights,strict=False)
    assert not result.unexpected_keys and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(p.detach().cpu(),weights[name]) for name,p in expected.items())
    return dict(protocol=PROTOCOL,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,
        source_run=str(source_run),source_checkpoint_audit_sha256=sha(audit_path),global_step=50000,
        source_normalization_sha256=saved['normalization_sha256'],source_split_identical=True,
        config_keys=list(CONFIG_KEYS),tasks=tasks,optimizer_reset=True,scheduler_reset=True,
        sampler_reset=True,local_steps_start_at_zero=True,transferred_tensors=len(expected),
        transferred_parameters=sum(p.numel() for p in expected.values()),
        video_vae_text_weights_transferred=False,original16_demo_step2000_ab=False)
