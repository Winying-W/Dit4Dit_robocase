"""Start a new same-data Composite16 experiment from an audited action checkpoint.

Transfers only the action branch. This is distinct from exact DDP resume, from
Atomic50k continuation, and from the historical16-demo step2000 comparison.
"""
import json
from pathlib import Path

from scripts.robocasa365.action_warm_start import CONFIG_KEYS, SEMANTIC_SOURCES, sha

PROTOCOL = 'same_composite16_action_warmstart_v1'


def load_composite_action_start(model, checkpoint, *, source_run, checkpoint_audit,
                                expected_step, base_checkpoint, target_config,
                                target_stats, target_split):
    import torch
    from omegaconf import OmegaConf

    if not isinstance(expected_step, int) or isinstance(expected_step, bool) or expected_step < 1:
        raise ValueError('Composite initialization requires an explicit positive source step')
    checkpoint = Path(checkpoint).resolve()
    source_run = Path(source_run).resolve()
    checkpoint_audit = Path(checkpoint_audit).resolve()
    root = Path(__file__).resolve().parents[2]
    hashes_path = source_run / 'source_hashes.json'
    hashes = json.loads(hashes_path.read_text())
    for name in SEMANTIC_SOURCES:
        expected = hashes[name]
        if sha(source_run / 'source' / name) != expected or sha(root / name) != expected:
            raise ValueError(f'Composite continuation adapter implementation differs: {name}')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if (saved.get('format_version') != 2 or saved.get('phase') != 'action'
            or saved.get('global_step') != expected_step or saved.get('phase_step') != expected_step):
        raise ValueError('Composite continuation source phase or explicitly selected step differs')
    digest = sha(checkpoint)
    evidence = json.loads(checkpoint_audit.read_text())
    if evidence.get('status') != 'complete':
        raise ValueError('Composite continuation audit is incomplete')
    audit = evidence.get('cpu_readback', evidence)
    if (audit.get('status') != 'complete' or audit.get('global_step') != expected_step
            or audit.get('checkpoint_sha256') != digest
            or audit.get('checkpoint_bytes') != checkpoint.stat().st_size):
        raise ValueError('Composite continuation source is not the explicitly audited weight')
    for flag in ['all_trained_weights_finite', 'all_adam_moments_finite',
                 'all_optimizer_steps_match', 'scheduler_step_matches', 'split_and_normalization_match']:
        if audit.get(flag) is not True:
            raise ValueError(f'Composite continuation audit lacks successful check: {flag}')
    origin = saved.get('warm_start') or {}
    initialization = audit.get('initialization') or {}
    if (origin.get('protocol') != 'atomic_to_composite_action_warmstart_v1'
            or origin.get('global_step') != 50000
            or initialization.get('protocol') != origin['protocol']
            or initialization.get('source_global_step') != 50000
            or initialization.get('checkpoint_sha256') != origin.get('checkpoint_sha256')
            or saved['training_schedule'].get('initialization_checkpoint_sha256') != origin.get('checkpoint_sha256')):
        raise ValueError('Composite continuation requires the verified Atomic50k-to-Composite lineage')
    if (not all(origin.get(flag) is True for flag in
                ['optimizer_reset', 'scheduler_reset', 'sampler_reset', 'local_steps_start_at_zero'])
            or initialization.get('local_initial_step') != 0
            or initialization.get('optimizer_scheduler_sampler_reset') is not True):
        raise ValueError('Composite continuation source local-step/reset contract differs')
    if Path(saved['base_checkpoint']).resolve() != Path(base_checkpoint).resolve():
        raise ValueError('Composite continuation released base differs')
    statistics_path = checkpoint.parent / 'normalization.json'
    if sha(statistics_path) != saved['normalization_sha256']:
        raise ValueError('Composite continuation source statistics do not match checkpoint')
    if json.loads(statistics_path.read_text()) != target_stats:
        raise ValueError('Composite continuation requires identical complete normalization')
    if target_split != saved['split']:
        raise ValueError('Composite continuation requires the original Composite16 split and manifest')
    tasks = [row['task'] for row in target_split['tasks']]
    if target_split['task_set'] != 'composite_seen' or len(tasks) != 16 or len(set(tasks)) != 16:
        raise ValueError('Composite continuation requires all16 Composite seen tasks')
    source_config = OmegaConf.to_container(OmegaConf.load(checkpoint.parent / 'data_config.yaml'), resolve=True)
    target_config = OmegaConf.to_container(target_config, resolve=True)
    for key in CONFIG_KEYS:
        if source_config.get(key) != target_config.get(key):
            raise ValueError(f'Composite continuation input contract differs: {key}')
    expected = {name: p for name, p in model.named_parameters() if name.startswith('action_model.')}
    weights = saved['trained_state']
    if not expected or set(weights) != set(expected):
        raise ValueError('Composite continuation must transfer the complete action branch only')
    if (audit.get('trained_tensors') != len(expected)
            or audit.get('trained_parameters') != sum(p.numel() for p in expected.values())):
        raise ValueError('Composite continuation audited parameter scope differs')
    for name, param in expected.items():
        value = weights[name]
        if value.shape != param.shape or not torch.isfinite(value).all():
            raise ValueError(f'Invalid Composite continuation tensor: {name}')
    rng = torch.get_rng_state().clone()
    result = model.load_state_dict(weights, strict=False)
    assert not result.unexpected_keys and torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(param.detach().cpu(), weights[name]) for name, param in expected.items())
    return dict(protocol=PROTOCOL, checkpoint=str(checkpoint), checkpoint_sha256=digest,
        global_step=expected_step, source_run=str(source_run), source_manifest_sha256=sha(hashes_path),
        source_checkpoint_audit=str(checkpoint_audit), source_checkpoint_audit_sha256=sha(checkpoint_audit),
        source_normalization_sha256=saved['normalization_sha256'], source_split_identical=True,
        original_atomic_initialization_sha256=origin['checkpoint_sha256'],
        config_keys=list(CONFIG_KEYS), tasks=tasks, optimizer_reset=True, scheduler_reset=True,
        sampler_reset=True, local_steps_start_at_zero=True, transferred_tensors=len(expected),
        transferred_parameters=sum(p.numel() for p in expected.values()),
        video_vae_text_weights_transferred=False, original16_demo_step2000_ab=False)
