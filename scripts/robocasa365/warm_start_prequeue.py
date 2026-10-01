"""Extra evidence requirements for the Atomic-to-Composite training candidate."""
import hashlib
import json
from pathlib import Path

from scripts.robocasa365.action_warm_start import PROTOCOL


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_warm_start_evidence(run, source, plan, read):
    requested=plan['warm_start']
    assert requested['protocol']==PROTOCOL
    assert requested['source_global_step']==50000 and requested['local_initial_step']==0
    reuse=read('warm_start_evidence_reuse.json')
    previous=Path(reuse['previous_run'])
    previous_acceptance=json.loads((previous/'prequeue_acceptance.json').read_text())
    assert previous_acceptance['status']=='complete'
    assert sha(previous/'prequeue_acceptance.json')==reuse['previous_acceptance_sha256']
    previous_source=json.loads((previous/'source_hashes.json').read_text())
    assert sha(previous/'source_hashes.json')==previous_acceptance['source_hashes_sha256']
    changed={name for name in previous_source if source.get(name)!=previous_source[name]}
    expected_changed={'scripts/robocasa365/train_distributed.py',
        'scripts/robocasa365/verify_model_batch_cpu.py','scripts/robocasa365/verify_prequeue.py'}
    assert changed==expected_changed==set(reuse['changed_sources']),changed
    unchanged=set(previous_source)-changed
    assert unchanged==set(reuse['unchanged_dependency_sha256'])
    for name in unchanged:
        assert source[name]==previous_source[name]==reuse['unchanged_dependency_sha256'][name]
        assert sha(previous/'source'/name)==source[name]
    # All model, data, simulator, scene and generic DDP-control sources are
    # unchanged. The changed trainer and initialization get fresh checks below.
    for name, digest in reuse['reused_evidence_sha256'].items():
        assert sha(run/name)==sha(previous/name)==digest,name
        if name in previous_acceptance['evidence_sha256']:
            assert previous_acceptance['evidence_sha256'][name]==digest,name
    assert {'adapter_verification/acceptance.json','environment_preflight/acceptance.json',
        'scene_protocol_cpu/acceptance.json','scene_protocol_tests.log','final_cpu_tests.log',
        'image_alignment.json'}.issubset(reuse['reused_evidence_sha256'])
    assert 'model_batch_cpu_backward/acceptance.json' not in reuse['reused_evidence_sha256']
    assert 'cpu_runtime_import/acceptance.json' not in reuse['reused_evidence_sha256']

    controls=read('warm_start_controls/cpu_acceptance_summary.json')
    assert controls['status']=='complete' and controls['unit_tests_passed']==6
    for name,digest in controls['evidence_sha256'].items():
        assert sha(run/'warm_start_controls'/name)==digest,name
    control_sources=read('warm_start_controls/source_hashes.json')
    for name in ['scripts/robocasa365/action_warm_start.py',
                 'scripts/robocasa365/tests/test_action_warm_start.py',
                 'scripts/robocasa365/verify_action_warm_start_cpu.py']:
        assert control_sources[name]==source[name],name
    actual=read('warm_start_controls/real_action_head/acceptance.json')
    model=read('model_batch_cpu_backward/acceptance.json')
    assert model['warm_start']==actual['lineage']
    lineage=model['warm_start']
    assert lineage['protocol']==PROTOCOL and lineage['checkpoint_sha256']==requested['checkpoint_sha256']
    assert Path(lineage['checkpoint'])==Path(requested['checkpoint'])
    assert Path(lineage['source_run'])==Path(requested['source_run'])
    assert lineage['global_step']==50000 and lineage['transferred_tensors']==247
    assert lineage['transferred_parameters']==163276320
    assert all(lineage[name] for name in ['optimizer_reset','scheduler_reset','sampler_reset','local_steps_start_at_zero'])
    assert not lineage['video_vae_text_weights_transferred'] and not lineage['original16_demo_step2000_ab']
    assert model['source_sha256']==source['scripts/robocasa365/verify_model_batch_cpu.py']
    assert model['manifest_sha256']==sha(run/'prepared/manifest.json')
    assert model['optimizer_updates']==0 and model['gradients']['trainable_tensors']==247
    runtime=read('cpu_runtime_import/acceptance.json')
    assert runtime['world_size']==4 and runtime['collective_sum']==10
    provenance=read('fresh_checks.json')
    assert provenance['source_hashes_sha256']==sha(run/'source_hashes.json')
    for check in ['model_batch_cpu_backward','cpu_runtime_import']:
        assert provenance[check]['exit_code']==0
        assert provenance[check]['acceptance_sha256']==sha(run/check/'acceptance.json')
    return dict(passed=True,protocol=PROTOCOL,lineage=lineage,
        full_model_cpu_forward_backward_fresh=True,trainer_four_process_import_fresh=True,
        reused_controls='Six warm-start contract tests, genuine247-tensor load, unchanged data/scene and generic DDP controls; source and evidence hashes verified.',
        pending='CUDA optimizer updates and independent checkpoint resume must pass after allocation; no Composite success rate yet.')
