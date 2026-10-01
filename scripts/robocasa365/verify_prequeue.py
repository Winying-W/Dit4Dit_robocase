"""Collect executed CPU/data/environment checks for the declared four-GPU run.

This gate deliberately does not claim that unallocated CUDA/NCCL hardware passed.
Run from the frozen source directory, after its hashes and task.yaml are written.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from scripts.robocasa365.scene_protocol import OFFICIAL, STABLE, protocol_spec
from scripts.robocasa365.verify_task_environments import validate_environment_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve()
    assert Path.cwd().resolve()==run/'source', 'Validate the frozen source, not the working tree'
    checks={}
    evidence={}
    def read(name):
        path=run/name
        evidence[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    source=read('source_hashes.json')
    for name,digest in source.items():
        path=run/'source'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
        if path.suffix=='.py':ast.parse(path.read_text(),filename=name)
    manifest=read('prepared/manifest.json');plan=read('plan.json')
    scene=plan.get('scene_protocol',protocol_spec())
    assert scene==protocol_spec(scene['name'])
    entrypoint=plan.get('entrypoint_script','scripts/volc/entrypoint_robocasa365_atomic18_4gpu.sh')
    assert entrypoint in source
    subprocess.run(['bash','-n',entrypoint],check=True)
    checks['source_and_syntax']=dict(passed=True,files=len(source))
    expected={
        'atomic_seen':dict(tasks=18,train=8216,validation=910,episodes=9126,frames=2231347),
        'composite_seen':dict(tasks=16,train=7271,validation=806,episodes=8077,frames=6002265),
    }[manifest['task_set']]
    names=[row['task'] for row in manifest['tasks']]
    assert len(names)==len(set(names))==expected['tasks']
    assert manifest['task_scope']=='full_official_task_set'
    assert manifest['train_episodes']==plan['train_episodes']==expected['train']
    assert manifest['validation_episodes']==plan['validation_episodes']==expected['validation']
    assert hashlib.sha256((run/'prepared/normalization.json').read_bytes()).hexdigest()==manifest['normalization_sha256']
    for row in manifest['tasks']:
        assert set(row['train_episodes']).isdisjoint(row['validation_episodes'])
        path=Path(row['path']);assert path.is_dir()
        for name in ['info.json','modality.json','tasks.jsonl','episodes.jsonl','stats.json']:
            assert (path/'meta'/name).is_file(),(path,name)
    adapter=read('adapter_verification/acceptance.json')
    assert adapter['status']=='complete' and [row['task'] for row in adapter['tasks']]==names
    assert adapter['train_episodes']==expected['train'] and adapter['validation_episodes']==expected['validation']
    assert adapter['episodes']==expected['episodes'] and adapter['frames']==expected['frames']
    assert all(row['boundary_mask_passed'] and row['roundtrip_max_error']<1e-6 for row in adapter['tasks'])
    checks['data_and_actions']=dict(passed=True,tasks=expected['tasks'],train_episodes=expected['train'],validation_episodes=expected['validation'],
        frames=adapter['frames'],max_roundtrip_error=max(row['roundtrip_max_error'] for row in adapter['tasks']))
    if (run/'data_evidence_reuse.json').exists():
        reused=read('data_evidence_reuse.json')
        assert reused['status']=='complete' and reused['tasks_paths_episode_splits_identical'] and reused['normalization_identical']
        assert reused['new_manifest_sha256']==evidence['prepared/manifest.json']
        assert reused['reused_evidence_sha256']==evidence['adapter_verification/acceptance.json']
        checks['data_evidence_reuse']=reused
        identity=read('data_content_identity.json')
        assert identity['status']=='complete' and identity['manifest_sha256']==evidence['prepared/manifest.json']
        assert identity['verified_parquet_files']==expected['episodes']
        checks['reused_data_content_identity']=identity
    environment=read('environment_preflight/acceptance.json')
    assert environment['status']=='complete' and [row['task'] for row in environment['tasks']]==names
    assert all(row['passed'] and row['action_chain_runtime_passed'] and row['returncode']==0
               and row['steps']>=8 and row['controller']['action_dim']==12
               and row['controller']['input_type']=='delta' for row in environment['tasks'])
    pending_render=False
    if scene['name']==STABLE:
        assert environment['manifest_sha256']==evidence['prepared/manifest.json']
        rendering=environment['rendering_enabled']
        assert isinstance(rendering,bool)
        validate_environment_report(environment,manifest,scene['name'],rendering)
        pending_render=not rendering
        if pending_render:
            assert plan['all_task_render_gate_before_training'] is True
        protocol_cpu=read('scene_protocol_cpu/acceptance.json')
        assert protocol_cpu['status']=='cpu_scope_complete_gpu_rendered_gate_pending'
        assert protocol_cpu['scene_protocol']==STABLE and protocol_cpu['learned_policy_trials_added']==0
        for name,digest in protocol_cpu['evidence_sha256'].items():
            assert hashlib.sha256((run/'scene_protocol_cpu'/name).read_bytes()).hexdigest()==digest,name
        ipc=json.loads((run/'scene_protocol_cpu/cpu_ipc_stable/acceptance.json').read_text())
        assert ipc['scene_protocol']==scene and ipc['learned_policy_trials']==0
        for name,digest in ipc['source_sha256'].items():
            assert source[name]==digest,name
        new_tests=run/'scene_protocol_tests.log'
        assert 'Ran 38 tests' in new_tests.read_text() and new_tests.read_text().rstrip().endswith('OK')
        evidence['scene_protocol_tests.log']=hashlib.sha256(new_tests.read_bytes()).hexdigest()
        checks['scene_protocol']=dict(passed=True,protocol=scene,tests=38,
            actual_simulator_cpu_ipc=protocol_cpu['actual_simulator_ipc_cpu_steps'],
            rendered_gate_pending=pending_render)
    else:
        assert scene['name']==OFFICIAL
    checks['environments']=dict(passed=True,tasks=expected['tasks'],steps_per_task=8,
        rendering_verified=not pending_render,
        scope='reset/step/controller; all-task rendering pending allocated hardware' if pending_render else
              'reset/step/controller/render only')
    if (run/'unchanged_evidence_reuse.json').exists():
        reuse=read('unchanged_evidence_reuse.json')
        previous=Path(reuse['previous_run'])
        old_acceptance_bytes=(previous/'prequeue_acceptance.json').read_bytes()
        assert hashlib.sha256(old_acceptance_bytes).hexdigest()==reuse['previous_acceptance_sha256']
        old_acceptance=json.loads(old_acceptance_bytes)
        assert old_acceptance['status']=='complete'
        old_source=json.loads((previous/'source_hashes.json').read_text())
        assert hashlib.sha256((previous/'source_hashes.json').read_bytes()).hexdigest()==old_acceptance['source_hashes_sha256']
        for name,digest in reuse['unchanged_dependency_sha256'].items():
            assert old_source[name]==source[name]==digest,name
        assert {name for name in old_source if name.startswith('DiT4DiT/')}.issubset(reuse['unchanged_dependency_sha256'])
        required=['scripts/robocasa365/train_distributed.py','scripts/robocasa365/distributed_support.py',
                  'scripts/robocasa365/multitask_data.py','scripts/robocasa365/eval_protocol.py',
                  'scripts/robocasa365/verify_model_batch_cpu.py','scripts/robocasa365/tests/test_distributed.py',
                  'scripts/robocasa365/tests/test_multitask.py']
        assert set(required).issubset(reuse['unchanged_dependency_sha256'])
        for name in reuse['reused_evidence_files']:
            assert hashlib.sha256((run/name).read_bytes()).hexdigest()==old_acceptance['evidence_sha256'][name],name
        checks['unchanged_evidence_reuse']=reuse
    model=read('model_batch_cpu_backward/acceptance.json')
    gradients=model['gradients']
    assert model['status']=='complete' and model['device']=='cpu' and model['batch_size']==4
    assert model['strict_released_checkpoint_load'] and model['real_dataset_examples']
    assert gradients['trainable_tensors']==247 and not gradients['missing_gradients'] and gradients['all_gradients_finite']
    assert all(value>0 for value in gradients['encoder_decoder_gradient_norms'].values())
    checks['full_model_cpu_forward_backward']=model
    tests=run/'final_cpu_tests.log'
    assert 'Ran 10 tests' in tests.read_text() and tests.read_text().strip().endswith('OK')
    evidence['final_cpu_tests.log']=hashlib.sha256(tests.read_bytes()).hexdigest()
    checks['four_process_gradient_and_resume_tests']=dict(passed=True,test_count=10,
        scope='Four-process Gloo gradient numerical oracle, AdamW/RNG resume and benchmark protocol tests')
    runtime=read('cpu_runtime_import/acceptance.json')
    assert runtime['status']=='complete' and runtime['world_size']==4 and runtime['collective_sum']==10
    assert all(row['reused_import_initialized_group'] for row in runtime['ranks'])
    checks['actual_torchrun_trainer_imports']=runtime
    alignment=read('image_alignment.json');assert alignment['status']=='complete'
    checks['camera_alignment']=dict(passed=True,scope=alignment['scope'])
    if (run/'gt_protocol_ep11/result.json').exists():
        gt=read('gt_protocol_ep11/result.json')
        assert gt['task']=='StirVegetables' and gt['episode']==11 and gt['success']
        checks['single_gt_protocol_check']=dict(passed=True,task=gt['task'],episode=11,steps=gt['steps'],scope=gt['scope'])
    evaluation=manifest['evaluation']
    assert evaluation['planned_trials']==100*expected['tasks'] and len(evaluation['seeds'])==100
    assert len(evaluation['selection_seeds'])==20
    assert set(evaluation['seeds']).isdisjoint(evaluation['selection_seeds'])
    assert plan['gpu_count']==4 and plan['global_batch_size']==64 and plan['planned_updates']==50000
    config=yaml.safe_load((run/'task.yaml').read_text())
    prior=yaml.safe_load((run.parent/'robocasa365_atomic_20260924/task.yaml').read_text())
    assert config['ImageUrl']==prior['ImageUrl'] and config['Storages']==prior['Storages']
    assert len(config['TaskRoleSpecs'])==1 and config['TaskRoleSpecs'][0]['RoleReplicas']==1
    assert config['TaskRoleSpecs'][0]['ResourceSpec']['GPUNum']==4
    assert config['Entrypoint']==f'bash {run}/source/{entrypoint}'
    assert all(row['Type']!='Nas' for row in config['Storages'])
    assert Path(os.environ['SIM365_PYTHON']).is_file()
    assert Path(os.environ['IMAGEIO_FFMPEG_EXE']).is_file()
    checkpoint=run.parents[1]/'artifacts/checkpoints/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt'
    verified=json.loads((checkpoint.parent/'DOWNLOAD_VERIFIED.json').read_text())
    assert checkpoint.stat().st_size==verified['bytes'] and verified['sha256']==plan['base_sha256']
    local=json.loads((checkpoint.parent.parent/'LOCAL_CONFIGURATION.json').read_text())
    assert hashlib.sha256((checkpoint.parent.parent/'config.yaml').read_bytes()).hexdigest()==local['local_sha256']
    assert torch.distributed.is_nccl_available()
    for script in ['train_distributed.py','evaluate_multitask.py','verify_four_gpu_evaluation.py']:
        with (run/(script.removesuffix('.py')+'_cli_check.txt')).open('w') as stream:
            subprocess.run([sys.executable,'scripts/robocasa365/'+script,'--help'],
                stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=90)
    with (run/'eval_simulator_cli_check.txt').open('w') as stream:
        subprocess.run([os.environ['SIM365_PYTHON'],'scripts/robocasa365/eval_simulator.py','--help'],
            stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=90)
    checks['runtime_and_cloud_configuration']=dict(passed=True,python=sys.version,torch=torch.__version__,
        cuda_build=torch.version.cuda,nccl_version=list(torch.cuda.nccl.version()),
        base_checkpoint_sha256=verified['sha256'],
        mounts_and_image_match_running_single_gpu=True,cloud_gpu_count=4)
    if plan.get('warm_start'):
        from scripts.robocasa365.warm_start_prequeue import verify_warm_start_evidence
        checks['warm_start']=verify_warm_start_evidence(run,source,plan,read)
    evidence['task.yaml']=hashlib.sha256((run/'task.yaml').read_bytes()).hexdigest()
    pending=[
        'Actual four-GPU CUDA/NCCL communication, memory and finite updates',
        'Independent torchrun checkpoint/optimizer/RNG resume and equal rank weights',
        'Four simultaneous CUDA policies and simulators with actual action audit']
    if pending_render:pending.insert(0,'All16 Composite environments with stable_counter_v1 and real three-camera rendering')
    report=dict(status='complete',verified_utc=datetime.now(timezone.utc).isoformat(),checks=checks,scene_protocol=scene,
        evidence_sha256=evidence,source_hashes_sha256=evidence['source_hashes.json'],
        pending_allocated_hardware_checks=pending,
        long_training_rule='All allocated hardware gates must pass before training beyond step12.',
        scope='Prequeue software/data checks passed. This does not guarantee cloud hardware or scheduler availability.')
    target=run/'prequeue_acceptance.json'
    temp=target.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(target)
    print(json.dumps(dict(status=report['status'],checks=list(checks),pending=report['pending_allocated_hardware_checks'])))


if __name__=='__main__':main()
