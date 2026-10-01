"""Read back the exact single-GPU Composite sampling experiment contract."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.robocasa365.phase_sampling import sha


def audit(checkpoint, run, mode, step, *, allow_cpu_preflight=False):
    checkpoint,run=Path(checkpoint),Path(run)
    plan=json.loads((run/'plan.json').read_text())
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    config=json.loads((checkpoint.parent/'run_config.json').read_text())
    manifest=json.loads(Path(plan['manifest']).read_text())
    schema=json.loads((run/'cpu_preflight/paired_uniform/action_trainable.json').read_text())
    assert payload['format_version']==1 and payload['phase']=='action'
    assert payload['global_step']==payload['phase_step']==step
    assert payload['base_checkpoint']==plan['base_checkpoint']==config['base_checkpoint']
    schedule=payload['training_schedule'];cpu=schedule.get('execution_device')=='cpu_preflight'
    assert cpu==allow_cpu_preflight, 'CPU preflight checkpoints are not CUDA training checkpoints'
    assert config['cpu_preflight']==cpu and config['execution_device']==('cpu' if cpu else 'cuda')
    assert schedule==config['training_schedule']
    assert schedule['action_steps']==plan['additional_updates_per_branch']==2000 and schedule['joint_steps']==0
    assert schedule['warmup_steps']==plan['warmup_steps']==200
    assert schedule['paired_ab_protocol']==plan['paired_rng'] and schedule['paired_ab_seed']==plan['seed']==42
    assert schedule['gradient_clipping']=='per_optimizer_group'
    batch=plan['cpu_preflight'] if cpu else plan
    accumulation=batch['accumulation'] if cpu else batch['gradient_accumulation']
    assert schedule['accumulation']==accumulation and schedule['microbatch_size']==batch['microbatch_size']
    assert config['effective_batch_size']==accumulation*batch['microbatch_size']
    sampling=schedule['sampling']
    assert sampling==config['sampling'] and sampling['mode']==mode and mode in plan['branches']
    assert sampling['protocol']==plan['paired_task_rng']
    assert sampling['pool_manifest_sha256']==sha(plan['phase_pools'])
    pools=json.loads(Path(plan['phase_pools']).read_text())
    assert sampling['pools_sha256']==pools['pools_sha256']
    assert sampling['source_sha256']==pools['source_sha256']
    assert sampling['mixture']==(dict(ordinary=1.,initial=0.,switch=0.) if mode=='paired_uniform' else dict(ordinary=.5,initial=.25,switch=.25))
    assert payload['warm_start']==config['warm_start']
    lineage=payload['warm_start']
    assert lineage['protocol']=='same_composite16_action_warmstart_v1' and lineage['global_step']==10000
    assert lineage['checkpoint_sha256']==plan['warm_start_sha256']==schedule['initialization_checkpoint_sha256']
    assert all(lineage[key] for key in ['optimizer_reset','scheduler_reset','sampler_reset','local_steps_start_at_zero'])
    assert payload['split']==config['split']==json.loads((checkpoint.parent/'split.json').read_text())
    split=payload['split']
    assert split['manifest_sha256']==sha(plan['manifest'])==plan['manifest_sha256']
    assert split['task_set']=='composite_seen' and len(split['tasks'])==16
    assert split['train_episodes']==7271 and split['validation_episodes']==806
    for actual,expected in zip(split['tasks'],manifest['tasks']):
        assert actual['task']==expected['task']
        for key in ['train_episodes','validation_episodes']:assert actual[key]==expected[key]
        assert set(actual['train_episodes']).isdisjoint(actual['validation_episodes'])
    assert sha(checkpoint.parent/'normalization.json')==manifest['normalization_sha256']
    trained=payload['trained_state']
    assert len(trained)==len(schema)==247 and set(trained)==set(schema)
    assert all(name.startswith('action_model.') for name in trained)
    assert sum(value.numel() for value in trained.values())==163276320
    groups=payload['optimizer']['param_groups'];state=payload['optimizer']['state']
    assert len(groups)==1 and groups[0]['name']=='action_model.'
    ids=groups[0]['params'];assert len(ids)==247 and set(ids)==set(state)
    assert len(set(ids))==len(ids)
    for (name,value),index in zip(trained.items(),ids):
        assert list(value.shape)==schema[name] and value.dtype==torch.float32 and torch.isfinite(value).all(),name
        entry=state[index];assert float(entry['step'])==step,name
        for key in ['exp_avg','exp_avg_sq']:
            assert entry[key].shape==value.shape and torch.isfinite(entry[key]).all(),(name,key)
        assert (entry['exp_avg_sq']>=0).all(),name
    assert payload['scheduler']['last_epoch']==step and payload['continuation_schedule'] is None
    expected_lr=plan['action_lr']*min((step+1)/200,1.)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-200)/1800)))
    assert math.isclose(groups[0]['lr'],expected_lr,rel_tol=1e-12)
    assert payload['scheduler']['_last_lr']==[groups[0]['lr']]
    assert len(payload['cuda_rng'])==(0 if cpu else 1)
    assert payload['torch_rng'].dtype==torch.uint8
    rng=np.random.default_rng(plan['seed'])
    examples=step*accumulation*batch['microbatch_size']
    for _ in range(examples):
        rng.integers(16);rng.integers(np.iinfo(np.int64).max)
    assert payload['sampler_rng']==rng.bit_generator.state
    return dict(status='passed',verified_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
        checkpoint_step=step,mode=mode,cpu_preflight=cpu,trained_tensors=247,trained_parameters=163276320,
        all_parameters_and_adam_moments_finite=True,optimizer_steps=step,
        sampler_draws_independently_reproduced=examples,sampler_rng_matches=True,
        source_composite_step=10000,source_checkpoint_sha256=plan['warm_start_sha256'],
        new_policy_trials=0,scope='CPU deserialization and contract audit. No new forward or closed-loop success claim.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--mode',choices=['paired_uniform','initial_switch'],required=True)
    parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-cpu-preflight',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    result=audit(args.checkpoint,args.run,args.mode,args.step,allow_cpu_preflight=args.allow_cpu_preflight)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
