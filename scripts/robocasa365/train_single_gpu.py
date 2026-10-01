"""Bounded single-task experiment: frozen video backbone, then partial video thaw.

Uses the released model's forward/loss unchanged. Stores cumulative trained weights,
optimizer, scheduler, RNG and immutable base reference for resumable checkpoints.
"""
import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from DiT4DiT.dataloader.robocasa365_datasets import Robocasa365DatasetAdapter
from DiT4DiT.dataloader.gr00t_lerobot.schema import DatasetStatisticalValues
from DiT4DiT.model.framework.base_framework import baseframework
from scripts.robocasa365.continuation_schedule import continuation_lrs
from scripts.robocasa365.paired_randomness import (
    paired_microbatch, paired_schedule, validate_paired_resume, clip_paired_groups,
)


def atomic_json(path, obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)
    mirror=os.environ.get('ROBO365_STATUS_MIRROR')
    if mirror:
        target=Path(mirror)/path.name;target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_suffix('.tmp');tmp.write_text(path.read_text());tmp.replace(target)


def prepare_dataset(path, output, seed, overfit_episode=None):
    # Select fixed-base demos by actual commands, never silently zero demonstrations.
    candidates=[]
    for p in sorted((path/'data').glob('*/episode_*.parquet')):
        df=pd.read_parquet(p,columns=['action'])
        ac=np.stack(df.action)
        if np.all(ac[:,:4]==0) and np.all(ac[:,4]==-1):
            candidates.append(int(p.stem.split('_')[-1]))
    if len(candidates)<20:raise ValueError(f'Need 20 fixed-base episodes, found {len(candidates)}')
    shuffled=np.random.default_rng(seed).permutation(candidates).tolist()
    train_eps=sorted(shuffled[:16]);val_eps=sorted(shuffled[16:20])
    arrays={'action':[], 'observation.state':[]}
    for ep in train_eps:
        df=pd.read_parquet(next((path/'data').glob(f'*/episode_{ep:06d}.parquet')))
        for key in arrays:arrays[key].append(np.stack(df[key]))
    stats={}
    for key,parts in arrays.items():
        x=np.concatenate(parts).astype(np.float32)
        stats[key]={name:fn(x,axis=0).tolist() for name,fn in [('min',np.min),('max',np.max),('mean',np.mean),('std',np.std)]}
        stats[key].update(q01=np.quantile(x,.01,axis=0).tolist(),q99=np.quantile(x,.99,axis=0).tolist())
    cfg=OmegaConf.create(dict(dataset_py='robocasa365_datasets',dataset_path=str(path),action_horizon=16,
        max_state_dim=64,max_action_dim=32,image_size=[128,128],video_delta_indices=list(range(17)),
        action_video_freq_ratio=2,video_backend='decord',lerobot_version='v2.0'))
    ds=Robocasa365DatasetAdapter(path,cfg)
    modality=json.loads((path/'meta/modality.json').read_text())
    for group,col in [('action','action'),('state','observation.state')]:
        for name,desc in modality[group].items():
            values={k:v[desc['start']:desc['end']] for k,v in stats[col].items()}
            getattr(ds.dataset.metadata.statistics,group)[name]=DatasetStatisticalValues.model_validate(values)
    ds.dataset.transforms.set_metadata(ds.dataset.metadata)
    normalization_episodes=list(train_eps)
    if overfit_episode is not None:
        if overfit_episode not in train_eps:
            raise ValueError('Overfit episode must be in the original training split')
        train_eps=[overfit_episode]
    train_indices=np.array([i for i,(ep,t) in enumerate(ds.dataset.all_steps) if int(ep) in train_eps],dtype=np.int64)
    val_indices=[]
    for ep in val_eps:
        idx=[i for i,(e,t) in enumerate(ds.dataset.all_steps) if int(e)==ep and t+16<ds.dataset.trajectory_lengths[ds.dataset.get_trajectory_index(ep)]]
        val_indices.extend([idx[len(idx)//3],idx[2*len(idx)//3]])
    split=dict(seed=seed,train_episodes=train_eps,validation_episodes=val_eps,non_mobile_candidates=len(candidates),
        train_frames=len(train_indices),validation_indices=val_indices,statistics_source='train episodes only')
    if overfit_episode is not None:
        split.update(experiment='single_demo_overfit',normalization_train_episodes=normalization_episodes,
            statistics_source='Frozen original 16-training-episode statistics; no held-out statistics')
    atomic_json(output/'split.json',split);atomic_json(output/'normalization.json',stats)
    OmegaConf.save(cfg,output/'data_config.yaml')
    return ds,train_indices,val_indices,split


from scripts.robocasa365.video_precision import prepare_partial_video_fp32


def configure(model,phase,*,video_parameters_fp32=False):
    if video_parameters_fp32:prepare_partial_video_fp32(model)
    model.requires_grad_(False)
    model.eval()
    model.action_model.requires_grad_(True);model.action_model.train()
    if phase=='partial_joint':
        transformer=model.backbone_interface.extractor.transformer
        # Blocks up to the extracted representation can adapt features used by action DiT.
        for i in (16,17):transformer.transformer_blocks[i].requires_grad_(True)
        transformer.train();transformer.enable_gradient_checkpointing()
    # Preserve upstream detach behavior: video loss updates video; action loss updates action.
    model.config.framework.cosmos25.training='joint' if phase=='partial_joint' else 'action'
    model.config.framework.cosmos25.future_loss_type='flow_matching'
    return {n:p for n,p in model.named_parameters() if p.requires_grad}


def batch_for(ds,index,phase):
    b=ds[index]
    if phase=='action':b['image']=b['image'][:1]
    return [b]


def forward_training_microbatch(model, batch, *, paired_ab=False, seed=42, update=0, micro=0):
    context=paired_microbatch(model,seed=seed,step=update,micro=micro) if paired_ab else nullcontext()
    with context:
        return model(examples=batch)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',type=Path);ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--action-steps',type=int,default=300)
    ap.add_argument('--joint-steps',type=int,default=100);ap.add_argument('--accumulation',type=int,default=4)
    ap.add_argument('--microbatch-size',type=int,default=1,help='Examples per model forward; effective batch is this value times accumulation')
    ap.add_argument('--stop-after',type=int);ap.add_argument('--resume',type=Path);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--extend-action-schedule',action='store_true',help='Explicit 20-step ramp from checkpoint LR to 3e-5, then cosine to 3e-6; preserve AdamW state')
    ap.add_argument('--snapshot-steps',type=int,nargs='*',default=[])
    ap.add_argument('--overfit-episode',type=int,help='Train one original train episode, keeping original training statistics and validation split')
    ap.add_argument('--warm-start',type=Path,help='Load action weights only; reset optimizer, scheduler, RNG and local update counter')
    ap.add_argument('--manifest',type=Path,help='Audited multitask manifest; mutually exclusive with --dataset')
    ap.add_argument('--warm-start-run',type=Path,help='Audited source run for an explicit same-data paired experiment')
    ap.add_argument('--warm-start-task-set',choices=['atomic_seen','composite_seen'],default='atomic_seen',
        help='Default Atomic18 step50000; Composite16 additionally requires an explicit source step and audit')
    ap.add_argument('--warm-start-step',type=int,help='Explicit source update count for Composite16 initialization')
    ap.add_argument('--warm-start-audit',type=Path,help='Completed CPU checkpoint audit or verified backup receipt for Composite16')
    ap.add_argument('--sampling-mode',choices=['legacy','paired_uniform','initial_switch'],default='legacy')
    ap.add_argument('--phase-pools',type=Path,help='Verified training-only phase pool manifest for the paired sampling comparison')
    ap.add_argument('--validation-interval',type=int,default=50)
    ap.add_argument('--checkpoint-interval',type=int,default=50)
    ap.add_argument('--warmup-steps',type=int,help='Explicit LR warmup for a longer training schedule; otherwise preserve the original short-run warmup')
    ap.add_argument('--paired-ab',action='store_true',help='Opt in to independent backbone/action RNG streams and per-group gradient clipping for a controlled single-GPU A/B')
    ap.add_argument('--video-parameters-fp32',action='store_true',help='Paired single-phase experiment: same FP32 video blocks16/17 in both arms, BF16 forward autocast')
    ap.add_argument('--cpu-preflight',action='store_true',help='Bounded real action or paired partial-joint training on CPU; isolated from CUDA checkpoints by the saved schedule')
    ap.add_argument('--preflight-validation-count',type=int,help='CPU preflight only: bound held-out forwards without changing the source data split')
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.accumulation < 1 or args.microbatch_size < 1 or args.validation_interval < 1 or args.checkpoint_interval < 1:
        raise ValueError('Batch sizes and logging/checkpoint intervals must be positive')
    if args.video_parameters_fp32 and not (args.paired_ab and ((args.action_steps>0 and args.joint_steps==0) or (args.action_steps==0 and args.joint_steps>0))):
        raise ValueError('FP32 video parameters require a single positive phase and --paired-ab')
    if args.cpu_preflight:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_available():
            raise ValueError('CPU preflight requires CUDA_VISIBLE_DEVICES empty')
        if args.stop_after is None or not 1<=args.stop_after<=4:
            raise ValueError('CPU preflight requires --stop-after in [1,4]')
        if args.joint_steps!=0 and not (args.action_steps==0 and args.joint_steps>0 and args.paired_ab):
            raise ValueError('CPU partial-joint preflight requires --action-steps 0, positive --joint-steps and --paired-ab')
        if args.preflight_validation_count is None or args.preflight_validation_count<1:
            raise ValueError('CPU preflight requires a positive --preflight-validation-count')
        torch.set_num_threads(8);torch.set_num_interop_threads(2)
    elif args.preflight_validation_count is not None:
        raise ValueError('Limited validation is supported only for CPU preflight')
    device=torch.device('cpu' if args.cpu_preflight else 'cuda')
    def precision_context():
        return torch.autocast('cpu',dtype=torch.bfloat16) if args.cpu_preflight else nullcontext()
    if args.warmup_steps is not None and args.warmup_steps < 1:
        raise ValueError('Warmup must be positive')
    schedule=dict(action_steps=args.action_steps,joint_steps=args.joint_steps,
        accumulation=args.accumulation,warmup_steps=args.warmup_steps)
    if args.microbatch_size!=1:schedule['microbatch_size']=args.microbatch_size
    if args.cpu_preflight:
        schedule.update(execution_device='cpu_preflight',preflight_validation_count=args.preflight_validation_count)
    if args.paired_ab:schedule.update(paired_schedule(args.seed))
    if args.video_parameters_fp32:schedule['selected_video_parameter_precision']='float32'
    if args.warm_start and args.resume:raise ValueError('Warm start and exact resume are mutually exclusive')
    if args.warm_start_run and not (args.manifest and args.warm_start and args.paired_ab):
        raise ValueError('Source run requires --manifest, --warm-start and --paired-ab')
    if args.warm_start_task_set=='composite_seen':
        if not (args.warm_start_run and args.manifest and args.warm_start and args.paired_ab):
            raise ValueError('Composite initialization requires --manifest, --warm-start, --warm-start-run and --paired-ab')
        if args.warm_start_step is None or args.warm_start_step < 1 or args.warm_start_audit is None:
            raise ValueError('Composite initialization requires --warm-start-step and --warm-start-audit')
    elif args.warm_start_step is not None or args.warm_start_audit is not None:
        raise ValueError('Source step/audit overrides are only supported for explicit Composite initialization')
    if args.sampling_mode!='legacy':
        if not (args.manifest and args.paired_ab and args.phase_pools):
            raise ValueError('Paired sampling requires --manifest, --paired-ab and --phase-pools')
        if not args.resume and not (args.warm_start and args.warm_start_task_set=='composite_seen'):
            raise ValueError('A new paired sampling run requires explicit Composite initialization')
    elif args.phase_pools:
        raise ValueError('Phase pools require an explicit paired sampling mode')
    if args.overfit_episode is not None and (args.joint_steps!=0 or not (args.warm_start or args.resume)):
        raise ValueError('Overfit requires frozen video and explicit warm-start/resume')
    if (out/'metrics.jsonl').exists() and args.resume is None:raise ValueError('Existing run: use --resume or a new output directory')
    source=Path(__file__).read_bytes()
    source_hash=hashlib.sha256(source).hexdigest()
    (out/f'trainer_{source_hash[:12]}.py').write_bytes(source)
    atomic_json(out/'trainer_source.json',dict(sha256=source_hash,path=str(Path(__file__).resolve())))
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if not args.cpu_preflight:torch.cuda.manual_seed_all(args.seed)
    rng=np.random.default_rng(args.seed)
    if bool(args.manifest) == bool(args.dataset):
        raise ValueError('Specify exactly one of --dataset or --manifest')
    if args.manifest:
        if args.overfit_episode is not None:
            raise ValueError('Multitask single-demo overfit is not supported')
        if args.warm_start and not args.warm_start_run:
            raise ValueError('Multitask warm start requires an explicit audited source run')
        from scripts.robocasa365.multitask_data import prepare_multitask_dataset
        ds,indices,val_indices,split=prepare_multitask_dataset(args.manifest,out)
    else:
        ds,indices,val_indices,split=prepare_dataset(args.dataset,out,args.seed,args.overfit_episode)
    if args.sampling_mode!='legacy':
        from scripts.robocasa365.phase_sampling import PhaseBalancedTasks, sha as pool_sha
        ds=PhaseBalancedTasks(ds,args.phase_pools,args.manifest,args.sampling_mode)
        schedule['sampling']=ds.sampling_spec
        pool_metadata=json.loads(args.phase_pools.read_text())
        pool_output=out/'phase_sampling';pool_output.mkdir(exist_ok=True)
        for source_file,destination in [(args.phase_pools,pool_output/'manifest.json'),
                (args.phase_pools.parent/pool_metadata['pools_file'],pool_output/pool_metadata['pools_file'])]:
            if destination.exists():
                if pool_sha(destination)!=pool_sha(source_file):raise ValueError('Saved phase pools differ on resume')
            # Keep the output process umask: copy2 can create root-owned0600
            # metadata on cloud workers, preventing the user's backup reader.
            else:shutil.copyfile(source_file,destination)
        atomic_json(out/'sampling.json',ds.sampling_spec)
    if args.cpu_preflight:val_indices=val_indices[:args.preflight_validation_count]
    model=baseframework.from_pretrained(str(args.checkpoint))
    model.backbone_interface.extractor.text_encoder.to(torch.bfloat16)
    model.requires_grad_(False);model=model.to(device).eval()
    if args.video_parameters_fp32:prepare_partial_video_fp32(model)
    cuda_devices=[] if args.cpu_preflight else [torch.cuda.current_device()]
    model.config.trainer.repeated_diffusion_steps=1
    model.config.datasets.vla_data=OmegaConf.load(out/'data_config.yaml')
    trained_names=set();start_time=time.monotonic();global_step=0
    resume=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
    lineage=resume.get('warm_start') if resume else None
    if args.warm_start:
        if args.manifest:
            warm_kwargs=dict(source_run=args.warm_start_run,base_checkpoint=args.checkpoint,
                target_config=model.config.datasets.vla_data,
                target_stats=json.loads((out/'normalization.json').read_text()),target_split=split)
            if args.warm_start_task_set=='composite_seen':
                from scripts.robocasa365.composite_continuation import load_composite_action_start
                lineage=load_composite_action_start(model,args.warm_start,
                    checkpoint_audit=args.warm_start_audit,expected_step=args.warm_start_step,**warm_kwargs)
            else:
                from scripts.robocasa365.atomic_continuation import load_atomic_action_start
                lineage=load_atomic_action_start(model,args.warm_start,**warm_kwargs)
            trained_names.update(name for name,_ in model.named_parameters() if name.startswith('action_model.'))
        else:
            warm=torch.load(args.warm_start,map_location='cpu',weights_only=False)
            assert warm['phase']=='action' and warm['base_checkpoint']==str(args.checkpoint.resolve())
            expected={name for name,_ in model.named_parameters() if name.startswith('action_model.')}
            assert set(warm['trained_state'])==expected
            assert json.loads((args.warm_start.parent/'normalization.json').read_text())==json.loads((out/'normalization.json').read_text())
            if args.overfit_episode is not None:
                assert split['normalization_train_episodes']==warm['split']['train_episodes']
                assert split['validation_episodes']==warm['split']['validation_episodes']
            result=model.load_state_dict(warm['trained_state'],strict=False);assert not result.unexpected_keys
            trained_names.update(warm['trained_state'])
            lineage=dict(checkpoint=str(args.warm_start.resolve()),global_step=warm['global_step'],optimizer_reset=True,local_steps_start_at_zero=True)
            del warm
        atomic_json(out/'warm_start.json',lineage)
    if lineage and lineage.get('protocol') in ('same_atomic18_action_warmstart_v1','same_composite16_action_warmstart_v1'):
        schedule['initialization_checkpoint_sha256']=lineage['checkpoint_sha256']
    if resume:
        assert resume['base_checkpoint']==str(args.checkpoint.resolve()) and resume['split']==split
        assert json.loads((args.resume.parent/'normalization.json').read_text()) == json.loads((out/'normalization.json').read_text())
        validate_paired_resume(resume.get('training_schedule',{}),schedule)
        if 'training_schedule' in resume and not args.extend_action_schedule:
            assert resume['training_schedule'] == schedule, 'Exact resume requires the same LR schedule and accumulation'
    if args.extend_action_schedule and (not resume or resume['phase']!='action' or args.joint_steps!=0):
        raise ValueError('LR extension requires an action checkpoint and joint_steps=0')
    if resume:
        result=model.load_state_dict(resume['trained_state'],strict=False);assert not result.unexpected_keys
        trained_names.update(resume['trained_state']);global_step=resume['global_step']
    phases=[('action',args.action_steps),('partial_joint',args.joint_steps)]
    atomic_json(out/'run_config.json',dict(base_checkpoint=str(args.checkpoint.resolve()),phases=phases,accumulation=args.accumulation,
        microbatch_size=args.microbatch_size,effective_batch_size=args.microbatch_size*args.accumulation,
        execution_device=str(device),cpu_preflight=args.cpu_preflight,
        selected_video_parameter_precision='float32' if args.video_parameters_fp32 else 'base_dtype',
        image_size=[128,128],video_lr=1e-5,action_lr=1e-4,seed=args.seed,loss='action + video (partial_joint only)',
        freeze='text encoder + VAE always; all video in action phase; video except blocks 16,17 in partial_joint',
        model_source=('Released GR1 backbone plus explicit action warm-start; see lineage' if lineage else
            'Released GR1 initialization; target normalization fit on training episodes'),
        gradient_routing='upstream stop-gradient retained',split=split,warm_start=lineage,training_schedule=schedule,
        sampling=schedule.get('sampling',dict(protocol='legacy',description=split.get('sampling','uniform training frame')))))
    completed=[]
    for phase,steps in phases:
        if resume and phases.index((phase,steps))<next(i for i,(p,_) in enumerate(phases) if p==resume['phase']):continue
        if steps<=0:continue
        params=configure(model,phase,video_parameters_fp32=args.video_parameters_fp32);trained_names.update(params)
        groups=[]
        for prefix,lr in [('action_model.',1e-4),('backbone_interface.',1e-5)]:
            ps=[p for n,p in params.items() if n.startswith(prefix)]
            if ps:groups.append({'params':ps,'lr':lr,'name':prefix})
        optimizer=torch.optim.AdamW(groups,betas=(.9,.95),eps=1e-8,weight_decay=1e-8,foreach=False)
        warmup=min(args.warmup_steps,steps) if args.warmup_steps is not None else min(20,max(1,steps//10))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min((s+1)/warmup,1.)*(.1+.9*.5*(1+math.cos(math.pi*max(0,s-warmup)/max(1,steps-warmup)))))
        phase_start=0
        continuation=None
        if resume and resume['phase']==phase:
            phase_start=resume['phase_step'];optimizer.load_state_dict(resume['optimizer'])
            continuation=resume.get('continuation_schedule')
            if continuation:
                assert phase=='action' and continuation['end_step']==steps
            elif args.extend_action_schedule:
                assert phase_start < steps
                continuation=dict(start_step=phase_start,end_step=steps,initial_lrs=[g['lr'] for g in optimizer.param_groups],
                    peak_lrs=[3e-5 for g in optimizer.param_groups],warmup_steps=20,final_fraction=.1)
            else:
                scheduler.load_state_dict(resume['scheduler'])
            rng.bit_generator.state=resume['sampler_rng'];torch.set_rng_state(resume['torch_rng'])
            if args.cpu_preflight:assert resume['cuda_rng']==[]
            else:torch.cuda.set_rng_state_all(resume['cuda_rng'])
            resume=None
        if continuation:atomic_json(out/'continuation_schedule.json',continuation)
        atomic_json(out/f'{phase}_trainable.json',{n:list(p.shape) for n,p in params.items()})
        print(json.dumps({'event':'phase_start','phase':phase,'trainable_parameters':sum(p.numel() for p in params.values()),'steps':steps}),flush=True)
        # Fixed-noise held-out action FM proxy. Current RGB only, no future-label leakage.
        def evaluate():
            model.action_model.eval();video=model.backbone_interface.extractor.transformer;was_training=video.training;video.eval()
            values=[]
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(123)
                if not args.cpu_preflight:torch.cuda.manual_seed_all(123)
                with torch.no_grad(),precision_context():
                    for idx in val_indices:
                        b=batch_for(ds,idx,'action')
                        values.append(float(model(examples=b)['action_loss']))
            model.action_model.train();video.train(was_training)
            value=float(np.mean(values));assert math.isfinite(value)
            return value
        baseline=evaluate();print(json.dumps({'event':'validation','phase':phase,'phase_step':phase_start,'action_fm_loss':baseline}),flush=True)
        def save(phase_step):
            payload=dict(format_version=1,base_checkpoint=str(args.checkpoint.resolve()),phase=phase,phase_step=phase_step,global_step=global_step,
                trained_state={n:p.detach().cpu() for n,p in model.named_parameters() if n in trained_names},optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict() if continuation is None else None,continuation_schedule=continuation,sampler_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=[] if args.cpu_preflight else torch.cuda.get_rng_state_all(),split=split,warm_start=lineage,training_schedule=schedule)
            target=out/f'{phase}_latest.pt';temp=target.with_suffix('.tmp');torch.save(payload,temp);temp.replace(target)
            # Read it back to detect incomplete writes and verify representative trained tensors.
            check=torch.load(target,map_location='cpu',weights_only=False)
            for name in [next(iter(params)),next(reversed(params))]:
                assert torch.equal(check['trained_state'][name],params[name].detach().cpu())
            assert check['global_step']==global_step
            if global_step in args.snapshot_steps:
                snapshot=out/f'{phase}_step_{global_step:06d}.pt'
                if snapshot.exists():raise FileExistsError(snapshot)
                shutil.copyfile(target,snapshot)
            atomic_json(out/'latest.json',dict(path=str(target),phase=phase,phase_step=phase_step,global_step=global_step,checkpoint_readback_passed=True))
            print(json.dumps({'event':'checkpoint','path':str(target),'step':global_step}),flush=True)
        if not args.cpu_preflight:torch.cuda.reset_peak_memory_stats()
        for step in range(phase_start+1,steps+1):
            optimizer.zero_grad(set_to_none=True);metrics={};before={}
            if step==phase_start+1:
                for prefix in ['action_model.','backbone_interface.']:
                    names=[n for n,p in params.items() if n.startswith(prefix) and p.ndim>=2]
                    if names:before[names[-1]]=params[names[-1]].detach().clone()
            tick=time.monotonic();sampled_indices=[]
            for micro in range(args.accumulation):
                batch=[]
                for _ in range(args.microbatch_size):
                    index=ds.sample_index(rng) if hasattr(ds,'sample_index') else int(rng.choice(indices))
                    sampled_indices.append(index)
                    batch.extend(batch_for(ds,index,phase))
                with precision_context():
                    losses=forward_training_microbatch(model,batch,paired_ab=args.paired_ab,
                        seed=args.seed,update=global_step+1,micro=micro)
                loss=losses['action_loss']
                if phase=='partial_joint':
                    assert losses['future_video_loss'].requires_grad
                    loss=loss+losses['future_video_loss']
                assert loss.requires_grad and torch.isfinite(loss),loss
                (loss/args.accumulation).backward()
                for name,value in losses.items():metrics[name]=metrics.get(name,0.)+float(value.detach())/args.accumulation
                del losses,loss,batch
            if args.cpu_preflight:
                missing=[name for name,p in params.items() if p.grad is None]
                assert not missing, ('Missing trainable gradients',missing)
                assert all(torch.isfinite(p.grad).all() for p in params.values())
                metrics['trainable_tensors_with_finite_gradients']=len(params)
                for prefix in ['action_model.state_encoder.','action_model.action_encoder.',
                               'action_model.action_decoder.','action_model.model.',
                               'backbone_interface.extractor.transformer.transformer_blocks.16.',
                               'backbone_interface.extractor.transformer.transformer_blocks.17.']:
                    selected=[p for name,p in params.items() if name.startswith(prefix)]
                    if not selected:continue
                    norm=sum(float(p.grad.detach().double().square().sum()) for p in selected)**.5
                    assert norm>0 and math.isfinite(norm),(prefix,norm)
                    metrics[prefix+'raw_gradient_norm']=norm
            grad_norm=(clip_paired_groups(optimizer.param_groups) if args.paired_ab else
                torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True))
            for group in groups:
                if step==phase_start+1:
                    norm=sum(float(p.grad.detach().float().square().sum()) for p in group['params'] if p.grad is not None)**.5
                    assert norm>0 and math.isfinite(norm),(group['name'],norm)
                    metrics[group['name']+'gradient_norm']=norm
            if continuation:
                for group,lr in zip(optimizer.param_groups,continuation_lrs(step,continuation)):group['lr']=lr
            optimizer.step()
            if continuation is None:scheduler.step()
            global_step+=1
            for name,old in before.items():
                delta=float((params[name].detach()-old).abs().max());assert delta>0,(name,delta)
                metrics[name+'/update_max']=delta
            metrics.update(phase=phase,phase_step=step,global_step=global_step,gradient_norm=float(grad_norm),seconds=time.monotonic()-tick,
                peak_allocated_gib=0. if args.cpu_preflight else torch.cuda.max_memory_allocated()/2**30,learning_rates=[g['lr'] for g in optimizer.param_groups])
            if args.paired_ab:metrics['sample_indices']=sampled_indices
            if step%args.validation_interval==0 or step==steps:metrics['validation_action_fm_loss']=evaluate()
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
            if step<=3 or step%10==0:print(json.dumps(metrics),flush=True)
            if step in (1,10) or step%args.checkpoint_interval==0 or step==steps or global_step in args.snapshot_steps:save(step)
            atomic_json(out/'status.json',dict(status='running',**metrics))
            if args.stop_after and global_step>=args.stop_after:
                if step not in (1,10) and step%args.checkpoint_interval and step!=steps and global_step not in args.snapshot_steps:save(step)
                atomic_json(out/'status.json',dict(status='checkpointed_for_resume',global_steps=global_step))
                print('RESUME_GATE_READY',global_step,flush=True)
                return
        completed.append(dict(phase=phase,steps=steps,baseline_validation=baseline,final_validation=evaluate()))
        del optimizer,scheduler,params,groups
        if not args.cpu_preflight:torch.cuda.empty_cache()
    atomic_json(out/'summary.json',dict(status='complete',global_steps=global_step,phases=completed,elapsed_seconds=time.monotonic()-start_time))
    atomic_json(out/'status.json',dict(status='complete',global_steps=global_step))
    print('TRAINING_COMPLETE',json.dumps(completed),flush=True)

if __name__=='__main__':main()
