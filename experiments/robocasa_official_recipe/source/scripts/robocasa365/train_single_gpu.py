"""Bounded single-task experiment: frozen video backbone, then partial video thaw.

Uses the released model's forward/loss unchanged. Stores cumulative trained weights,
optimizer, scheduler, RNG and immutable base reference for resumable checkpoints.
"""
import argparse
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


def configure(model,phase):
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


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',type=Path);ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--action-steps',type=int,default=300)
    ap.add_argument('--joint-steps',type=int,default=100);ap.add_argument('--accumulation',type=int,default=4)
    ap.add_argument('--stop-after',type=int);ap.add_argument('--resume',type=Path);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--extend-action-schedule',action='store_true',help='Explicit 20-step ramp from checkpoint LR to 3e-5, then cosine to 3e-6; preserve AdamW state')
    ap.add_argument('--snapshot-steps',type=int,nargs='*',default=[])
    ap.add_argument('--overfit-episode',type=int,help='Train one original train episode, keeping original training statistics and validation split')
    ap.add_argument('--warm-start',type=Path,help='Load action weights only; reset optimizer, scheduler, RNG and local update counter')
    ap.add_argument('--manifest',type=Path,help='Audited 16-task manifest; mutually exclusive with --dataset')
    ap.add_argument('--validation-interval',type=int,default=50)
    ap.add_argument('--checkpoint-interval',type=int,default=50)
    ap.add_argument('--warmup-steps',type=int,help='Explicit LR warmup for a longer training schedule; otherwise preserve the original short-run warmup')
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.accumulation < 1 or args.validation_interval < 1 or args.checkpoint_interval < 1:
        raise ValueError('Accumulation and logging/checkpoint intervals must be positive')
    if args.warmup_steps is not None and args.warmup_steps < 1:
        raise ValueError('Warmup must be positive')
    schedule=dict(action_steps=args.action_steps,joint_steps=args.joint_steps,
        accumulation=args.accumulation,warmup_steps=args.warmup_steps)
    if args.warm_start and args.resume:raise ValueError('Warm start and exact resume are mutually exclusive')
    if args.overfit_episode is not None and (args.joint_steps!=0 or not (args.warm_start or args.resume)):
        raise ValueError('Overfit requires frozen video and explicit warm-start/resume')
    if (out/'metrics.jsonl').exists() and args.resume is None:raise ValueError('Existing run: use --resume or a new output directory')
    source=Path(__file__).read_bytes()
    source_hash=hashlib.sha256(source).hexdigest()
    (out/f'trainer_{source_hash[:12]}.py').write_bytes(source)
    atomic_json(out/'trainer_source.json',dict(sha256=source_hash,path=str(Path(__file__).resolve())))
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    rng=np.random.default_rng(args.seed)
    if bool(args.manifest) == bool(args.dataset):
        raise ValueError('Specify exactly one of --dataset or --manifest')
    if args.manifest:
        if args.warm_start or args.overfit_episode is not None:
            raise ValueError('Changed multitask normalization requires released-base initialization, not fixed-base warm start')
        from scripts.robocasa365.multitask_data import prepare_multitask_dataset
        ds,indices,val_indices,split=prepare_multitask_dataset(args.manifest,out)
    else:
        ds,indices,val_indices,split=prepare_dataset(args.dataset,out,args.seed,args.overfit_episode)
    model=baseframework.from_pretrained(str(args.checkpoint))
    model.backbone_interface.extractor.text_encoder.to(torch.bfloat16)
    model.requires_grad_(False);model=model.to('cuda').eval()
    model.config.trainer.repeated_diffusion_steps=1
    model.config.datasets.vla_data=OmegaConf.load(out/'data_config.yaml')
    trained_names=set();start_time=time.monotonic();global_step=0
    resume=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
    lineage=resume.get('warm_start') if resume else None
    if args.warm_start:
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
        atomic_json(out/'warm_start.json',lineage)
        del warm
    if resume:
        assert resume['base_checkpoint']==str(args.checkpoint.resolve()) and resume['split']==split
        assert json.loads((args.resume.parent/'normalization.json').read_text()) == json.loads((out/'normalization.json').read_text())
        if 'training_schedule' in resume and not args.extend_action_schedule:
            assert resume['training_schedule'] == schedule, 'Exact resume requires the same LR schedule and accumulation'
    if args.extend_action_schedule and (not resume or resume['phase']!='action' or args.joint_steps!=0):
        raise ValueError('LR extension requires an action checkpoint and joint_steps=0')
    if resume:
        result=model.load_state_dict(resume['trained_state'],strict=False);assert not result.unexpected_keys
        trained_names.update(resume['trained_state']);global_step=resume['global_step']
    phases=[('action',args.action_steps),('partial_joint',args.joint_steps)]
    atomic_json(out/'run_config.json',dict(base_checkpoint=str(args.checkpoint.resolve()),phases=phases,accumulation=args.accumulation,
        image_size=[128,128],video_lr=1e-5,action_lr=1e-4,seed=args.seed,loss='action + video (partial_joint only)',
        freeze='text encoder + VAE always; all video in action phase; video except blocks 16,17 in partial_joint',
        model_source='released GR1 initialization; target normalization fit on training episodes',
        gradient_routing='upstream stop-gradient retained',split=split,warm_start=lineage,training_schedule=schedule))
    completed=[]
    for phase,steps in phases:
        if resume and phases.index((phase,steps))<next(i for i,(p,_) in enumerate(phases) if p==resume['phase']):continue
        if steps<=0:continue
        params=configure(model,phase);trained_names.update(params)
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
            rng.bit_generator.state=resume['sampler_rng'];torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])
            resume=None
        if continuation:atomic_json(out/'continuation_schedule.json',continuation)
        atomic_json(out/f'{phase}_trainable.json',{n:list(p.shape) for n,p in params.items()})
        print(json.dumps({'event':'phase_start','phase':phase,'trainable_parameters':sum(p.numel() for p in params.values()),'steps':steps}),flush=True)
        # Fixed-noise held-out action FM proxy. Current RGB only, no future-label leakage.
        def evaluate():
            model.action_model.eval();video=model.backbone_interface.extractor.transformer;was_training=video.training;video.eval()
            values=[]
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(123);torch.cuda.manual_seed_all(123)
                with torch.no_grad():
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
                scheduler=scheduler.state_dict() if continuation is None else None,continuation_schedule=continuation,sampler_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),split=split,warm_start=lineage,training_schedule=schedule)
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
        torch.cuda.reset_peak_memory_stats()
        for step in range(phase_start+1,steps+1):
            optimizer.zero_grad(set_to_none=True);metrics={};before={}
            if step==phase_start+1:
                for prefix in ['action_model.','backbone_interface.']:
                    names=[n for n,p in params.items() if n.startswith(prefix) and p.ndim>=2]
                    if names:before[names[-1]]=params[names[-1]].detach().clone()
            tick=time.monotonic()
            for micro in range(args.accumulation):
                index=ds.sample_index(rng) if hasattr(ds,'sample_index') else rng.choice(indices)
                batch=batch_for(ds,index,phase)
                losses=model(examples=batch)
                loss=losses['action_loss']
                if phase=='partial_joint':
                    assert losses['future_video_loss'].requires_grad
                    loss=loss+losses['future_video_loss']
                assert loss.requires_grad and torch.isfinite(loss),loss
                (loss/args.accumulation).backward()
                for name,value in losses.items():metrics[name]=metrics.get(name,0.)+float(value.detach())/args.accumulation
                del losses,loss,batch
            grad_norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True)
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
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,learning_rates=[g['lr'] for g in optimizer.param_groups])
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
        torch.cuda.empty_cache()
    atomic_json(out/'summary.json',dict(status='complete',global_steps=global_step,phases=completed,elapsed_seconds=time.monotonic()-start_time))
    atomic_json(out/'status.json',dict(status='complete',global_steps=global_step))
    print('TRAINING_COMPLETE',json.dumps(completed),flush=True)

if __name__=='__main__':main()
