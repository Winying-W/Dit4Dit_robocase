"""Four-GPU, task-balanced DiT4DiT adaptation with auditable exact resume.

The video backbone remains frozen in this scaling baseline. The complete action
branch, including the state encoder, is trained. Checkpoints retain the evaluator's
incremental-policy format plus each rank's RNG and sampler state.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from DiT4DiT.model.framework.base_framework import baseframework
from scripts.robocasa365.multitask_data import prepare_multitask_dataset
from scripts.robocasa365.train_single_gpu import configure, atomic_json
from scripts.robocasa365.distributed_support import (
    masked_ddp_scale, capture_rng, restore_rng, validate_resume_schedule, ensure_process_group,
)


def state_digest(model):
    digest=hashlib.sha256()
    for name,param in model.named_parameters():
        if param.requires_grad:
            digest.update(name.encode());digest.update(param.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=50000)
    parser.add_argument('--microbatch',type=int,default=4)
    parser.add_argument('--accumulation',type=int,default=4)
    parser.add_argument('--warmup-steps',type=int,default=1000)
    parser.add_argument('--validation-interval',type=int,default=1000)
    parser.add_argument('--checkpoint-interval',type=int,default=1000)
    parser.add_argument('--snapshot-steps',type=int,nargs='*',default=[10000,25000,50000])
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--warm-start',type=Path,help='Explicit compatible Atomic action checkpoint; reset optimizer and local update counter')
    parser.add_argument('--warm-start-run',type=Path,help='Source Atomic run with frozen adapter source hashes; required for warm-start')
    parser.add_argument('--stop-after',type=int)
    parser.add_argument('--stop-file',type=Path)
    parser.add_argument('--cpu-runtime-preflight',action='store_true',
        help='Exercise actual torchrun imports and process-group setup on four CPUs, then exit without loading a model')
    args=parser.parse_args()
    if bool(args.warm_start)!=bool(args.warm_start_run):
        raise ValueError('Warm-start checkpoint and source run must be provided together')
    if args.resume and args.warm_start:raise ValueError('Warm start and exact resume are mutually exclusive')
    if min(args.steps,args.microbatch,args.accumulation,args.warmup_steps,args.validation_interval,args.checkpoint_interval)<1:
        raise ValueError('All step, batch and interval settings must be positive')
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world != 4:
        raise ValueError('This declared experiment requires exactly four distributed ranks')
    if args.cpu_runtime_preflight:
        assert not torch.cuda.is_available(), 'CPU import preflight must hide CUDA devices'
        reused=ensure_process_group('gloo',rank,world)
        # Repeated callers must also be safe, including the upstream logger.
        assert ensure_process_group('gloo',rank,world)
        records=[None]*world
        dist.all_gather_object(records,dict(rank=rank,local_rank=local,reused_import_initialized_group=reused))
        total=torch.tensor(rank+1);dist.all_reduce(total)
        assert total.item()==sum(range(1,world+1))
        if rank==0:
            args.output.mkdir(parents=True,exist_ok=True)
            atomic_json(args.output/'acceptance.json',dict(status='complete',backend='gloo',world_size=world,
                ranks=records,collective_sum=total.item(),scope='Actual trainer torchrun import and idempotent process-group initialization; no model training'))
        dist.barrier();dist.destroy_process_group();return
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    ensure_process_group('nccl',rank,world)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.jsonl').exists() and not args.resume:
        raise ValueError('Existing training output requires explicit --resume')
    random.seed(args.seed+rank);np.random.seed(args.seed+rank)
    torch.manual_seed(args.seed+rank);torch.cuda.manual_seed(args.seed+rank)
    sampler=np.random.default_rng(np.random.SeedSequence([args.seed,rank]))
    local_out=out if rank==0 else out/'rank_data'/f'rank_{rank}'
    local_out.mkdir(parents=True,exist_ok=True)
    dataset,_,validation,split=prepare_multitask_dataset(args.manifest,local_out)
    signatures=[None]*world
    split_signature=hashlib.sha256(json.dumps(split,sort_keys=True).encode()).hexdigest()
    dist.all_gather_object(signatures,split_signature)
    assert len(set(signatures))==1,'Ranks disagree on the training split'
    model=baseframework.from_pretrained(str(args.checkpoint))
    model.backbone_interface.extractor.text_encoder.to(torch.bfloat16)
    model.config.trainer.repeated_diffusion_steps=1
    model.config.datasets.vla_data=OmegaConf.load(local_out/'data_config.yaml')
    lineage=None
    if args.warm_start:
        from scripts.robocasa365.action_warm_start import load_action_warm_start
        lineage=load_action_warm_start(model,args.warm_start,source_run=args.warm_start_run,
            base_checkpoint=args.checkpoint,target_config=model.config.datasets.vla_data,
            target_stats=json.loads((local_out/'normalization.json').read_text()),target_split=split)
    model=model.to(device);params=configure(model,'action')
    assert all(name.startswith('action_model.') for name in params)
    optimizer=torch.optim.AdamW(params.values(),lr=1e-4,betas=(.9,.95),eps=1e-8,
        weight_decay=1e-8,foreach=False)
    warmup=min(args.warmup_steps,args.steps)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:
        min((s+1)/warmup,1.)*(.1+.9*.5*(1+math.cos(math.pi*max(0,s-warmup)/max(1,args.steps-warmup)))))
    schedule=dict(steps=args.steps,world_size=world,microbatch=args.microbatch,
        accumulation=args.accumulation,warmup_steps=args.warmup_steps,seed=args.seed,
        peak_lr=1e-4,final_lr_fraction=.1,phase='action')
    start=0;resume_rng=None
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        lineage=saved.get('warm_start')
        if lineage:schedule['initialization_checkpoint_sha256']=lineage['checkpoint_sha256']
        validate_resume_schedule(saved['training_schedule'],schedule)
        assert saved['base_checkpoint']==str(args.checkpoint.resolve()) and saved['split']==split
        assert saved['normalization_sha256']==hashlib.sha256((local_out/'normalization.json').read_bytes()).hexdigest()
        assert set(saved['trained_state'])==set(params)
        result=model.load_state_dict(saved['trained_state'],strict=False);assert not result.unexpected_keys
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        start=saved['global_step'];resume_rng=saved['distributed_rng'][rank]
        del saved
    elif lineage:
        schedule['initialization_checkpoint_sha256']=lineage['checkpoint_sha256']
    distributed=DDP(model,device_ids=[local],output_device=local,broadcast_buffers=False,
        find_unused_parameters=False,gradient_as_bucket_view=True)
    if resume_rng is not None:restore_rng(resume_rng,sampler,device)
    sampler_probes=[None]*world
    probe=np.random.default_rng();probe.bit_generator.state=sampler.bit_generator.state
    dist.all_gather_object(sampler_probes,[dataset.sample_index(probe) for _ in range(8)])
    assert len({json.dumps(p) for p in sampler_probes})==world,'Ranks repeat the same sampler stream'
    devices=[None]*world
    props=torch.cuda.get_device_properties(local)
    dist.all_gather_object(devices,dict(rank=rank,local_rank=local,name=props.name,
        total_bytes=props.total_memory,uuid=str(getattr(props,'uuid','unavailable'))))
    assert sorted(row['local_rank'] for row in devices)==list(range(world))
    if all(row['uuid']!='unavailable' for row in devices):assert len({row['uuid'] for row in devices})==world
    if rank==0:
        atomic_json(out/'run_config.json',dict(base_checkpoint=str(args.checkpoint.resolve()),
            phases=[['action',args.steps]],training_schedule=schedule,global_batch_size=world*args.microbatch*args.accumulation,
            gradient_objective='Global valid-action-element mean across all ranks and accumulation microbatches',
            image_size=[128,128],action_lr=1e-4,seed=args.seed,split=split,warm_start=lineage,
            freeze='Cosmos Video DiT, VAE and Qwen frozen; all action_model parameters trainable',
            model_source=('Released GR1 video/VAE/text with compatible Atomic action warm-start; local updates start at zero'
                if lineage else 'Released GR1 initialization; no old step2000 warm-start')))
        atomic_json(out/'action_trainable.json',{name:list(p.shape) for name,p in params.items()})
        atomic_json(out/'distributed_preflight.json',dict(world_size=world,sampler_probes=sampler_probes,
            split_signatures=signatures,devices=devices,trainable_parameters=sum(p.numel() for p in params.values()),
            trainable_tensors=len(params),resume_step=start))
        source=Path(__file__).read_bytes();sha=hashlib.sha256(source).hexdigest()
        (out/f'trainer_{sha[:12]}.py').write_bytes(source)
        atomic_json(out/'trainer_source.json',dict(sha256=sha,path=str(Path(__file__).resolve())))

    def example(index):
        value=dataset[index];value['image']=value['image'][:1]
        return value

    def evaluate():
        model.action_model.eval();totals=torch.zeros(2,dtype=torch.float64,device=device)
        with torch.random.fork_rng(devices=[local]),torch.no_grad():
            for global_index in range(rank,len(validation),world):
                torch.manual_seed(123+global_index);torch.cuda.manual_seed(123+global_index)
                loss=model(examples=[example(validation[global_index])])['action_loss']
                assert torch.isfinite(loss),'Non-finite validation loss'
                totals[0]+=loss.double();totals[1]+=1
        dist.all_reduce(totals)
        model.action_model.train()
        return float((totals[0]/totals[1]).item())

    def save(step):
        rng_states=[None]*world
        dist.all_gather_object(rng_states,capture_rng(sampler,device))
        if step in (1,10) or step==start+1:
            hashes=[None]*world;dist.all_gather_object(hashes,state_digest(model))
            assert len(set(hashes))==1,'Trainable model weights differ between ranks'
            if rank==0:atomic_json(out/f'rank_weights_step_{step:06d}.json',dict(step=step,rank_sha256=hashes,passed=True))
        if rank==0:
            payload=dict(format_version=2,base_checkpoint=str(args.checkpoint.resolve()),phase='action',
                phase_step=step,global_step=step,trained_state={name:p.detach().cpu() for name,p in params.items()},
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),distributed_rng=rng_states,
                training_schedule=schedule,split=split,warm_start=lineage,
                normalization_sha256=hashlib.sha256((out/'normalization.json').read_bytes()).hexdigest())
            path=out/'action_latest.pt';temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)
            check=torch.load(path,map_location='cpu',weights_only=False)
            assert check['global_step']==step and len(check['distributed_rng'])==world
            for name in [next(iter(params)),next(reversed(params))]:
                assert torch.equal(check['trained_state'][name],params[name].detach().cpu())
            if step in args.snapshot_steps:
                snapshot=out/f'action_step_{step:06d}.pt'
                if snapshot.exists():raise FileExistsError(snapshot)
                shutil.copyfile(path,snapshot)
            atomic_json(out/'latest.json',dict(path=str(path),phase='action',phase_step=step,global_step=step,
                world_size=world,checkpoint_readback_passed=True))
            print('DISTRIBUTED_CHECKPOINT',step,flush=True)
        dist.barrier()

    baseline=evaluate()
    if rank==0:print(json.dumps(dict(event='validation',global_step=start,action_fm_loss=baseline)),flush=True)
    torch.cuda.reset_peak_memory_stats()
    for step in range(start+1,args.steps+1):
        tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
        before={}
        if step==start+1:
            matrices=[name for name,p in params.items() if p.ndim>=2]
            before={name:params[name].detach().clone() for name in [matrices[0],matrices[-1]]}
        batches=[[example(dataset.sample_index(sampler)) for _ in range(args.microbatch)] for _ in range(args.accumulation)]
        counts=[int(sum(np.asarray(ex['action_mask']).sum() for ex in batch)) for batch in batches]
        global_count=torch.tensor(sum(counts),dtype=torch.float64,device=device);dist.all_reduce(global_count)
        local_numerator=torch.zeros((),dtype=torch.float64,device=device)
        for micro,(batch,count) in enumerate(zip(batches,counts)):
            context=distributed.no_sync() if micro+1<args.accumulation else nullcontext()
            with context:
                loss=distributed(examples=batch)['action_loss']
                assert loss.requires_grad and torch.isfinite(loss),'Invalid distributed action loss'
                (loss*masked_ddp_scale(count,global_count.item(),world)).backward()
            local_numerator+=loss.detach().double()*count
        del batches,loss
        encoder_gradients=None
        if step==start+1:
            missing=[name for name,p in params.items() if p.grad is None]
            assert not missing, f'Trainable tensors have no gradient: {missing}'
            encoder_gradients={}
            for group in ['state_encoder','action_encoder','action_decoder']:
                squared=[p.grad.detach().float().square().sum() for name,p in params.items()
                         if name.startswith('action_model.'+group+'.')]
                assert squared, group
                norm=float(torch.stack(squared).sum().sqrt())
                assert math.isfinite(norm) and norm>0,(group,norm)
                encoder_gradients[group]=norm
        gradient_norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True)
        assert gradient_norm>0,'Zero action gradient'
        optimizer.step();scheduler.step()
        update_max=max((float((params[name].detach()-old).abs().max()) for name,old in before.items()),default=None)
        if before:assert update_max>0,'Optimizer did not change the checked action tensors'
        dist.all_reduce(local_numerator)
        timings=torch.tensor([time.monotonic()-tick,torch.cuda.max_memory_allocated()/2**30],device=device)
        dist.all_reduce(timings,op=dist.ReduceOp.MAX)
        metrics=dict(phase='action',phase_step=step,global_step=step,action_loss=float((local_numerator/global_count).item()),
            gradient_norm=float(gradient_norm),seconds=float(timings[0]),peak_allocated_gib=float(timings[1]),
            learning_rates=[g['lr'] for g in optimizer.param_groups],world_size=world,
            global_batch_size=world*args.microbatch*args.accumulation,valid_action_elements=int(global_count.item()))
        if before:metrics['checked_parameter_update_max']=update_max
        if encoder_gradients is not None:metrics['encoder_decoder_gradient_norms']=encoder_gradients
        if step%args.validation_interval==0 or step==args.steps:metrics['validation_action_fm_loss']=evaluate()
        stopped=torch.tensor(int(bool(args.stop_file and args.stop_file.exists())),device=device)
        dist.all_reduce(stopped,op=dist.ReduceOp.MAX)
        stop=bool(stopped.item()) or bool(args.stop_after and step>=args.stop_after)
        if step in (1,10) or step==start+1 or step%args.checkpoint_interval==0 or step in args.snapshot_steps or step==args.steps or stop:
            save(step)
        if rank==0:
            with (out/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(metrics)+'\n')
            atomic_json(out/'status.json',dict(status='running',**metrics))
            if step<=3 or step%10==0:print(json.dumps(metrics),flush=True)
        if stop:
            if rank==0:atomic_json(out/'status.json',dict(status='checkpointed_for_resume',global_step=step,world_size=world))
            dist.destroy_process_group();return
    if rank==0:
        atomic_json(out/'summary.json',dict(status='complete',global_steps=args.steps,world_size=world,
            final_validation_action_fm_loss=metrics['validation_action_fm_loss']))
        atomic_json(out/'status.json',dict(status='complete',global_steps=args.steps,world_size=world))
    dist.destroy_process_group()


if __name__=='__main__':
    main()
