"""Human300 joint video/action training from Cosmos and a seeded fresh action branch."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf

from scripts.robocasa365.multitask_data import prepare_multitask_dataset
from scripts.robocasa365.human300_metrics import sampled_action_metrics
from scripts.robocasa365.human300_evaluation import joint_evaluation_mode
from scripts.robocasa365.distributed_support import capture_rng, restore_rng, ensure_process_group, masked_ddp_scale


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def lr_factor(step,total,warmup):
    if step<warmup:return (step+1)/warmup
    return .1+.9*.5*(1+math.cos(math.pi*min(step-warmup,total-warmup)/max(1,total-warmup)))


def configure_joint(model):
    """Configure the exact joint objective shared by preflight and formal training."""
    model.requires_grad_(False);model.eval()
    video=model.backbone_interface.extractor.transformer
    model.action_model.float().requires_grad_(True).train();video.float().requires_grad_(True).train()
    video.enable_gradient_checkpointing()
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    groups=[dict(params=list(model.action_model.parameters()),lr=1e-4),dict(params=list(video.parameters()),lr=1e-5)]
    return video,params,groups


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','config','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight',action='store_true',help='Allow a declared subset for an isolated run of at most 200 steps')
    p.add_argument('--cpu-preflight',action='store_true',help='Isolated FP32 CPU checkpoint integration test; requires --preflight and at most 4 steps')
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--warmup',type=int,default=500)
    p.add_argument('--microbatch',type=int,default=1);p.add_argument('--accumulation',type=int,default=16)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--resume',type=Path)
    p.add_argument('--stop-after',type=int);p.add_argument('--save-every',type=int,default=2000)
    p.add_argument('--validate-every',type=int,default=1000);p.add_argument('--validation-examples',type=int,default=120);p.add_argument('--prediction-examples',type=int,default=16)
    a=p.parse_args()
    if a.cpu_preflight:
        assert a.preflight and a.steps<=4, 'CPU mode is only for bounded diagnostics'
    declared=json.loads(a.manifest.read_text())
    if a.preflight:
        assert a.steps<=200, 'Preflight must be bounded'
    else:
        assert declared['task_set']=='pretrain_human300' and declared['split']=='pretrain' and len(declared['tasks'])==300, 'Formal training requires the entire declared human300 task set'
        assert declared.get('episode_selection', {}).get('filter_key') == '100_demos'
        assert declared['episode_selection']['filter_key_seed'] == 0
        assert all(len(r['train_episodes']) == 95 and len(r['validation_episodes']) == 5 for r in declared['tasks'])
    assert 0<a.warmup<=a.steps and min(a.microbatch,a.accumulation,a.save_every,a.validate_every,a.validation_examples,a.prediction_examples)>0
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if not a.preflight:
        assert world in (1,2,4) and world*a.microbatch*a.accumulation==64, 'Formal run uses available 1/2/4 GPUs and global batch 64'
    # Select a backend before the framework logger initializes Accelerate's shared state.
    from accelerate import PartialState
    PartialState(cpu=a.cpu_preflight,backend='gloo' if a.cpu_preflight else 'nccl')
    from DiT4DiT.model.framework import build_framework
    if a.cpu_preflight:
        device=torch.device('cpu');ensure_process_group('gloo',rank,world)
    else:
        torch.cuda.set_device(local);device=torch.device('cuda',local);ensure_process_group('nccl',rank,world)
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.jsonl').exists() and not a.resume:raise FileExistsError('Use --resume for existing run')
    local_out=out if rank==0 else out/'rank_data'/str(rank);local_out.mkdir(parents=True,exist_ok=True)
    # Rank zero populates immutable dataset index caches before the other ranks read them.
    if rank!=0:dist.barrier()
    ds,_,validation,split=prepare_multitask_dataset(a.manifest,local_out)
    if rank==0:dist.barrier()
    signatures=[None]*world
    dist.all_gather_object(signatures,hashlib.sha256(json.dumps(split,sort_keys=True).encode()).hexdigest())
    assert len(set(signatures))==1, 'Ranks disagree on the data split'
    # Identical initialization across ranks; rank-specific sampler/noise begins only after construction.
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    cfg=OmegaConf.load(a.config);cfg.datasets.vla_data=OmegaConf.load(local_out/'data_config.yaml')
    cfg.framework.cosmos25.training='joint';cfg.trainer.repeated_diffusion_steps=1
    cfg.trainer.gradient_accumulation_steps=a.accumulation
    cfg.datasets.vla_data.per_device_batch_size=a.microbatch
    base=Path(cfg.framework.cosmos25.base_model)
    receipt=base/'CONVERSION_VERIFIED.json'
    identity=hashlib.sha256(receipt.read_bytes()).hexdigest()
    if rank==0:
        for rel,record in json.loads(receipt.read_text())['files'].items():
            file=base/rel;assert file.stat().st_size==record['bytes'],file
            digest=hashlib.sha256()
            with file.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
            assert digest.hexdigest()==record['sha256'],file
        (out/'initialization_receipt.json').write_bytes(receipt.read_bytes())
    dist.barrier()
    model=build_framework(cfg)
    if a.cpu_preflight:model.float()
    video,params,groups=configure_joint(model)
    model.to(device)
    optimizer=torch.optim.AdamW(groups,betas=(.9,.95),eps=1e-8,weight_decay=1e-8,foreach=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:lr_factor(s,a.steps,a.warmup))
    sampler=np.random.default_rng(np.random.SeedSequence([a.seed,rank]))
    random.seed(a.seed+rank);np.random.seed(a.seed+rank);torch.manual_seed(a.seed+rank);torch.cuda.manual_seed(a.seed+rank)
    schedule=dict(steps=a.steps,warmup=a.warmup,world_size=world,microbatch=a.microbatch,accumulation=a.accumulation,seed=a.seed,validation_examples=a.validation_examples,prediction_examples=a.prediction_examples,
        action_lr=1e-4,video_lr=1e-5,phase='joint',preflight=a.preflight,cpu_preflight=a.cpu_preflight,config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),initialization_sha256=identity,
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path('scripts/robocasa365/multitask_data.py'),Path('DiT4DiT/dataloader/robocasa365_datasets.py'),Path('scripts/robocasa365/human300_evaluation.py'),Path('scripts/robocasa365/human300_metrics.py')]})
    start=0;restore=None
    if a.resume:
        saved=torch.load(a.resume,map_location='cpu',weights_only=False)
        assert saved['schedule']==schedule and saved['split']==split
        assert set(saved['trained_state'])==set(params)
        result=model.load_state_dict(saved['trained_state'],strict=False);assert not result.unexpected_keys
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);start=saved['step'];restore=saved['rng'][rank]
        del saved
    wrapped=DDP(model,device_ids=None if a.cpu_preflight else [local],broadcast_buffers=False,gradient_as_bucket_view=True,find_unused_parameters=False)
    if restore:restore_rng(restore,sampler,None if a.cpu_preflight else device)
    probe=np.random.default_rng(0);probe.bit_generator.state=sampler.bit_generator.state
    sampler_probes=[None]*world;dist.all_gather_object(sampler_probes,[ds.sample_index(probe) for _ in range(8)])
    assert len({json.dumps(p) for p in sampler_probes})==world, 'Ranks share a sampler stream'
    if rank==0:
        write(out/'distributed_preflight.json',dict(world_size=world,split_signatures=signatures,sampler_probes=sampler_probes,device=str(device)))
        OmegaConf.save(cfg,out/'config.yaml')
        write(out/'run_config.json',dict(schedule=schedule,global_batch=world*a.microbatch*a.accumulation,split=split,
            initialization='Cosmos pretrained components and seed-initialized Action DiT; no GR1 policy loaded',
            trainable={n:list(p.shape) for n,p in params.items()}))
    dist.barrier()

    def save(step):
        rng=[None]*world;dist.all_gather_object(rng,capture_rng(sampler,None if a.cpu_preflight else device))
        if rank==0:
            payload=dict(step=step,schedule=schedule,split=split,trained_state={n:p.detach().cpu() for n,p in params.items()},optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng)
            path=out/f'joint_step_{step:06d}.pt';temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)
            del payload
            check=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            assert check['step']==step and set(check['trained_state'])==set(params)
            for n,param in params.items():assert torch.equal(check['trained_state'][n],param.detach().cpu()),n
            assert check['scheduler']['last_epoch']==step
            assert len(check['optimizer']['state'])==len(params)
            del check
            write(out/'latest.json',dict(path=str(path.resolve()),step=step,checkpoint_readback_passed=True))
        dist.barrier()

    def evaluate(step):
        with joint_evaluation_mode(model):
            return evaluate_in_mode(step)

    def evaluate_in_mode(step):
        totals=torch.zeros(3,dtype=torch.float64,device=device)
        # Evenly sample fixed held-out windows across the entire ordered 300-task list.
        ids=np.linspace(0,len(validation)-1,min(a.validation_examples,len(validation)),dtype=int)
        with torch.random.fork_rng(devices=[] if a.cpu_preflight else [local]),torch.no_grad():
            for j in range(rank,len(ids),world):
                torch.manual_seed(10000+j);torch.cuda.manual_seed(10000+j)
                loss=model(examples=[ds[validation[int(ids[j])]]])
                assert all(torch.isfinite(v) for v in loss.values())
                totals[0]+=loss['action_loss'].double();totals[1]+=loss['future_video_loss'].double();totals[2]+=1
        dist.all_reduce(totals)
        prediction_ids=np.linspace(0,len(validation)-1,min(a.prediction_examples,len(validation)),dtype=int)
        records=[]
        with torch.random.fork_rng(devices=[] if a.cpu_preflight else [local]),torch.no_grad():
            for j in range(rank,len(prediction_ids),world):
                index=validation[int(prediction_ids[j])];task_id,local_index=index
                ex=ds[index];adapter=ds.datasets[task_id]
                torch.manual_seed(20000+j);torch.cuda.manual_seed(20000+j)
                output=model.predict_action(examples=[dict(image=ex['image'][:1],state=ex['state'],lang=ex['lang'])])['normalized_actions'][0]
                records.append(dict(prediction=adapter.decode_actions(output,env_order=False),target=adapter.decode_actions(ex['action'],env_order=False),valid=ex['action_mask'][:,:12],task_id=task_id,index=local_index))
        gathered=[None]*world;dist.all_gather_object(gathered,records)
        report=dict(action=float(totals[0]/totals[2]),video=float(totals[1]/totals[2]),examples=int(totals[2]))
        if rank==0:
            all_records=[r for part in gathered for r in part]
            arrays={key:np.stack([r[key] for r in all_records]) for key in ['prediction','target','valid']}
            report['sampled_actions']=sampled_action_metrics(**arrays)
            np.savez_compressed(out/f'validation_predictions_{step:06d}.npz',**arrays,task_ids=[r['task_id'] for r in all_records],indices=[r['index'] for r in all_records])
        return report

    if not a.cpu_preflight:torch.cuda.reset_peak_memory_stats()
    for step in range(start+1,a.steps+1):
        tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
        batches=[[ds[ds.sample_index(sampler)] for _ in range(a.microbatch)] for _ in range(a.accumulation)]
        counts=[sum(int(np.asarray(ex['action_mask']).sum()) for ex in batch) for batch in batches]
        count=torch.tensor(sum(counts),dtype=torch.float64,device=device);dist.all_reduce(count)
        losses=torch.zeros(2,dtype=torch.float64,device=device)
        probes={}
        for micro,(batch,n) in enumerate(zip(batches,counts)):
            with wrapped.no_sync() if micro+1<a.accumulation else nullcontext():
                loss=wrapped(examples=batch);action=loss['action_loss'];future=loss['future_video_loss']
                assert action.requires_grad and future.requires_grad and torch.isfinite(action) and torch.isfinite(future)
                (action*masked_ddp_scale(n,count.item(),world)+future/a.accumulation).backward()
                losses[0]+=action.detach().double()*n;losses[1]+=future.detach().double()/a.accumulation
        del batches,batch,loss,action,future
        norms=[]
        for group in groups:
            grads=[param.grad for param in group['params'] if param.grad is not None]
            assert grads
            norm=torch.stack([g.detach().float().square().sum() for g in grads]).sum().sqrt()
            assert torch.isfinite(norm) and norm>0;norms.append(float(norm))
        if step==start+1:
            for prefix in ['action_model.','backbone_interface.extractor.transformer.']:
                candidates=[(n,p) for n,p in params.items() if n.startswith(prefix) and p.grad is not None]
                name,param=max(candidates,key=lambda pair:float(pair[1].grad.detach().abs().max()))
                index=int(param.grad.detach().abs().reshape(-1).argmax())
                probes[name]=(index,param.detach().reshape(-1)[index].clone())
        torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True)
        optimizer.step();scheduler.step()
        changes={n:float((params[n].detach().reshape(-1)[index]-old).abs()) for n,(index,old) in probes.items()}
        assert all(v>0 for v in changes.values()),changes
        dist.all_reduce(losses)
        timing=torch.tensor([time.monotonic()-tick,0. if a.cpu_preflight else torch.cuda.max_memory_allocated()/2**30],device=device);dist.all_reduce(timing,op=dist.ReduceOp.MAX)
        metrics=dict(step=step,device=str(device),diagnostic=a.preflight,action_loss=float(losses[0]/count),video_loss=float(losses[1]/world),gradient_norms=norms,
            seconds=float(timing[0]),peak_gib=float(timing[1]),lr=[g['lr'] for g in optimizer.param_groups],parameter_changes=changes)
        if step%a.validate_every==0 or step==a.steps:metrics['validation']=evaluate(step)
        stop=bool(a.stop_after and step>=a.stop_after)
        if step%a.save_every==0 or step==a.steps or stop:save(step)
        if rank==0:
            with (out/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(metrics)+'\n')
            write(out/'status.json',dict(status='complete' if step==a.steps else 'checkpointed' if stop else 'running',**metrics))
            if step<=3 or step%10==0:print(json.dumps(metrics),flush=True)
        if stop:break
    dist.destroy_process_group()


if __name__=='__main__':main()
