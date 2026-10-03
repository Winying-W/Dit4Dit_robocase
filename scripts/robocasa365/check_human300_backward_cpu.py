"""One isolated FP32 CPU joint update, using the formal trainer's trainable groups."""
import argparse
import json
from pathlib import Path
import time
import torch
from omegaconf import OmegaConf
from DiT4DiT.model.framework import build_framework
from DiT4DiT.dataloader.robocasa365_datasets import Robocasa365DatasetAdapter
from scripts.robocasa365.train_human300 import configure_joint, lr_factor


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=8);a=p.parse_args()
    torch.set_num_threads(a.threads);torch.manual_seed(42)
    manifest=json.loads(a.manifest.read_text());row=manifest['tasks'][0]
    cfg=OmegaConf.load(a.config);cfg.datasets.vla_data=OmegaConf.create(dict(dataset_path=row['path'],action_horizon=16,max_state_dim=64,max_action_dim=32,image_size=[224,224],video_delta_indices=list(range(17)),action_video_freq_ratio=2,video_backend='decord',lerobot_version='v2.0',statistics_path=str(a.manifest.parent/'normalization.json')))
    ds=Robocasa365DatasetAdapter(row['path'],cfg.datasets.vla_data);allowed=set(row['train_episodes']);index=next(i for i,(ep,t) in enumerate(ds.dataset.all_steps) if int(ep) in allowed and int(t)==20);example=ds[index]
    model=build_framework(cfg).float();video,params,groups=configure_joint(model)
    opt=torch.optim.AdamW(groups,betas=(.9,.95),eps=1e-8,weight_decay=1e-8,foreach=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:lr_factor(s,10000,500))
    print('CPU_BACKWARD_MODEL_READY',flush=True);tick=time.monotonic()
    loss=model(examples=[example]);assert set(loss)=={'action_loss','future_video_loss'} and all(torch.isfinite(v) and v.requires_grad for v in loss.values())
    (loss['action_loss']+loss['future_video_loss']).backward();print('CPU_BACKWARD_FINISHED',time.monotonic()-tick,flush=True)
    norms=[];probes={}
    for prefix,group in zip(['action_model.','backbone_interface.extractor.transformer.'],groups):
        grads=[p.grad for p in group['params'] if p.grad is not None];norm=torch.stack([g.detach().float().square().sum() for g in grads]).sum().sqrt();assert torch.isfinite(norm) and norm>0;norms.append(float(norm))
        named=[(n,p) for n,p in params.items() if n.startswith(prefix) and p.grad is not None];name,param=max(named,key=lambda pair:float(pair[1].grad.detach().abs().max()));i=int(param.grad.detach().abs().reshape(-1).argmax());probes[name]=(i,param.detach().reshape(-1)[i].clone())
    frozen=[n for n,p in model.named_parameters() if not p.requires_grad and p.grad is not None];assert not frozen,frozen
    missing=[n for n,p in params.items() if p.grad is None];assert not missing,missing
    torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True);lrs=[g['lr'] for g in opt.param_groups];opt.step();scheduler.step()
    changed={n:float((params[n].detach().reshape(-1)[i]-old).abs()) for n,(i,old) in probes.items()};assert all(v>0 for v in changed.values()),changed
    report=dict(passed=True,scope='One isolated FP32 CPU joint update with full-resolution real data and formal trainable groups; not GPU/DDP/resume acceptance and not part of formal 10k',formal_training_steps=0,diagnostic_updates=1,task=row['task'],losses={k:float(v.detach()) for k,v in loss.items()},gradient_norms=norms,parameter_updates=changed,used_learning_rates=lrs,next_learning_rates=[g['lr'] for g in opt.param_groups],seconds=time.monotonic()-tick,trainable_tensors=len(params))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
