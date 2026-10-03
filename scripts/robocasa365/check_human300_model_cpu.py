"""Bounded full-resolution real-example CPU forward; does not claim GPU/backward acceptance."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from omegaconf import OmegaConf
from DiT4DiT.model.framework import build_framework
from DiT4DiT.dataloader.robocasa365_datasets import Robocasa365DatasetAdapter


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=8);a=p.parse_args()
    torch.set_num_threads(a.threads);torch.manual_seed(42)
    manifest=json.loads(a.manifest.read_text());row=manifest['tasks'][0]
    cfg=OmegaConf.load(a.config)
    data=OmegaConf.create(dict(dataset_path=row['path'],action_horizon=16,max_state_dim=64,max_action_dim=32,image_size=[224,224],video_delta_indices=list(range(17)),action_video_freq_ratio=2,video_backend='decord',lerobot_version='v2.0',statistics_path=str(a.manifest.parent/'normalization.json')))
    cfg.datasets.vla_data=data
    ds=Robocasa365DatasetAdapter(row['path'],data)
    allowed=set(row['train_episodes']);index=next(i for i,(ep,t) in enumerate(ds.dataset.all_steps) if int(ep) in allowed and int(t)==20)
    example=ds[index];print('REAL_EXAMPLE_READY',row['task'],index,flush=True)
    model=build_framework(cfg).float().requires_grad_(False).eval();print('MODEL_READY_FP32_CPU',flush=True)
    start=time.monotonic()
    with torch.inference_mode():result=model(examples=[example])
    losses={k:float(v) for k,v in result.items()}
    assert set(losses)=={'action_loss','future_video_loss'} and all(np.isfinite(v) for v in losses.values()),losses
    report=dict(passed=True,scope='Real full-resolution forward with frozen FP32 CPU model, original Cosmos and random action branch; no backward, optimizer, or GPU acceptance',losses=losses,seconds=time.monotonic()-start,task=row['task'],image_shapes=[list(i.shape) for i in example['image']],threads=a.threads)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
