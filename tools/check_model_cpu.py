"""Real CPU action-path forward/backward for baseline then random-head Qwen concat.

Does not test video-loss training, CUDA kernels, GPU memory or NCCL.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/concat_random_head/source'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--cosmos-model',type=Path,required=True)
    p.add_argument('--qwen-model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    os.environ['COSMOS_MODEL_DIR']=str(a.cosmos_model.resolve())
    import torch
    from omegaconf import OmegaConf
    from DiT4DiT.model.framework.base_framework import baseframework
    from DiT4DiT.dataloader.robocasa365_datasets import Robocasa365DatasetAdapter
    from scripts.robocasa365.multitask_data import apply_statistics
    from scripts.robocasa365.head_init import reinitialize_action_head
    from DiT4DiT.model.framework.qwen_context import attach_qwen
    if torch.cuda.is_available():raise RuntimeError('Set CUDA_VISIBLE_DEVICES= for this CPU-only test')
    torch.set_num_threads(8)
    m=json.loads(a.manifest.read_text());row=m['tasks'][0]
    cfg=OmegaConf.create(dict(action_horizon=16,max_state_dim=64,max_action_dim=32,image_size=[224,224],
        video_delta_indices=list(range(17)),action_video_freq_ratio=2,video_backend='decord',lerobot_version='v2.0',
        dataset_path=row['path'],statistics_path=str(a.manifest.parent/'normalization.json')))
    ds=Robocasa365DatasetAdapter(Path(row['path']),cfg)
    apply_statistics(ds,Path(row['path']),json.loads((a.manifest.parent/'normalization.json').read_text()))
    example=ds[0];example['image']=example['image'][:1]
    print('CPU_STRICT_GR1_LOAD',flush=True)
    model=baseframework.from_pretrained(str(a.checkpoint))
    model.backbone_interface.extractor.text_encoder.to(torch.bfloat16)
    model.config.datasets.vla_data=cfg
    model.config.framework.cosmos25.training='action'
    model.config.trainer.repeated_diffusion_steps=4
    results={}
    for arm in ['baseline','concat']:
        if arm=='concat':
            record=reinitialize_action_head(model,42)
            attach_qwen(model,a.qwen_model,42)
        model.requires_grad_(False);model.eval();model.action_model.requires_grad_(True);model.action_model.train()
        if arm=='concat':model.qwen_context.projection.requires_grad_(True);model.qwen_context.train()
        model.zero_grad(set_to_none=True);torch.manual_seed(42)
        tick=time.monotonic();print('CPU_FORWARD_BACKWARD',arm,flush=True)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            loss=model(examples=[example])['action_loss']
        if not torch.isfinite(loss):raise ValueError('Non-finite loss')
        loss.backward()
        trainable={n:q for n,q in model.named_parameters() if q.requires_grad}
        missing=[n for n,q in trainable.items() if q.grad is None]
        if missing:raise ValueError(f'Missing gradients: {missing}')
        if not all(torch.isfinite(q.grad).all() for q in trainable.values()):raise ValueError('Non-finite gradient')
        if any(q.grad is not None for q in model.backbone_interface.parameters()):raise ValueError('Action gradient reached Cosmos')
        if arm=='concat' and any(q.grad is not None for q in model.qwen_context.encoder.parameters()):raise ValueError('Qwen is not frozen')
        results[arm]=dict(loss=loss.item(),trainable_tensors=len(trainable),seconds=time.monotonic()-tick,
                          all_gradients_finite=True,backbone_frozen=True)
        print(json.dumps(results[arm]),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(passed=True,scope='CPU action-path only; no video-loss training, optimizer, CUDA, NCCL or simulator',results=results),indent=2)+'\n')

if __name__=='__main__':main()
