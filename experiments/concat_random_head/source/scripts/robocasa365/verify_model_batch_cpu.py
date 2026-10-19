"""Real released-model multi-task forward before queuing GPU resources.

CPU bfloat16 autocast verifies model/batch compatibility. It does not certify CUDA
kernels, GPU memory, or NCCL; those remain the allocated four-GPU short gate.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from DiT4DiT.model.framework.base_framework import baseframework
from scripts.robocasa365.multitask_data import prepare_multitask_dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--backward',action='store_true',help='Also check all trainable action tensors receive finite gradients; no optimizer step')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if torch.cuda.is_available():raise RuntimeError('Set CUDA_VISIBLE_DEVICES empty for this CPU-only preflight')
    torch.set_num_threads(8);torch.set_num_interop_threads(2)
    dataset,_,validation,split=prepare_multitask_dataset(a.manifest,a.output)
    examples=[];tasks=[]
    for i in range(a.batch_size):
        task=i%len(dataset.datasets)
        item=dataset[[task,split['tasks'][task]['validation_indices'][0]]]
        item['image']=item['image'][:1];examples.append(item);tasks.append(split['tasks'][task]['task'])
    print('CPU_BATCH_ASSEMBLED',tasks,flush=True)
    model=baseframework.from_pretrained(str(a.checkpoint))
    model.requires_grad_(False);model.eval()
    if a.backward:model.action_model.requires_grad_(True);model.action_model.train()
    model.config.framework.cosmos25.training='action'
    model.config.trainer.repeated_diffusion_steps=1
    model.config.datasets.vla_data=OmegaConf.load(a.output/'data_config.yaml')
    assert all(p.device.type=='cpu' for p in model.parameters())
    torch.manual_seed(123)
    started=time.monotonic();print('CPU_FULL_MODEL_FORWARD_START',flush=True)
    with torch.set_grad_enabled(a.backward),torch.autocast('cpu',dtype=torch.bfloat16):
        result=model(examples=examples)
    loss=float(result['action_loss'].detach());assert np.isfinite(loss)
    gradients=None
    if a.backward:
        result['action_loss'].backward()
        trainable={name:p for name,p in model.named_parameters() if p.requires_grad}
        missing=[name for name,p in trainable.items() if p.grad is None]
        assert not missing, f'Trainable tensors unused by the loss; DDP configuration needs review: {missing}'
        assert all(torch.isfinite(p.grad).all() for p in trainable.values())
        groups={}
        for group in ['state_encoder','action_encoder','action_decoder']:
            norm=sum(float(p.grad.double().square().sum()) for name,p in trainable.items() if name.startswith('action_model.'+group+'.'))**.5
            assert np.isfinite(norm) and norm>0,(group,norm)
            groups[group]=norm
        gradients=dict(trainable_tensors=len(trainable),trainable_parameters=sum(p.numel() for p in trainable.values()),
            missing_gradients=missing,all_gradients_finite=True,encoder_decoder_gradient_norms=groups)
    report=dict(status='complete',device='cpu',batch_size=len(examples),tasks=tasks,
        action_fm_loss=loss,forward_seconds=time.monotonic()-started,
        strict_released_checkpoint_load=True,real_dataset_examples=True,gradients=gradients,
        scope='Actual DiT4DiT/Cosmos/action forward, optionally backward, with CPU bfloat16 autocast; no optimizer update. GPU memory, CUDA kernels and NCCL still require the short cloud gate.')
    (a.output/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    print('CPU_FULL_MODEL_BATCH_PASSED',json.dumps(report),flush=True)


if __name__=='__main__':main()
