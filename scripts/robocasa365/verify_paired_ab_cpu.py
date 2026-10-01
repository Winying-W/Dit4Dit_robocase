"""Real-model A/B input/noise/gradient controls, without optimizer updates.

Uses released GR1 or an explicitly audited same-task Atomic/Composite warm start.
It does not recover or execute the historical step2000 comparison.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from DiT4DiT.dataloader.robocasa365_datasets import Robocasa365DatasetAdapter
from DiT4DiT.model.framework.base_framework import baseframework
from scripts.robocasa365.multitask_data import apply_statistics
from scripts.robocasa365.paired_randomness import clip_paired_groups, PROTOCOL
from scripts.robocasa365.train_single_gpu import configure, forward_training_microbatch


def tensor_hash(tensor):
    value=tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def gradients(parameters):
    assert parameters and all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters.values())
    return {name:tensor_hash(p.grad) for name,p in parameters.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warm-start',type=Path)
    parser.add_argument('--warm-start-run',type=Path)
    parser.add_argument('--warm-start-task-set',choices=['atomic_seen','composite_seen'],default='atomic_seen')
    parser.add_argument('--warm-start-step',type=int)
    parser.add_argument('--warm-start-audit',type=Path)
    parser.add_argument('--task',default='PickPlaceCounterToCabinet')
    parser.add_argument('--video-parameters-fp32',action='store_true')
    args=parser.parse_args()
    if bool(args.warm_start)!=bool(args.warm_start_run):
        raise ValueError('Warm-start checkpoint and source run must be provided together')
    if args.warm_start_task_set=='composite_seen':
        if not (args.warm_start and args.warm_start_run and args.warm_start_audit
                and args.warm_start_step is not None and args.warm_start_step>0):
            raise ValueError('Composite controls require a checkpoint, source run, positive explicit step and audit')
    elif args.warm_start_step is not None or args.warm_start_audit is not None:
        raise ValueError('Step/audit overrides require explicit Composite controls')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_available():raise RuntimeError('CPU only')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8);torch.set_num_interop_threads(2)
    started=time.monotonic()
    config=OmegaConf.load(args.prepared/'data_config.yaml')
    split=json.loads((args.prepared/'split.json').read_text())
    normalization=(args.prepared/'normalization.json').read_bytes()
    if args.warm_start:
        row=next(row for row in split['tasks'] if row['task']==args.task)
        local_config=OmegaConf.merge(config,{'dataset_path':row['path']})
        dataset=Robocasa365DatasetAdapter(row['path'],local_config)
        apply_statistics(dataset,row['path'],json.loads(normalization))
        episode=row['train_episodes'][0]
        indices=[i for i,(ep,t) in enumerate(dataset.dataset.all_steps) if int(ep)==episode]
        index=indices[len(indices)//2]
    else:
        assert hashlib.sha256(normalization).hexdigest()=='77d32db8bb1b30e24ec7680585858118e4bda30a6f3102eb2392ef7d1011408a'
        dataset=Robocasa365DatasetAdapter(config.dataset_path,config)
        apply_statistics(dataset,config.dataset_path,json.loads(normalization))
        index=next(i for i,(episode,step) in enumerate(dataset.dataset.all_steps) if int(episode)==5 and int(step)==100)
        assert 5 in split['train_episodes']
    episode,step=map(int,dataset.dataset.all_steps[index])
    example=dataset[index];assert len(example['image'])==9
    model=baseframework.from_pretrained(str(args.checkpoint))
    model.backbone_interface.extractor.text_encoder.to(torch.bfloat16)
    model.config.datasets.vla_data=config;model.config.trainer.repeated_diffusion_steps=1
    lineage=None
    if args.warm_start:
        common=dict(source_run=args.warm_start_run,
            base_checkpoint=args.checkpoint,target_config=config,target_stats=json.loads(normalization),target_split=split)
        if args.warm_start_task_set=='composite_seen':
            from scripts.robocasa365.composite_continuation import load_composite_action_start
            lineage=load_composite_action_start(model,args.warm_start,checkpoint_audit=args.warm_start_audit,
                expected_step=args.warm_start_step,**common)
        else:
            from scripts.robocasa365.atomic_continuation import load_atomic_action_start
            lineage=load_atomic_action_start(model,args.warm_start,**common)
    captures={};current={}
    def head_hook(module, values):
        for key,value in zip(('features','targets','mask','state'),values):current[key]=value.detach().cpu().clone()
        current['head_pre_hook_rng_sha256']=tensor_hash(torch.get_rng_state())
    def encoder_hook(module, values):
        current['noisy_actions']=values[0].detach().cpu().clone()
        current['action_timesteps']=values[1].detach().cpu().clone()
        current['encoder_pre_hook_rng_sha256']=tensor_hash(torch.get_rng_state())
    handles=[model.action_model.register_forward_pre_hook(head_hook),model.action_model.action_encoder.register_forward_pre_hook(encoder_hook)]
    rows=[];gradient_checks={}
    try:
        for label,phase,paired,alter_future,backward in [
                ('uncontrolled_frozen','action',False,False,False),
                ('uncontrolled_partial','partial_joint',False,False,False),
                ('paired_frozen','action',True,False,True),
                ('paired_partial','partial_joint',True,False,True),
                ('paired_changed_future','partial_joint',True,True,False)]:
            print('AB_CONTROL_FORWARD',label,flush=True)
            current={};model.zero_grad(set_to_none=True)
            parameters=configure(model,phase,video_parameters_fp32=args.video_parameters_fp32)
            action={name:p for name,p in parameters.items() if name.startswith('action_model.')}
            video={name:p for name,p in parameters.items() if name.startswith('backbone_interface.')}
            if args.video_parameters_fp32 and phase=='partial_joint':assert all(p.dtype==torch.float32 for p in video.values())
            batch=dict(example,image=list(example['image'] if phase=='partial_joint' else example['image'][:1]))
            if alter_future:batch['image']=[batch['image'][0],*[torch.zeros_like(frame) for frame in batch['image'][1:]]]
            torch.manual_seed(123)
            with torch.autocast('cpu',dtype=torch.bfloat16):
                losses=forward_training_microbatch(model,[batch],paired_ab=paired,seed=42,update=1,micro=0)
            values={name:float(value.detach()) for name,value in losses.items()}
            assert all(math.isfinite(value) for value in values.values())
            captures[label]=current
            if backward:
                losses['action_loss'].backward()
                assert len(action)==247 and all(p.grad is None for p in video.values())
                raw_action=gradients(action)
                if video:
                    losses['future_video_loss'].backward()
                    assert len(video)==40
                    gradients(video)
                groups=[dict(params=list(action.values()))]
                if video:groups.append(dict(params=list(video.values())))
                clip_paired_groups(groups)
                gradient_checks[label]=dict(raw_action=raw_action,clipped_action=gradients(action),
                    video_tensors=len(video),video_parameter_dtypes=sorted({str(p.dtype) for p in video.values()}),action_loss_updates_video=False)
            rows.append(dict(label=label,phase=phase,paired=paired,altered_future=alter_future,losses=values,
                features_shape=list(current['features'].shape),
                tensors={key:tensor_hash(value) for key,value in current.items() if torch.is_tensor(value)},
                rng={key:value for key,value in current.items() if not torch.is_tensor(value)},
                active_video_dropout=[name for name,module in model.backbone_interface.extractor.transformer.named_modules()
                                      if isinstance(module,torch.nn.Dropout) and module.training and module.p>0]))
            del losses
    finally:
        for handle in handles:handle.remove()
    comparisons=[]
    for first,second in [('uncontrolled_frozen','uncontrolled_partial'),('paired_frozen','paired_partial'),('paired_partial','paired_changed_future')]:
        a,b=captures[first],captures[second];comparison=dict(first=first,second=second,tensors={})
        for name in ('features','targets','mask','state','noisy_actions','action_timesteps'):
            av,bv=a[name],b[name];equal=bool(torch.equal(av,bv))
            comparison['tensors'][name]=dict(equal=equal,max_abs_error=float((av.float()-bv.float()).abs().max()))
        comparisons.append(comparison)
    controlled=all(value['equal'] for pair in comparisons[1:] for value in pair['tensors'].values())
    same_gradients=(gradient_checks['paired_frozen']['raw_action']==gradient_checks['paired_partial']['raw_action'] and
                    gradient_checks['paired_frozen']['clipped_action']==gradient_checks['paired_partial']['clipped_action'])
    future_loss_changed=rows[3]['losses']['future_video_loss']!=rows[4]['losses']['future_video_loss']
    sources=['scripts/robocasa365/verify_paired_ab_cpu.py','scripts/robocasa365/paired_randomness.py',
             'scripts/robocasa365/train_single_gpu.py','DiT4DiT/model/modules/vlm/Cosmos25.py',
             'DiT4DiT/model/framework/DiT4DiT.py','DiT4DiT/model/modules/action_model/ActionDiT.py']
    if lineage:
        sources+=['scripts/robocasa365/action_warm_start.py',
                  'scripts/robocasa365/composite_continuation.py' if args.warm_start_task_set=='composite_seen'
                  else 'scripts/robocasa365/atomic_continuation.py']
    report=dict(status='complete' if controlled and same_gradients and future_loss_changed else 'controls_not_proven',
        completed_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,
        protocol=PROTOCOL,initialization=(f'audited_{args.warm_start_task_set}_step{lineage["global_step"]:06d}' if lineage else 'released_GR1_not_original_step2000'),
        warm_start=lineage,checkpoint=str(args.checkpoint.resolve()),
        device='cpu',task=args.task if lineage else 'StirVegetables',episode=episode,step=step,rows=rows,comparisons=comparisons,
        paired_input_and_action_noise_equal=controlled,paired_action_gradients_equal=same_gradients,
        selected_video_parameter_precision='float32' if args.video_parameters_fp32 else 'base_dtype',
        auxiliary_video_loss_changed_when_future_labels_changed=future_loss_changed,
        gradient_digests=gradient_checks,optimizer_updates=0,new_policy_trials=0,
        source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources},
        scope='Actual CPU forwards and separate backwards at common initialized weights, one real training observation. '
              'Verifies common action inputs/noise/gradients and separation from future-label supervision. '
              'Not the historical step2000 A/B, GPU execution, or success rate.')
    (args.output/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['status','paired_input_and_action_noise_equal','paired_action_gradients_equal','optimizer_updates']}),flush=True)
    if report['status']!='complete':raise RuntimeError('Paired experiment controls not proven; inspect acceptance.json')


if __name__=='__main__':main()
