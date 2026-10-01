"""Compare released GR1 and Atomic18 action weights on Composite observations.

All sixteen tasks, one train and one held-out demonstration each, initial and
middle observations, paired noise draws. Frozen video features are shared. This
is an initialization diagnostic, not training or a policy success-rate test.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

from scripts.robocasa365.probe_atomic_cpu import sha, binary_metrics
from scripts.robocasa365.prediction_metrics import action_metrics


def save(path, value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    import torch
    from omegaconf import OmegaConf
    from DiT4DiT.model.framework.base_framework import baseframework
    from scripts.robocasa365.multitask_data import apply_statistics, prepare_multitask_dataset

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--atomic-checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selection-seed',type=int,default=79)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8);torch.set_num_interop_threads(2);started=time.monotonic()
    manifest_path=args.prepared/'manifest.json';manifest=json.loads(manifest_path.read_text())
    assert manifest['task_set']=='composite_seen' and len(manifest['tasks'])==16
    stats_path=args.prepared/'normalization.json';stats=json.loads(stats_path.read_text())
    assert sha(stats_path)==manifest['normalization_sha256']
    payload=torch.load(args.atomic_checkpoint,map_location='cpu',weights_only=False,mmap=True)
    assert payload['phase']=='action' and payload['global_step']==50000
    assert len(payload['trained_state'])==247 and all(k.startswith('action_model.') for k in payload['trained_state'])
    atomic_dir=args.atomic_checkpoint.parent;atomic_stats_path=atomic_dir/'normalization.json'
    atomic_stats=json.loads(atomic_stats_path.read_text());assert sha(atomic_stats_path)==payload['normalization_sha256']
    source_tasks={row['task'] for row in payload['split']['tasks']}
    assert payload['split']['task_set']=='atomic_seen' and len(source_tasks)==18
    assert source_tasks.isdisjoint({row['task'] for row in manifest['tasks']})
    prepared=args.output/'prepared';prepared.mkdir()
    collection,_,_,_=prepare_multitask_dataset(manifest_path,prepared)
    cfg=OmegaConf.load(prepared/'data_config.yaml');atomic_cfg=OmegaConf.load(atomic_dir/'data_config.yaml')
    compatible_keys=['action_horizon','max_state_dim','max_action_dim','image_size','action_video_freq_ratio']
    assert all(cfg[key]==atomic_cfg[key] for key in compatible_keys)
    assert all(stats['action'][key]==atomic_stats['action'][key] for key in ['min','max'])
    rng=np.random.default_rng(args.selection_seed)
    datasets={};examples=[];observations=[];targets=[];masks=[]
    for task_index,task_row in enumerate(manifest['tasks']):
        task=task_row['task'];ds=collection.datasets[task_index];datasets[task]=ds
        lookup={(int(ep),int(step)):i for i,(ep,step) in enumerate(ds.dataset.all_steps)}
        assert set(task_row['train_episodes']).isdisjoint(task_row['validation_episodes'])
        for split in ['train','validation']:
            episode=int(rng.choice(task_row[split+'_episodes']))
            length=ds.dataset.trajectory_lengths[ds.dataset.get_trajectory_index(episode)]
            assert length>=16
            for label,step in [('initial',0),('middle',min(length//2,length-16))]:
                index=lookup[episode,step]
                apply_statistics(ds,task_row['path'],stats);example=ds[index]
                raw=ds.dataset.get_step_data(episode,step)
                target=np.concatenate([np.asarray(raw[key]).copy() for key in ds.robot.action_keys],axis=-1)
                np.testing.assert_allclose(ds.decode_actions(example['action'],env_order=False),target,atol=1e-6,rtol=1e-5)
                decoder_probe=np.linspace(-2,2,16*32,dtype=np.float32).reshape(16,32)
                composite_decoded=ds.decode_actions(decoder_probe,env_order=False)
                apply_statistics(ds,task_row['path'],atomic_stats);atomic_example=ds[index]
                for key in ['state','action','state_mask','action_mask']:
                    np.testing.assert_array_equal(example[key],atomic_example[key])
                np.testing.assert_array_equal(ds.decode_actions(decoder_probe,env_order=False),composite_decoded)
                apply_statistics(ds,task_row['path'],stats)
                example['image']=example['image'][:1]
                examples.append(example);targets.append(target);masks.append(example['action_mask'][:,:12])
                observations.append(dict(task=task,split=split,episode=episode,step=int(step),stage=label,language=example['lang']))
        print('COMPOSITE_DATA_READY',task,flush=True)
    identity=dict(created_utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=sha(manifest_path),
        composite_normalization_sha256=sha(stats_path),atomic_normalization_sha256=sha(atomic_stats_path),
        effective_continuous_action_normalization='min_max',effective_action_min_max_identical=True,
        state_normalization='raw16D padded to64; no additional normalization',
        actual_adapter_state_action_masks_equal=True,normalization_bytes_identical=sha(stats_path)==sha(atomic_stats_path),
        source_task_set='atomic_seen',source_and_target_tasks_disjoint=True,compatible_data_config_keys=compatible_keys,
        base_checkpoint=payload['base_checkpoint'],atomic_checkpoint=str(args.atomic_checkpoint.resolve()),
        atomic_checkpoint_sha256=sha(args.atomic_checkpoint),base_checkpoint_sha256=sha(payload['base_checkpoint']),
        observations=observations,unique_episodes=len({(x['task'],x['episode']) for x in observations}),
        selection_seed=args.selection_seed,noise_seeds=[123,124],inference_steps=4,source_sha256=sha(__file__),
        scope='CPU prediction diagnostic on32 complete train/validation demonstrations,64 observations. Paired noise repetitions are not independent episodes. No optimizer or closed-loop policy trials; original16-demo step2000 A/B remains separate.')
    save(args.output/'identity.json',identity)
    model=baseframework.from_pretrained(payload['base_checkpoint']);model.requires_grad_(False);model.eval()
    model.config.datasets.vla_data=cfg;model.config.framework.cosmos25.training='action'
    model.config.trainer.repeated_diffusion_steps=1;model.action_model.float()
    assert all(p.device.type=='cpu' for p in model.parameters())
    features=[];records=[]
    with torch.inference_mode():
        for i,example in enumerate(examples):
            torch.manual_seed(50000+i)
            inputs=model.backbone_interface.build_cosmos_inputs(images=[example['image']],instructions=[example['lang']])
            with torch.autocast('cpu',dtype=torch.bfloat16):
                output=model.backbone_interface(**inputs,output_attentions=False,output_hidden_states=True,return_dict=True)
            hidden=output.hidden_states[-1].detach().float();assert torch.isfinite(hidden).all();features.append(hidden)
            save(args.output/'progress.json',dict(stage='encoding',completed=i+1,planned=len(examples),seconds=time.monotonic()-started))
            print('COMPOSITE_FEATURE',i+1,len(examples),flush=True)
        for initialization in ['released_gr1','atomic18_step050000']:
            if initialization=='atomic18_step050000':
                model.action_model.load_state_dict({k.removeprefix('action_model.'):v for k,v in payload['trained_state'].items()},strict=True)
            model.action_model.num_inference_timesteps=4;values=[]
            for i,(hidden,example) in enumerate(zip(features,examples)):
                state=torch.as_tensor(example['state'][None],dtype=torch.float32)
                for seed in [123,124]:
                    draw_seed=seed+1000*i;torch.manual_seed(draw_seed)
                    normalized=model.action_model.predict_action(hidden,state)[0].float().numpy()
                    assert normalized.shape==(16,32) and np.isfinite(normalized).all()
                    decoded=datasets[observations[i]['task']].decode_actions(normalized,env_order=False)
                    values.append(dict(observation=i,noise_seed=draw_seed,normalized=normalized,decoded=decoded))
            np.savez_compressed(args.output/(initialization+'.npz'),
                normalized=np.stack([v['normalized'] for v in values]),decoded=np.stack([v['decoded'] for v in values]),
                target=np.stack([targets[v['observation']] for v in values]),valid=np.stack([masks[v['observation']] for v in values]),
                observation=np.asarray([v['observation'] for v in values]),noise_seed=np.asarray([v['noise_seed'] for v in values]))
            for task in [x['task'] for x in manifest['tasks']]:
                for split in ['train','validation']:
                    group=[v for v in values if observations[v['observation']]['task']==task and observations[v['observation']]['split']==split]
                    prediction=np.stack([v['decoded'] for v in group])[:,:8]
                    target=np.stack([targets[v['observation']] for v in group])[:,:8]
                    valid=np.stack([masks[v['observation']] for v in group])[:,:8]
                    records.append(dict(initialization=initialization,task=task,split=split,distinct_episodes=1,
                        distinct_windows=2,noise_repeats=2,predictions=len(group),execution_prefix=8,
                        metrics=action_metrics(prediction,target,valid),base_mode=binary_metrics(prediction[...,4],target[...,4],valid[...,4]),
                        gripper=binary_metrics(prediction[...,11],target[...,11],valid[...,11])))
            save(args.output/'partial_metrics.json',records)
            print('COMPOSITE_INITIALIZATION_COMPLETE',initialization,flush=True)
    report=dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),observations=len(observations),
        unique_episodes=identity['unique_episodes'],sampled_action_chunks=2*len(observations)*2,records=records,
        seconds=time.monotonic()-started,identity_sha256=sha(args.output/'identity.json'),optimizer_updates=0,new_policy_trials=0,scope=identity['scope'])
    save(args.output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
