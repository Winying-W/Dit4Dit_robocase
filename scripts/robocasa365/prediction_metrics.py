"""Offline *sampled* action diagnostics, separate from noisy FM training loss."""
import json
from pathlib import Path
import numpy as np


def action_metrics(decoded, target, valid):
    decoded=np.asarray(decoded);target=np.asarray(target);valid=np.asarray(valid,dtype=bool)
    assert decoded.shape==target.shape==valid.shape and decoded.shape[-1]==12
    assert np.isfinite(decoded).all() and np.isfinite(target).all()
    clipped=np.clip(decoded,-1,1)
    def group(columns, scale=1.):
        error=(clipped[...,columns]-target[...,columns])*scale
        selection=valid[...,columns]
        count=int(selection.sum())
        if not count:return dict(count=0,mae=None,rmse=None)
        error=error[selection]
        return dict(count=count,mae=float(np.mean(np.abs(error))),rmse=float(np.sqrt(np.mean(error**2))))
    grip_mask=valid[...,11]
    pred=clipped[...,11]>=.5;truth=target[...,11]>=.5
    accuracy=float(np.mean(pred[grip_mask]==truth[grip_mask]))
    recalls={}
    for label in (False,True):
        selected=grip_mask & (truth==label)
        recalls['closed' if label else 'open']=float(np.mean(pred[selected]==truth[selected])) if selected.any() else None
    return dict(position_command=group(slice(5,8)),rotation_command=group(slice(8,11)),arm_command=group(slice(5,11)),
        position_scaled_component_m=group(slice(5,8),.05),rotation_scaled_component_rad=group(slice(8,11),.5),
        gripper_accuracy=accuracy,gripper_class_recalls=recalls,
        gripper_balanced_accuracy=float(np.mean([v for v in recalls.values() if v is not None])),
        target_gripper_closed_fraction=float(np.mean(truth[grip_mask])),
        predicted_gripper_closed_fraction=float(np.mean(pred[grip_mask])),
        predicted_base_mode_fraction=float(np.mean(clipped[...,4][valid[...,4]]>=.5)),
        clipped_arm_component_fraction=float(np.mean((np.abs(decoded[...,5:11])>1)[valid[...,5:11]])))


def probe_dataset_predictions(model,ds,episode,output,windows=64,seeds=(123,124,125)):
    import torch
    if windows<1:raise ValueError('Positive probe window count required')
    all_indices=[i for i,(ep,_) in enumerate(ds.dataset.all_steps) if int(ep)==episode]
    if not all_indices:raise ValueError(f'No such episode: {episode}')
    indices=np.asarray(all_indices)[np.unique(np.linspace(0,len(all_indices)-1,min(windows,len(all_indices)),dtype=int))]
    records={key:[] for key in ('normalized','decoded','target','valid','dataset_index','step','rng_seed','state')}
    with torch.no_grad(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for window,index in enumerate(indices):
            ep,t=ds.dataset.all_steps[int(index)]
            raw=ds.dataset.get_step_data(ep,t)
            target=np.concatenate([np.asarray(raw[k]).copy() for k in ds.robot.action_keys],axis=-1)
            ex=ds[int(index)]
            for seed in seeds:
                draw_seed=seed+1000*window
                torch.manual_seed(draw_seed);torch.cuda.manual_seed_all(draw_seed)
                normalized=model.predict_action(examples=[dict(image=ex['image'][:1],state=ex['state'],lang=ex['lang'])])['normalized_actions'][0]
                assert normalized.shape==(16,32) and np.isfinite(normalized).all()
                decoded=ds.decode_actions(normalized,env_order=False)
                values=dict(normalized=normalized,decoded=decoded,target=target,valid=ex['action_mask'][:,:12],
                    dataset_index=int(index),step=int(t),rng_seed=draw_seed,state=ex['state'][0,:16])
                for key,value in values.items():records[key].append(np.asarray(value).copy())
            if (window+1)%16==0:print('OFFLINE_PROBE',episode,window+1,len(indices),flush=True)
    arrays={key:np.asarray(value) for key,value in records.items()}
    report=dict(episode=episode,windows=len(indices),noise_repeats=len(seeds),predictions=len(arrays['decoded']),
        scope='Training-episode prediction fit; not validation or task success. Scaled errors are controller-command component errors, not achieved pose errors.',
        overall=action_metrics(arrays['decoded'],arrays['target'],arrays['valid']),
        by_chunk_step={str(t+1):action_metrics(arrays['decoded'][:,t:t+1],arrays['target'][:,t:t+1],arrays['valid'][:,t:t+1]) for t in (0,7,15)})
    # A fixed zero-arm / open-gripper reference shows whether small commands alone explain errors.
    constant=np.zeros_like(arrays['decoded']);constant[...,4]=-1;constant[...,11]=-1
    report['zero_arm_open_gripper_reference']=action_metrics(constant,arrays['target'],arrays['valid'])
    output=Path(output)
    np.savez_compressed(output/'offline_predictions.npz',**arrays)
    (output/'offline_prediction_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print('OFFLINE_PROBE_COMPLETE',json.dumps(report),flush=True)
    return report
