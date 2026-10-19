"""CPU-only actual-data acceptance for every task before spending GPU updates."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.robocasa365.multitask_data import prepare_multitask_dataset
from scripts.robocasa365.action_audit import audit_dataset_actions
from DiT4DiT.dataloader.robocasa365_datasets import collate_fn


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    ds,_,validation,split=prepare_multitask_dataset(a.manifest,a.output)
    reports=[]
    for i,(adapter,row) in enumerate(zip(ds.datasets,split['tasks'])):
        audit=audit_dataset_actions(adapter,Path(row['path']),row,require_fixed_base=False)
        samples=[adapter[row['validation_indices'][0]],adapter[len(adapter)-1]]
        for sample in samples:
            assert len(sample['image'])==9 and all(tuple(im.shape)==(3,128,384) for im in sample['image'])
            assert sample['state'].shape==(1,64) and sample['action'].shape==(16,32)
            assert sample['action_mask'].shape==(16,32)
            assert not sample['action_mask'][:,12:].any()
            assert np.isfinite(sample['state']).all() and np.isfinite(sample['action']).all()
            assert isinstance(sample['lang'],str) and sample['lang']
        assert samples[1]['action_mask'].sum()==12, 'Last frame must expose only one valid action'
        reports.append(dict(task=row['task'],frames=audit['frames'],episodes=len(audit['episodes']),
            roundtrip_max_error=audit['max_roundtrip_error'],
            nonzero_base_frames=sum(ep['nonzero_base_frames'] for ep in audit['episodes']),
            base_mode_frames=sum(ep['base_mode_frames'] for ep in audit['episodes']),
            video_shape=[9,3,128,384],state_shape=[1,64],action_shape=[16,32],mask_shape=[16,32],
            boundary_mask_passed=True,first_instruction=samples[0]['lang']))
        (a.output/f"{row['task']}_action_audit.json").write_text(json.dumps(audit,indent=2)+'\n')
        print('MULTITASK_ADAPTER_VERIFIED',row['task'],audit['frames'],audit['max_roundtrip_error'],flush=True)
    rng=np.random.default_rng(135)
    counts=np.zeros(len(ds.datasets),dtype=np.int64)
    for _ in range(16000):
        task,index=ds.sample_index(rng);counts[task]+=1
        episode,_=ds.datasets[task].dataset.all_steps[index]
        assert int(episode) in split['tasks'][task]['train_episodes']
    expected = 16000/len(ds.datasets)
    assert counts.min() > expected*.8 and counts.max() < expected*1.2, counts
    batch=collate_fn([ds[[i%len(ds.datasets),split['tasks'][i%len(ds.datasets)]['validation_indices'][0]]] for i in range(4)])
    assert len(batch)==4
    report=dict(status='complete',tasks=reports,episodes=sum(row['episodes'] for row in reports),
        frames=sum(row['frames'] for row in reports),train_episodes=split['train_episodes'],
        validation_episodes=split['validation_episodes'],validation_windows=len(validation),
        sampling_task_counts_16000_draws=counts.tolist(),mixed_task_batch_size=len(batch),
        scope='Actual RGB/state/action/padding/normalization and balanced-sampling checks on CPU; not a model forward, training result, or policy success rate.')
    (a.output/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
