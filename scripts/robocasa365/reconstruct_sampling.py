"""Reconstruct a saved run's real sample stream and verify all rank RNG endpoints.

No model forward, backward, optimizer update, simulator or GPU work is performed.
Coverage is reported separately for conditioning observations and action targets.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from scripts.robocasa365.multitask_data import prepare_multitask_dataset


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b''):value.update(block)
    return value.hexdigest()


def save(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--checkpoint-audit',type=Path,required=True)
    parser.add_argument('--prospective-manifest',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_available():
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES empty for CPU-only reconstruction')
    torch.set_num_threads(2)
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    audit=json.loads(args.checkpoint_audit.read_text())
    assert audit['status']=='complete' and args.checkpoint.stat().st_size==audit['checkpoint_bytes']
    assert sha(args.checkpoint)==audit['checkpoint_sha256']
    payload=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    schedule=payload['training_schedule'];updates=payload['global_step'];world=schedule['world_size']
    assert payload['phase']=='action' and payload['warm_start'] is None
    assert len(payload['distributed_rng'])==world and updates==audit['global_step']
    trainer=Path('scripts/robocasa365/train_distributed.py')
    trainer_sha=sha(trainer)
    assert sha(args.checkpoint.parent/f'trainer_{trainer_sha[:12]}.py')==trainer_sha
    data_output=args.output/'prepared';data_output.mkdir()
    dataset,_,_,split=prepare_multitask_dataset(args.manifest,data_output)
    assert split==payload['split']
    assert sha(data_output/'normalization.json')==payload['normalization_sha256']
    counts=[np.zeros(len(adapter),dtype=np.uint32) for adapter in dataset.datasets]
    per_rank=updates*schedule['microbatch']*schedule['accumulation']
    report=dict(status='reconstructing',started_utc=datetime.now(timezone.utc).isoformat(),
                checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=audit['checkpoint_sha256'],
                checkpoint_step=updates,schedule=schedule,manifest_sha256=sha(args.manifest),
                source_sha256={str(trainer):trainer_sha,'scripts/robocasa365/multitask_data.py':sha('scripts/robocasa365/multitask_data.py'),
                               'scripts/robocasa365/reconstruct_sampling.py':sha(__file__)},
                rank_checks=[],tasks=[],new_training_updates=0,new_policy_trials=0)
    save(args.output/'progress.json',report)
    for rank in range(world):
        sampler=np.random.default_rng(np.random.SeedSequence([schedule['seed'],rank]))
        for draw in range(per_rank):
            task,index=dataset.sample_index(sampler)
            counts[task][index]+=1
        same=sampler.bit_generator.state==payload['distributed_rng'][rank]['sampler']
        report['rank_checks'].append(dict(rank=rank,draws=per_rank,endpoint_matches_checkpoint=same))
        save(args.output/'progress.json',report)
        if not same:
            report.update(status='failed_rng_endpoint_mismatch',failed_rank=rank)
            save(args.output/'report.json',report)
            raise RuntimeError(f'Rank {rank} RNG endpoint does not match; do not label counts actual coverage')
        print('SAMPLER_ENDPOINT_MATCHED',rank,per_rank,flush=True)
    horizon=16
    for task_id,(adapter,task_split,starts) in enumerate(zip(dataset.datasets,split['tasks'],counts)):
        all_steps=np.asarray(adapter.dataset.all_steps,dtype=np.int64)
        training_indices=dataset.indices[task_id]
        allowed=np.zeros(len(adapter),dtype=bool);allowed[training_indices]=True
        assert not starts[~allowed].any(), 'Validation episode was sampled'
        boundaries=np.r_[0,np.flatnonzero(np.diff(all_steps[:,0]))+1,len(all_steps)]
        episodes=[]
        for begin,end in zip(boundaries[:-1],boundaries[1:]):
            episode=int(all_steps[begin,0])
            if episode not in task_split['train_episodes']:continue
            np.testing.assert_array_equal(all_steps[begin:end,1],np.arange(end-begin))
            local=starts[begin:end].astype(np.int64)
            targets=np.convolve(local,np.ones(horizon,dtype=np.int64))[:len(local)]
            assert int(targets.sum())==int(np.dot(local,np.minimum(horizon,len(local)-np.arange(len(local)))))
            episodes.append(dict(episode=episode,frames=len(local),draws=int(local.sum()),
                unique_conditioning_frames=int(np.count_nonzero(local)),initial_frame_draws=int(local[0]),
                first8_conditioning_draws=int(local[:8].sum()),first8_frames=min(8,len(local)),
                first8_unique_conditioning_frames=int(np.count_nonzero(local[:8])),
                valid_action_target_presentations=int(targets.sum()),unique_action_target_frames=int(np.count_nonzero(targets))))
        frame_count=len(training_indices);draws=int(starts.sum())
        row=dict(task=task_split['task'],train_episodes=len(episodes),train_frames=frame_count,draws=draws,
                 episodes_seen=sum(r['draws']>0 for r in episodes),unique_conditioning_frames=int(np.count_nonzero(starts)),
                 conditioning_frame_coverage=float(np.count_nonzero(starts)/frame_count),
                 mean_conditioning_draws_per_training_frame=draws/frame_count,
                 episodes_with_initial_frame_seen=sum(r['initial_frame_draws']>0 for r in episodes),
                 initial_frame_draws=sum(r['initial_frame_draws'] for r in episodes),
                 first8_conditioning_draws=sum(r['first8_conditioning_draws'] for r in episodes),
                 first8_conditioning_frame_coverage=sum(r['first8_unique_conditioning_frames'] for r in episodes)/sum(r['first8_frames'] for r in episodes),
                 valid_action_target_presentations=sum(r['valid_action_target_presentations'] for r in episodes),
                 unique_action_target_frames=sum(r['unique_action_target_frames'] for r in episodes))
        assert row['train_episodes']==len(task_split['train_episodes'])
        report['tasks'].append(row)
    total_frames=sum(r['train_frames'] for r in report['tasks'])
    report['aggregate']=dict(draws=sum(r['draws'] for r in report['tasks']),train_frames=total_frames,
        train_episodes=sum(r['train_episodes'] for r in report['tasks']),episodes_seen=sum(r['episodes_seen'] for r in report['tasks']),
        unique_conditioning_frames=sum(r['unique_conditioning_frames'] for r in report['tasks']),
        conditioning_frame_coverage=sum(r['unique_conditioning_frames'] for r in report['tasks'])/total_frames,
        mean_conditioning_draws_per_training_frame=world*per_rank/total_frames,
        episodes_with_initial_frame_seen=sum(r['episodes_with_initial_frame_seen'] for r in report['tasks']),
        valid_action_target_presentations=sum(r['valid_action_target_presentations'] for r in report['tasks']),
        unique_action_target_frames=sum(r['unique_action_target_frames'] for r in report['tasks']))
    assert report['aggregate']['draws']==world*per_rank
    if args.prospective_manifest:
        prospective=json.loads(args.prospective_manifest.read_text());k=len(prospective['tasks']);future=[]
        for task in prospective['tasks']:
            frames=sum(e['frames'] for e in task['episodes'] if e['split']=='train')
            expected=-np.expm1(world*per_rank*np.log1p(-1/(k*frames)))
            future.append(dict(task=task['task'],train_frames=frames,expected_conditioning_draws=world*per_rank/k,
                               expected_unique_conditioning_fraction=float(expected)))
        report['prospective_composite']=dict(executed=False,manifest_sha256=sha(args.prospective_manifest),
            planned_updates=updates,planned_draws=world*per_rank,tasks=future,
            expected_overall_unique_conditioning_fraction=sum(t['train_frames']*t['expected_unique_conditioning_fraction'] for t in future)/sum(t['train_frames'] for t in future),
            scope='Analytic expectation under unchanged uniform-task/uniform-frame sampling. Not an actual Composite training result.')
    np.savez_compressed(args.output/'start_counts.npz',**{t['task']:c for t,c in zip(split['tasks'],counts)})
    report.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,
        scope='Actual sampler implementation, regenerated exact split and all rank RNG endpoints match the saved checkpoint. '
              'Coverage counts current observation starts separately from overlapping future action targets; neither proves convergence or policy success.')
    save(args.output/'report.json',report)
    print(json.dumps(dict(status=report['status'],aggregate=report['aggregate'],rank_checks=report['rank_checks'])),flush=True)


if __name__=='__main__':main()
