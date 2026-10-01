"""Training-only initial/switch pools with paired task draws and resumable RNG.

Command switches are annotations of recorded labels, not physical contact events.
The parent sampler draws task and a child seed using fixed bounds; differences in
pool sizes cannot alter the next task draw between the two comparison branches.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

PROTOCOL='composite_initial_switch_sampling_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def switch_starts(labels, prefix=8):
    labels=np.asarray(labels)
    if labels.ndim!=1 or prefix<2:raise ValueError('Expected a command sequence and prefix>=2')
    changes=np.r_[0,labels[1:]!=labels[:-1]].astype(np.int64)
    cumulative=np.cumsum(changes);starts=np.arange(len(labels))
    ends=np.minimum(starts+prefix-1,len(labels)-1)
    return np.flatnonzero(cumulative[ends]>cumulative[starts])


def build_pools(manifest_path, output, initial_frames=16, prefix=8):
    import pyarrow as pa
    import pyarrow.parquet as pq

    manifest_path=Path(manifest_path).resolve();output=Path(output)
    if initial_frames<1 or prefix<2:raise ValueError('Invalid phase window sizes')
    manifest=json.loads(manifest_path.read_text())
    if manifest['task_set']!='composite_seen' or len(manifest['tasks'])!=16:
        raise ValueError('Phase pools require the full16 Composite manifest')
    output.mkdir(parents=True,exist_ok=False);pa.set_cpu_count(2)
    arrays={};tasks=[];checked=0
    for task in manifest['tasks']:
        root=Path(task['path']);info=json.loads((root/'meta/info.json').read_text())
        metadata_path=root/'meta/episodes.jsonl'
        episodes=[json.loads(line) for line in metadata_path.read_text().splitlines() if line]
        expected={e['episode']:e for e in task['episodes']}
        assert {e['episode_index'] for e in episodes}==set(expected)
        train=set(task['train_episodes']);validation=set(task['validation_episodes'])
        assert train.isdisjoint(validation) and train|validation==set(expected)
        initial=[];switch=[];training=[];ranges=[];offset=0
        for episode in episodes:
            ep,n=episode['episode_index'],episode['length'];entry=expected[ep]
            assert n==entry['frames'] and n>0
            ranges.append(dict(episode=ep,start=offset,end=offset+n,split=entry['split']))
            if ep in train:
                assert entry['split']=='train'
                path=root/info['data_path'].format(episode_chunk=ep//info['chunks_size'],episode_index=ep)
                assert sha(path)==entry['parquet_sha256']
                column=pq.read_table(path,columns=['action'])['action'].combine_chunks()
                action=np.asarray(column.to_pylist())
                assert action.shape==(n,12) and np.isfinite(action).all()
                assert np.max(np.abs(action))<=1 and np.isin(action[:,[4,11]],[-1,1]).all()
                active=np.union1d(switch_starts(action[:,4],prefix),switch_starts(action[:,11],prefix))
                initial.extend((offset+np.arange(min(initial_frames,n))).tolist())
                switch.extend((offset+active).tolist())
                training.extend(range(offset,offset+n));checked+=1
            else:assert entry['split']=='validation'
            offset+=n
        name=task['task'];training=np.asarray(training,dtype=np.int64)
        arrays['initial__'+name]=np.asarray(initial,dtype=np.int64)
        arrays['switch__'+name]=np.asarray(switch,dtype=np.int64)
        assert np.all(np.diff(arrays['initial__'+name])>0)
        assert np.all(np.diff(arrays['switch__'+name])>0)
        tasks.append(dict(task=name,all_frames=offset,train_frames=len(training),
            train_episodes=len(train),validation_episodes=len(validation),episode_ranges=ranges,
            training_indices_sha256=hashlib.sha256(training.tobytes()).hexdigest(),
            initial_count=len(initial),switch_count=len(switch),episode_metadata_sha256=sha(metadata_path)))
        print('PHASE_POOLS_TASK',name,len(training),len(initial),len(switch),flush=True)
    assert checked==manifest['train_episodes']==7271
    np.savez_compressed(output/'pools.npz',**arrays)
    report=dict(status='complete',protocol=PROTOCOL,created_utc=datetime.now(timezone.utc).isoformat(),
        manifest_path=str(manifest_path),manifest_sha256=sha(manifest_path),source_sha256=sha(Path(__file__)),
        pools_file='pools.npz',pools_sha256=sha(output/'pools.npz'),initial_frames=initial_frames,
        execution_prefix=prefix,training_parquets_hash_checked=checked,validation_parquets_used=0,
        default_mixture=dict(ordinary=.5,initial=.25,switch=.25),tasks=tasks,
        scope='All training episodes retained. Initial pool uses first recorded frames; switch pool contains both recorded labels in the valid execution prefix. No physical-event labels or validation episodes used.')
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


class PhaseBalancedTasks:
    def __init__(self, dataset, pool_manifest, dataset_manifest, mode):
        if mode not in ['paired_uniform','initial_switch']:
            raise ValueError('Unknown paired sampling mode')
        self.dataset=dataset;self.datasets=dataset.datasets;self.indices=dataset.indices
        pool_manifest=Path(pool_manifest).resolve()
        metadata=json.loads(pool_manifest.read_text())
        if metadata['status']!='complete' or metadata['protocol']!=PROTOCOL:
            raise ValueError('Incomplete or incompatible phase pools')
        if metadata['manifest_sha256']!=sha(dataset_manifest):
            raise ValueError('Phase pools belong to another dataset manifest')
        source_manifest=json.loads(Path(dataset_manifest).read_text())
        if ([row['task'] for row in metadata['tasks']]!=[row['task'] for row in source_manifest['tasks']]
                or source_manifest['task_set']!='composite_seen'):
            raise ValueError('Phase pool task order differs')
        if metadata['default_mixture']!=dict(ordinary=.5,initial=.25,switch=.25):
            raise ValueError('Phase pool mixture differs from the implemented protocol')
        if metadata['source_sha256']!=sha(Path(__file__)):
            raise ValueError('Phase-pool builder/sampler source changed')
        path=pool_manifest.parent/metadata['pools_file']
        if sha(path)!=metadata['pools_sha256']:raise ValueError('Phase pool data digest differs')
        if len(metadata['tasks'])!=len(self.datasets):raise ValueError('Phase pool task count differs')
        self.pools=[];self.mode=mode
        with np.load(path,allow_pickle=False) as data:
            for row,adapter,indices in zip(metadata['tasks'],self.datasets,self.indices):
                actual=np.asarray(adapter.dataset.all_steps,dtype=np.int64)
                assert len(adapter)==row['all_frames'] and actual.shape==(row['all_frames'],2)
                assert hashlib.sha256(indices.tobytes()).hexdigest()==row['training_indices_sha256']
                for entry in row['episode_ranges']:
                    start,end=entry['start'],entry['end']
                    np.testing.assert_array_equal(actual[start:end,0],np.full(end-start,entry['episode']))
                    np.testing.assert_array_equal(actual[start:end,1],np.arange(end-start))
                allowed=np.zeros(len(adapter),dtype=bool);allowed[indices]=True
                local={}
                for kind in ['initial','switch']:
                    values=data[kind+'__'+row['task']].copy()
                    assert values.dtype==np.int64 and values.ndim==1 and len(values)==row[kind+'_count']
                    assert np.all(np.diff(values)>0) and np.all((values>=0)&(values<len(adapter)))
                    assert allowed[values].all(), 'Validation frame entered a phase pool'
                    local[kind]=values
                self.pools.append(local)
        self.sampling_spec=dict(protocol=PROTOCOL,mode=mode,pool_manifest_sha256=sha(pool_manifest),
            pools_sha256=metadata['pools_sha256'],source_sha256=sha(Path(__file__)),
            initial_frames=metadata['initial_frames'],execution_prefix=metadata['execution_prefix'],
            mixture=metadata['default_mixture'] if mode=='initial_switch' else dict(ordinary=1.,initial=0.,switch=0.),
            task_distribution='uniform',parent_rng='fixed task draw and private child seed',
            empty_pool_fallback='ordinary training frames',source_split_sampling_is_provenance=True)

    def sample_index(self, rng):
        task=int(rng.integers(len(self.datasets)))
        child=np.random.default_rng(int(rng.integers(np.iinfo(np.int64).max)))
        draw=float(child.random())
        kind='ordinary'
        if self.mode=='initial_switch':
            kind='ordinary' if draw<.5 else 'initial' if draw<.75 else 'switch'
        pool=self.indices[task] if kind=='ordinary' else self.pools[task][kind]
        if not len(pool):pool=self.indices[task]
        return [task,int(pool[int(child.integers(len(pool)))])]

    def __getitem__(self,index):
        return self.dataset[index]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=build_pools(args.manifest,args.output)
    print(json.dumps(dict(status=result['status'],tasks=len(result['tasks']),training_parquets_hash_checked=result['training_parquets_hash_checked'])),flush=True)
