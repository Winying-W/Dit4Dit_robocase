"""Build immutable human300 episode splits and exact disk-backed training statistics."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.robocasa365.prepare_multitask import deterministic_split, save


def official_human_episodes(episodes):
    """Match official get_subset_demos_filter_key(100_demos, seed=0)."""
    ids = [int(e["episode_index"]) for e in episodes]
    assert len(ids) == len(set(ids))
    if len(ids) < 100:
        raise ValueError("human300 task has fewer than 100 demonstrations")
    random.Random(0).shuffle(ids)
    return sorted(ids[:100])


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--preflight-tasks',nargs='+',help='Explicit small subset for isolated preflight, never labelled full human300');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'manifest.json').exists():raise FileExistsError('Immutable manifest exists')
    source=json.loads(a.source.read_text());assert len(source['tasks'])==300
    selected=source['tasks']
    if a.preflight_tasks:
        assert len(set(a.preflight_tasks))==len(a.preflight_tasks) and set(a.preflight_tasks).issubset({r['task'] for r in selected})
        selected=[r for r in selected if r['task'] in a.preflight_tasks]
    rows=[];total=0
    for item in selected:
        path=a.root/'extracted'/Path(item['remote']).parent
        assert (path/'EXTRACTION_VERIFIED.json').is_file(),path
        info=json.loads((path/'meta/info.json').read_text());assert info['fps']==20
        meta=json.loads((path/'extras/dataset_meta.json').read_text())['env_args']
        assert meta['env_name']==item['task'],(item['task'],meta['env_name'])
        env=meta['env_kwargs'];assert env['robots']=='PandaOmron' and env['obj_instance_split']=='pretrain'
        arm=env['controller_configs']['body_parts']['right']
        assert arm['type']=='OSC_POSE' and arm['input_type']=='delta' and arm['input_ref_frame']=='base'
        np.testing.assert_allclose(arm['output_max'],[.05,.05,.05,.5,.5,.5],rtol=0,atol=0)
        files={int(f.stem.split('_')[-1]):f for f in sorted((path/'data').glob('*/episode_*.parquet'))}
        assert len(files)==info['total_episodes']
        modality=json.loads((path/'meta/modality.json').read_text())
        for key,bounds in dict(base_motion=(0,4),control_mode=(4,5),end_effector_position=(5,8),end_effector_rotation=(8,11),gripper_close=(11,12)).items():
            assert (modality['action'][key]['start'],modality['action'][key]['end'])==bounds
        episodes=[json.loads(s) for s in (path/'meta/episodes.jsonl').read_text().splitlines() if s.strip()]
        lengths={int(e['episode_index']):int(e['length']) for e in episodes};assert set(lengths)==set(files)
        chosen=official_human_episodes(episodes)
        excluded=sorted(set(files)-set(chosen))
        train,val=deterministic_split(chosen,item['task'],a.seed,.05)
        total+=sum(lengths[ep] for ep in train)
        rows.append(dict(task=item['task'],path=str(path.resolve()),horizon=item['horizon'],train_episodes=train,validation_episodes=val,excluded_episodes=excluded,episode_lengths=lengths))
    scratch=a.output/'statistics_frames.f32'
    frames=np.memmap(scratch,dtype=np.float32,mode='w+',shape=(total,28));offset=0
    for row in rows:
        path=Path(row['path']);info=json.loads((path/'meta/info.json').read_text())
        for ep in row['train_episodes']+row['validation_episodes']:
            f=next((path/'data').glob(f'*/episode_{ep:06d}.parquet'))
            df=pd.read_parquet(f,columns=['action','observation.state']);action=np.stack(df['action']).astype(np.float32);state=np.stack(df['observation.state']).astype(np.float32)
            assert action.shape==(len(df),12) and state.shape==(len(df),16)
            assert len(df)==row['episode_lengths'][ep] and np.isfinite(action).all() and np.isfinite(state).all()
            assert np.abs(action).max()<=1.000001
            assert set(np.unique(action[:,4])).issubset({-1.,1.}) and set(np.unique(action[:,11])).issubset({-1.,1.})
            for cam in ('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand'):
                video=path/info['video_path'].format(episode_chunk=ep//info['chunks_size'],video_key='observation.images.'+cam,episode_index=ep)
                assert video.is_file() and video.stat().st_size>0,video
            if ep in row['train_episodes']:
                frames[offset:offset+len(df),:12]=action;frames[offset:offset+len(df),12:]=state;offset+=len(df)
        print('AUDITED',row['task'],flush=True)
    assert offset==total;frames.flush();stats={}
    for key,lo,hi in [('action',0,12),('observation.state',12,28)]:
        values={name:[] for name in ['min','max','mean','std','q01','q99']}
        for col in range(lo,hi):
            x=np.array(frames[:,col],dtype=np.float64)
            qs=np.quantile(x,[.01,.99]);v=dict(min=x.min(),max=x.max(),mean=x.mean(),std=x.std(),q01=qs[0],q99=qs[1])
            for name,value in v.items():values[name].append(float(value))
        stats[key]=values
    save(a.output/'normalization.json',stats)
    save(a.output/'manifest.json',dict(format_version=2,episode_selection=dict(filter_key='100_demos',filter_key_seed=0,implementation='robocasa.utils.groot_utils.groot_dataset.get_subset_demos_filter_key'),task_set='human300_preflight' if a.preflight_tasks else 'pretrain_human300',split='pretrain',seed=a.seed,tasks=rows,train_frames=total,
        train_episodes=sum(len(r['train_episodes']) for r in rows),validation_episodes=sum(len(r['validation_episodes']) for r in rows),
        normalization_sha256=hashlib.sha256((a.output/'normalization.json').read_bytes()).hexdigest(),
        source_manifest_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
        data_config=dict(image_size=[224,224]),statistics_source='Exact statistics of all training frames only; official 100_demos seed 0 then 95/5 episode split per task; shared continuous-action min/max; raw state and binary commands'))
    del frames;scratch.unlink()


if __name__=='__main__':main()
