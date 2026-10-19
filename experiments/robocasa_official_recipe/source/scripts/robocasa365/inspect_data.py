"""Inspect real RGB, schema and full-dataset base usage without importing DiT4DiT."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
info=json.loads((a.dataset/'meta/info.json').read_text())
mod=json.loads((a.dataset/'meta/modality.json').read_text())
episodes=[json.loads(x) for x in (a.dataset/'meta/episodes.jsonl').read_text().splitlines()]
report={'dataset':str(a.dataset),'fps':info['fps'],'num_episodes':len(episodes),'modality':mod,'episodes':[], 'image_shapes':{}}
for key in mod['video']:
    path=next((a.dataset/'videos').glob(f'*/observation.images.{key}/episode_000000.mp4'))
    cap=cv2.VideoCapture(str(path));ok,frame=cap.read();cap.release();assert ok
    report['image_shapes'][key]=list(frame.shape)
base_max=np.zeros(4);mode_values=set()
for ep in episodes:
    df=pd.read_parquet(next((a.dataset/'data').glob(f'*/episode_{ep["episode_index"]:06d}.parquet')))
    actions=np.stack(df.action);state=np.stack(df['observation.state'])
    assert actions.shape==(ep['length'],12) and state.shape==(ep['length'],16)
    assert np.isfinite(actions).all() and np.isfinite(state).all()
    base_max=np.maximum(base_max,np.abs(actions[:,:4]).max(0));mode_values.update(actions[:,4].tolist())
    if ep['episode_index']<3:
        report['episodes'].append(dict(index=ep['episode_index'],length=len(df),state_shape=list(state.shape),action_shape=list(actions.shape),instruction=ep['tasks'],base_max=np.abs(actions[:,:4]).max(0).tolist(),mode_values=np.unique(actions[:,4]).tolist(),gripper_values=np.unique(actions[:,11]).tolist()))
report['full_dataset_base_action_abs_max']=base_max.tolist();report['control_mode_values']=sorted(mode_values)
report['future_actions']='At t, action[t:t+16]; mask timesteps past episode end; never cross episode boundaries.'
a.output.mkdir(parents=True,exist_ok=True);(a.output/'dataset_schema.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
