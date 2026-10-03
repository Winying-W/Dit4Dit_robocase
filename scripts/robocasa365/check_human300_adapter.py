"""Check real human300 camera shapes and action normalization round-trip on CPU."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from DiT4DiT.dataloader.robocasa365_datasets import Robocasa365DatasetAdapter


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=OmegaConf.create(dict(dataset_path=str(a.dataset),action_horizon=16,max_state_dim=64,max_action_dim=32,image_size=[224,224],video_delta_indices=list(range(17)),action_video_freq_ratio=2,video_backend='decord',lerobot_version='v2.0'))
    ds=Robocasa365DatasetAdapter(a.dataset,cfg);reports=[]
    for index in [0,20,len(ds)-1]:
        sample=ds[index];ep,t=map(int,ds.dataset.all_steps[index]);f=next((a.dataset/'data').glob(f'*/episode_{ep:06d}.parquet'))
        raw=np.stack(pd.read_parquet(f,columns=['action']).action)[t:t+16]
        decoded=ds.decode_actions(sample['action'],env_order=False)[:len(raw)];err=float(np.max(np.abs(decoded-raw)));assert err<1e-6
        assert len(sample['image'])==9 and all(tuple(x.shape)==(3,224,672) for x in sample['image'])
        assert sample['state'].shape==(1,64) and sample['action'].shape==(16,32)
        assert int(sample['action_mask'].sum())==len(raw)*12
        reports.append(dict(index=index,episode=ep,step=t,valid_action_frames=len(raw),image_shapes=[list(i.shape) for i in sample['image']],action_roundtrip_max_error=err,instruction=sample['lang']))
    report=dict(passed=True,dataset=str(a.dataset),scope='CPU adapter and action roundtrip including episode tail; no optimizer or model training',examples=reports)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
