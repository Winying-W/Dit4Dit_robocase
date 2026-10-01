"""Compare dataset RGB against the saved policy-protocol initial observation."""
import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from DiT4DiT.dataloader.robocasa365_datasets import Robocasa365DatasetAdapter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--episode',type=int,required=True)
    parser.add_argument('--replay',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=OmegaConf.create(dict(action_horizon=16,max_state_dim=64,max_action_dim=32,
        image_size=[128,128],video_delta_indices=[0],action_video_freq_ratio=2,
        video_backend='decord',lerobot_version='v2.0'))
    adapter=Robocasa365DatasetAdapter(args.dataset,config)
    raw=adapter.dataset.get_step_data(args.episode,0)
    with np.load(args.replay/'initial_observation.npz',allow_pickle=False) as saved:
        policy=saved['images'].copy()
    assert policy.shape==(3,256,256,3)
    rows=[]
    for index,key in enumerate(adapter.robot.video_keys):
        dataset=np.asarray(raw[key])[0].astype(np.float64)
        image=policy[index].astype(np.float64)
        assert dataset.shape==image.shape
        mae=float(np.abs(dataset-image).mean())
        flipped=float(np.abs(dataset-image[::-1]).mean())
        assert mae<10 and mae<flipped/2,(key,mae,flipped)
        rows.append(dict(episode=args.episode,camera=key,dataset_vs_policy_initial_rgb_mae=mae,
            vertically_flipped_mae=flipped))
    replay=json.loads((args.replay/'result.json').read_text())
    assert replay['episode']==args.episode
    result=dict(status='complete',task=replay['task'],rows=rows,
        scope='Dataset first RGB compared with the recorded Gym initial observation for this explicit GT episode; startup includes collection zero step. This is a camera protocol check, not all-task replay or learned-policy success.')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
