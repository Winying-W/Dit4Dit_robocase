"""Semantic checks on real data: inverse normalization, action order and camera/time layout."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from DiT4DiT.dataloader import build_dataloader
from DiT4DiT.dataloader.robocasa365_datasets import camera_mosaics
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
cfg=OmegaConf.load('DiT4DiT/config/robocasa/dit4dit_robocasa365_interface.yaml')
loader=build_dataloader(cfg,dataset_py=cfg.datasets.vla_data.dataset_py);ds=loader.dataset
b=next(iter(loader));ex=ds[100]
ep,step=ds.dataset.all_steps[100];raw=ds.dataset.get_step_data(ep,step)
gt=np.concatenate([raw[k] for k in ds.robot.action_keys],axis=-1)
state=np.concatenate([raw[k] for k in ds.robot.state_keys],axis=-1)
restored=ds.decode_actions(ex['action'],env_order=False)
err=float(np.max(np.abs(restored-gt)));assert err<1e-6,err
assert np.array_equal(ex['state'][:,:16],state.astype(np.float32))
assert np.allclose(ds.decode_actions(ex['action']),gt[:,list(ds.robot.dataset_to_env_action)],atol=1e-6)
# Camera-major input must become timestep-major mosaics, even without 'wrist' in names.
frames={k:np.stack([np.full((8,8,3),20*c+t,dtype=np.uint8) for t in range(3)]) for c,k in enumerate(ds.robot.video_keys)}
imgs=camera_mosaics(frames,ds.robot.video_keys,(8,8))
for t,img in enumerate(imgs):
 for c in range(3):assert torch.allclose(img[:,:,8*c:8*(c+1)],torch.full((3,8,8),(20*c+t)/255))
assert ds[902]['action_mask'].sum()==12
report={'passed':True,'action_roundtrip_max_error':err,'state_order_exact':True,'camera_time_layout':True,'tail_mask':True,
'factory_batch_size':len(b),'video_shape':list(torch.stack(b[0]['image']).shape),'state_shape':list(b[0]['state'].shape),'action_shape':list(b[0]['action'].shape)}
pred_path=a.output/'normalized_predictions.npy'
if pred_path.exists():
 pred=np.load(pred_path);env_actions=ds.decode_actions(pred)
 assert env_actions.shape==(1,16,12) and np.isfinite(env_actions).all()
 np.save(a.output/'env_order_predictions.npy',env_actions)
 report['decoded_prediction_shape']=list(env_actions.shape)
a.output.mkdir(parents=True,exist_ok=True);(a.output/'adapter_verified.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
