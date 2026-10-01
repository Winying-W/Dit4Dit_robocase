"""Real RoboCasa365 batch -> unchanged DiT4DiT forward + prediction. No training."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader,Subset
from DiT4DiT.dataloader.robocasa365_datasets import Robocasa365DatasetAdapter,collate_fn

p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--image-size',type=int,default=128);p.add_argument('--checkpoint',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
torch.manual_seed(0);np.random.seed(0)
cfg=OmegaConf.create(dict(dataset_py='robocasa365_datasets',dataset_path=str(a.dataset),action_horizon=16,
    max_state_dim=64,max_action_dim=32,image_size=[a.image_size,a.image_size],video_delta_indices=list(range(17)),
    action_video_freq_ratio=2,video_backend='decord',lerobot_version='v2.0'))
OmegaConf.save(cfg,a.output/'data_config.yaml')
ds=Robocasa365DatasetAdapter(a.dataset,cfg)
indices=[0,30,100,200,902,903]
loader=DataLoader(Subset(ds,indices),batch_size=2,collate_fn=collate_fn,num_workers=0)
report={'per_camera_resolution':a.image_size,'dataset':str(a.dataset),'length':len(ds),'batches':[],'optimizer_steps':0,'backward_calls':0}
for batch in loader:
    shapes={k:list(np.stack([b[k] for b in batch]).shape) for k in ('state','action','state_mask','action_mask')}
    shapes['video']=list(torch.stack([torch.stack(b['image']) for b in batch]).shape)
    for b in batch:
        assert np.isfinite(b['action']).all() and np.isfinite(b['state']).all()
        assert torch.isfinite(torch.stack(b['image'])).all()
    report['batches'].append(shapes)
# Explicit tail mask: episode 0 length=903, timestep=902 has exactly one valid action.
assert ds[902]['action_mask'].sum()==12
example=ds[100]
print('REAL_BATCH',json.dumps(report),flush=True)
(a.output/'dataloader.json').write_text(json.dumps(report,indent=2)+'\n')
# Save a reviewable mosaic and real numeric input.
from PIL import Image
Image.fromarray((example['image'][0].permute(1,2,0).numpy()*255).astype('uint8')).save(a.output/'input_mosaic.png')
np.savez(a.output/'real_batch.npz',state=example['state'],action=example['action'],action_mask=example['action_mask'])
if not a.checkpoint:
    raise SystemExit(0)
from DiT4DiT.model.framework.base_framework import baseframework
start=time.monotonic()
model=baseframework.from_pretrained(str(a.checkpoint))
model.backbone_interface.extractor.text_encoder.to(torch.bfloat16)
model=model.to('cuda').eval()
# Keep video/action weights fp32; frozen text encoder uses bf16 to fit L20. The stock model
# applies its own autocast. Use one FM sample; no backward or optimizer is created.
model.config.trainer.repeated_diffusion_steps=1
model.config.datasets.vla_data.video_delta_indices=list(range(17))
model.config.datasets.vla_data.action_video_freq_ratio=2
assert model.config.framework.action_model.state_dim==64
assert model.config.framework.action_model.action_dim==32
assert model.chunk_len==16
trace={}
def capture_backbone(module,args,out):
    trace['video_hidden']=list(out.hidden_states[-1].shape)
    assert torch.isfinite(out.hidden_states[-1]).all()
hook=model.backbone_interface.register_forward_hook(capture_backbone)
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    losses=model(examples=[example])
    loss_report={k:float(v.detach().float().cpu()) for k,v in losses.items()}
    assert 'action_loss' in losses and 'future_video_loss' in losses
    assert all(np.isfinite(v) for v in loss_report.values()),loss_report
    print('FORWARD_LOSSES',loss_report,flush=True)
    # Inference observes t=0 only; future demonstration frames are training labels.
    pred=model.predict_action(examples=[dict(image=[example['image'][0]],state=example['state'],lang=example['lang'])])
hook.remove()
actions=np.asarray(pred['normalized_actions'])
assert actions.shape==(1,16,32) and np.isfinite(actions).all(),actions.shape
np.save(a.output/'normalized_predictions.npy',actions)
report.update(passed=True,checkpoint=str(a.checkpoint),text_encoder_dtype='bfloat16',video_action_weight_dtype='float32',weight_purpose='GR1 pretrained initialization for interface verification only; not trained on RoboCasa365',
    prediction_shape=list(actions.shape),losses=loss_report,trace=trace,
    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,elapsed_seconds=time.monotonic()-start)
(a.output/'model_forward.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
