"""CPU checks: HF-exact schedule, 224 mosaic from real data, sealed patch markers."""
import ast,json,math,sys,torch,os
from pathlib import Path
root=Path(__file__).resolve().parents[1];src=root/'source';sys.path.insert(0,str(src))
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
out={}
# 1. Schedule: our lambda vs HF with official base lr 3e-5 and min_lr 5e-7.
code=(src/'scripts/robocasa365/train_distributed.py').read_text()
ns={'math':math};exec(code[code.index('OFFICIAL_MIN_LR_RATE='):].split('\n')[0],ns)
floor=ns['OFFICIAL_MIN_LR_RATE'];assert abs(floor-1/60)<1e-15
steps,warmup=2000,50
def ours(s):
    if s<warmup:return float(s)/float(max(1,warmup))
    p=float(s-warmup)/float(max(1,steps-warmup));return max(0.,0.5*(1.+math.cos(math.pi*p))*(1-floor)+floor)
w=torch.nn.Parameter(torch.zeros(1));opt=torch.optim.AdamW([w],lr=3e-5)
hf=get_cosine_with_min_lr_schedule_with_warmup(opt,warmup,steps,min_lr=5e-7)
diff=0.
for s in range(steps+1):
    diff=max(diff,abs(opt.param_groups[0]['lr']/3e-5-ours(s)));opt.step();hf.step()
assert diff<1e-12,diff
out['schedule_max_abs_factor_diff']=diff;out['final_factor']=ours(steps)
assert '\n    model.config.trainer.repeated_diffusion_steps=4\n' in code
assert '((loss+video_loss)/args.accumulation).backward()' in code
ast.parse(code)
# 2. Real data: 17 sampled frames, every 2nd kept (action_video_freq_ratio=2, as official) -> 9 = current + 8 future.
from omegaconf import OmegaConf
from scripts.robocasa365 import multitask_data as md
import inspect;assert 'image_size=[224,224]' in inspect.getsource(md.prepare_multitask_dataset)
from DiT4DiT.dataloader.robocasa365_datasets import Robocasa365DatasetAdapter
manifest=json.loads(Path(os.environ['ROBOCASA_MANIFEST']).read_text())
row=manifest['tasks'][0]
cfg=OmegaConf.create(dict(dataset_py='robocasa365_datasets',action_horizon=16,max_state_dim=64,max_action_dim=32,image_size=[224,224],
    video_delta_indices=list(range(17)),action_video_freq_ratio=2,video_backend='decord',lerobot_version='v2.0',dataset_path=row['path']))
ds=Robocasa365DatasetAdapter(Path(row['path']),cfg);ex=ds[0]
shapes={tuple(x.shape) for x in ex['image']}
out.update(task=row['task'],frames=len(ex['image']),frame_shapes=sorted(shapes),action=list(ex['action'].shape),state=list(ex['state'].shape),
    action_mask_true_dims=int(ex['action_mask'][0].sum()))
assert shapes=={(3,224,672)} and len(ex['image'])==9 and ex['action'].shape==(16,32) and ex['state'].shape==(1,64)
assert int(ex['action_mask'][0].sum())==12
print(json.dumps(out,indent=1))
(root/'tests/test_official_recipe_result.json').write_text(json.dumps(dict(passed=True,**out),indent=1))
