"""Exercise one real LIBERO environment, including GPU rendering and stepping."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

out = Path('results/preflight')
out.mkdir(parents=True, exist_ok=True)
suite = benchmark.get_benchmark_dict()['libero_spatial']()
task = suite.get_task(0)
env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file), camera_heights=256, camera_widths=256)
try:
    env.seed(7)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(0)[0])
    for _ in range(10):
        obs, _, done, _ = env.step([0.0]*6+[-1.0])
    shapes = {k:list(obs[k].shape) for k in ['agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_gripper_qpos']}
    assert obs['agentview_image'].shape == (256,256,3)
    assert np.isfinite(obs['robot0_eef_pos']).all()
    Image.fromarray(obs['agentview_image'][::-1,::-1]).save(out/'libero_task0.png')
    x=torch.randn(256,256,device='cuda')
    y=x@x.T
    assert torch.isfinite(y).all()
    report={'status':'passed','task':task.language,'environment_steps':10,'observation_shapes':shapes,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}
    (out/'simulation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
finally:
    env.close()
