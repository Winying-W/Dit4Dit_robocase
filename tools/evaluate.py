"""Evaluate every task using the final snapshot, in a separate simulator environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=['robocasa_official_recipe','concat_random_head'],required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--cosmos-model',type=Path,required=True)
    p.add_argument('--sim-python',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',default='0')
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(100,110)))
    a=p.parse_args()
    source=ROOT/'experiments'/a.experiment/'source'
    config=json.loads((a.checkpoint.parent/'run_config.json').read_text())
    expected='random' if a.experiment=='concat_random_head' else 'gr1'
    if config['training_schedule']['action_head_init'] != expected or bool(config['qwen']) != (expected=='random'):
        raise ValueError('Checkpoint belongs to a different experiment')
    env=os.environ.copy()
    for key in list(env):
        if key in {'RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'} or key.startswith('TORCHELASTIC_'):
            env.pop(key,None)
    env.update(PYTHONPATH=str(source),COSMOS_MODEL_DIR=str(a.cosmos_model.resolve()),CUDA_VISIBLE_DEVICES=a.gpu,
               MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0',NO_ALBUMENTATIONS_UPDATE='1')
    for row in json.loads(a.manifest.read_text())['tasks']:
        target=a.output.resolve()/row['task']
        if target.exists():
            raise FileExistsError(f'Use a fresh evaluation output: {target}')
        subprocess.run([sys.executable,'-m','scripts.robocasa365.evaluate_policy',
            '--checkpoint',str(a.checkpoint.resolve()),'--dataset',row['path'],'--output',str(target),
            '--sim-python',str(a.sim_python.absolute()),'--task',row['task'],'--execute-horizon','8',
            '--seeds',*map(str,a.seeds)],cwd=source,env=env,check=True)

if __name__=='__main__':main()
