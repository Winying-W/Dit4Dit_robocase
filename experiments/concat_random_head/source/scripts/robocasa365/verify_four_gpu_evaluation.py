"""Bounded CUDA policy/simulator interface gate before long distributed training.

These four truncated rollouts are infrastructure checks, never success-rate trials.
Each worker runs the same policy and simulator entry points used by the benchmark.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from scripts.robocasa365.audit_saved_actions import audit_phase
from scripts.robocasa365.evaluate_multitask import benchmark_lock, save, policy_worker_environment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sim-python',required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if torch.cuda.device_count()!=4:
        raise RuntimeError('Evaluation gate requires four allocated GPUs')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    devices=visible.split(',') if visible else ['0','1','2','3']
    assert len(devices)==len(set(devices))==4
    manifest=json.loads(args.manifest.read_text())
    by_name={row['task']:row for row in manifest['tasks']}
    names=['PickPlaceCounterToCabinet','NavigateKitchen','OpenDrawer','TurnOnSinkFaucet']
    assert set(names).issubset(by_name)
    statistics=json.loads((args.checkpoint.parent/'normalization.json').read_text())
    with benchmark_lock(args.output) as lock_fd:
        def worker(pair):
            device,name=pair
            # A failed gate can be rerun without overwriting its evidence.
            parent=args.output/name;parent.mkdir(exist_ok=True)
            index=0
            while (parent/f'attempt_{index}').exists():index+=1
            folder=parent/f'attempt_{index}';folder.mkdir()
            environment=policy_worker_environment(device)
            command=[sys.executable,'-u','scripts/robocasa365/evaluate_policy.py',
                '--checkpoint',str(args.checkpoint),'--dataset',by_name[name]['path'],
                '--task',name,'--output',str(folder),'--sim-python',args.sim_python,
                '--execute-horizon','8','--seeds','90000','--max-steps','8']
            with (folder/'driver.log').open('w') as log:
                result=subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,
                    pass_fds=(lock_fd,))
            save(folder/'process.json',dict(returncode=result.returncode,command=command,gpu=device))
            if result.returncode:
                raise RuntimeError(f'GPU {device} policy/simulator gate failed; inspect {folder}')
            evaluation=json.loads((folder/'evaluation.json').read_text())
            assert evaluation['status']=='complete' and evaluation['horizon']==8
            assert evaluation['initialization']=='fresh_gym_target' and len(evaluation['episodes'])==1
            assert evaluation['episodes'][0]['policy_queries']>0
            audit=audit_phase(folder,statistics);assert audit['passed']
            save(folder/'independent_action_audit.json',audit)
            return dict(task=name,gpu=device,passed=True,artifact=str(folder),
                audited_steps=audit['audited_steps'])
        with ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(worker,zip(devices,names)))
        save(args.output/'acceptance.json',dict(status='complete',workers=results,
            checkpoint=str(args.checkpoint.resolve()),
            scope='Four concurrent CUDA policies and simulators, at most eight steps each. '
                  'Prediction, inverse normalization and actual simulator commands audited. '
                  'These truncated rollouts are excluded from all policy success rates.'))


if __name__=='__main__':main()
