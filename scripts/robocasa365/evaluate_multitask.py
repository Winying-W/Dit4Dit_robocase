"""Run and audit declared final or development trials for an explicit task set."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import queue
import fcntl
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from scripts.robocasa365.audit_saved_actions import audit_phase
from scripts.robocasa365.artifact_identity import policy_identity
from scripts.robocasa365.scene_protocol import OFFICIAL, PROTOCOLS, protocol_spec, validate_protocol_records


def wilson(successes, count):
    if not 0 <= successes <= count or count <= 0:
        raise ValueError((successes,count))
    z=1.959963984540054; fraction=successes/count; d=1+z*z/count
    center=(fraction+z*z/(2*count))/d
    radius=z*math.sqrt(fraction*(1-fraction)/count+z*z/(4*count*count))/d
    return [max(0.,center-radius),min(1.,center+radius)]


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def policy_worker_environment(device=None, parent=None):
    """Independent policies must not inherit a torchrun/platform process group."""
    environment=dict(os.environ if parent is None else parent)
    distributed_keys={'RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK',
        'ROLE_RANK','ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'}
    for key in list(environment):
        if key in distributed_keys or key.startswith('TORCHELASTIC_'):
            environment.pop(key)
    if device is not None:
        environment['CUDA_VISIBLE_DEVICES']=device
        if device.isdigit():environment['MUJOCO_EGL_DEVICE_ID']=device
    return environment


def summarize(task_reports, planned_tasks, seeds, purpose='final', task_set='composite_seen', task_scope='full_official_task_set'):
    successes=sum(row['successes'] for row in task_reports)
    completed=sum(row['completed_trials'] for row in task_reports)
    complete=len(task_reports)==len(planned_tasks) and all(row['complete'] for row in task_reports)
    return dict(status='complete' if complete else 'incomplete', task_set=task_set, task_scope=task_scope, purpose=purpose,
        planned_tasks=planned_tasks, seeds=seeds, planned_trials=len(planned_tasks)*len(seeds),
        completed_trials=completed, successes=successes, tasks=task_reports,
        pooled_success_rate=successes/completed if complete else None,
        equal_task_macro_success_rate=sum(row['successes']/len(seeds) for row in task_reports)/len(planned_tasks) if complete else None,
        pooled_wilson_95=wilson(successes,completed) if complete else None,
        uncertainty_note='Wilson intervals describe these trials. Fixed equal trials per task make pooled and macro point estimates equal; this is not a result on unseen tasks or all 365 tasks.',
        scope='Fresh target Gym resets, full official horizons, learned policy commands, official success checker. GT/demo-init excluded; infrastructure failures retained separately. '+('Development seeds only; excluded from the final result.' if purpose=='development' else 'Final seeds only; development trials excluded.'))


def select_evaluation_tasks(manifest, requested=None, *, development=False, allow_subset=False):
    """Select development tasks without rewriting the checkpoint's data manifest."""
    tasks=manifest['tasks'];names=[row['task'] for row in tasks]
    if not names or len(names)!=len(set(names)):
        raise ValueError('Manifest task names must be nonempty and unique')
    scope=manifest.get('task_scope','full_official_task_set')
    if requested is None:return tasks,scope
    if not development or not allow_subset:
        raise ValueError('--tasks requires --development and --allow-task-subset')
    if not requested or len(requested)!=len(set(requested)) or not set(requested)<=set(names):
        raise ValueError('Requested tasks must be a nonempty unique subset of the training manifest')
    selected=[row for row in tasks if row['task'] in set(requested)]
    return selected,'declared_task_subset' if len(selected)<len(tasks) else scope


@contextmanager
def benchmark_lock(output):
    with (Path(output)/'evaluation.lock').open('a+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another evaluator or its policy worker still holds this benchmark lock') from exc
        yield lock.fileno()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sim-python',required=True)
    p.add_argument('--workers',type=int,choices=[1,4],default=1,help='One serial GPU or four independent GPU policy workers')
    p.add_argument('--max-attempts',type=int,choices=range(2,9),default=2,
                   help='Total attempts per task, including preserved attempts; increase explicitly after repairing infrastructure')
    p.add_argument('--development',action='store_true',help='Use only the declared selection seeds, never the held-out final seeds')
    p.add_argument('--allow-task-subset',action='store_true',help='Explicit diagnostic subset, labelled with its exact tasks; never treated as the full official task set')
    p.add_argument('--tasks',nargs='+',help='Development only: select tasks from the unchanged training manifest; requires --allow-task-subset')
    p.add_argument('--scene-protocol',choices=PROTOCOLS,default=OFFICIAL)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with benchmark_lock(a.output) as lock_fd:
        run(a,lock_fd)


def run(a,lock_fd):
    manifest=json.loads(a.manifest.read_text())
    split=json.loads((a.checkpoint.parent/'split.json').read_text())
    assert split['manifest_sha256']==hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    tasks,task_scope=select_evaluation_tasks(manifest,getattr(a,'tasks',None),
        development=a.development,allow_subset=a.allow_task_subset)
    planned_tasks=[row['task'] for row in tasks]
    assert len(planned_tasks)>0 and len(set(planned_tasks))==len(planned_tasks)
    if not a.allow_task_subset:
        assert manifest.get('task_scope','full_official_task_set')=='full_official_task_set'
        expected_count={'composite_seen':16,'atomic_seen':18}[manifest['task_set']]
        assert len(planned_tasks)==expected_count
    final_seeds=manifest['evaluation']['seeds']
    selection_seeds=manifest['evaluation']['selection_seeds']
    assert set(final_seeds).isdisjoint(selection_seeds)
    seeds=selection_seeds if a.development else final_seeds
    assert len(seeds)>=20 and len(set(seeds))==len(seeds)
    purpose='development' if a.development else 'final'
    stats=json.loads((a.checkpoint.parent/'normalization.json').read_text())
    reports=[]
    config=json.loads((a.checkpoint.parent/'run_config.json').read_text())
    artifacts=policy_identity(a.checkpoint,config['base_checkpoint'])
    identity=dict(policy_artifacts=artifacts,checkpoint=str(a.checkpoint.resolve()),manifest_sha256=split['manifest_sha256'],
                  seeds=seeds,tasks=planned_tasks,execute_horizon=8,purpose=purpose,
                  task_set=manifest['task_set'],task_scope=task_scope,
                  scene_protocol=protocol_spec(a.scene_protocol))
    identity_path=a.output/'identity.json'
    if identity_path.exists():
        previous=json.loads(identity_path.read_text())
        previous.setdefault('scene_protocol',protocol_spec())
        assert previous==identity, 'Cannot resume a benchmark with different policy or scene protocol'
    else:save(identity_path,identity)
    workers=getattr(a,'workers',1)
    if workers==4:
        import torch
        if torch.cuda.device_count()!=4:
            raise RuntimeError('Four-worker evaluation requires exactly four allocated visible GPUs')
        visible=os.environ.get('CUDA_VISIBLE_DEVICES')
        devices=visible.split(',') if visible else ['0','1','2','3']
        if len(devices)!=4 or len(set(devices))!=4:
            raise ValueError('Expected four distinct visible GPU identifiers')
    else:
        devices=[None]
    available=queue.SimpleQueue()
    for device in devices:available.put(device)
    by_task={}
    def work(task):
        device=available.get()
        try:
            environment=policy_worker_environment(device)
            return evaluate_task(a,task,seeds,stats,artifacts,lock_fd,environment)
        finally:available.put(device)
    def collect(row):
        by_task[row['task']]=row
        ordered=[by_task[name] for name in planned_tasks if name in by_task]
        save(a.output/'report.json',summarize(ordered,planned_tasks,seeds,purpose,manifest['task_set'],identity['task_scope']))
        print('MULTITASK_EVAL',row['task'],row['successes'],row['completed_trials'],flush=True)
    if workers==1:
        for task in tasks:collect(work(task))
    else:
        # Longer official horizons start first to reduce idle GPUs at the end.
        ordered=sorted(tasks,key=lambda task:task['horizon'],reverse=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(work,task) for task in ordered]
            for future in as_completed(futures):collect(future.result())
    reports=[by_task[name] for name in planned_tasks]
    report=summarize(reports,planned_tasks,seeds,purpose,manifest['task_set'],identity['task_scope']);save(a.output/'report.json',report)
    if report['status']!='complete':raise RuntimeError('Incomplete benchmark: infrastructure failures are not policy outcomes')


def evaluate_task(a,task,seeds,stats,artifacts,lock_fd,environment):
    task_dir=a.output/task['task'];task_dir.mkdir(exist_ok=True)
    attempt_limit=getattr(a,'max_attempts',2)
    if attempt_limit not in range(2,9):
        raise ValueError('Total attempt budget must be between 2 and 8')
    existing=[]
    for folder in task_dir.glob('attempt_*'):
        suffix=folder.name.removeprefix('attempt_')
        if not folder.is_dir() or not suffix.isdigit() or str(int(suffix))!=suffix:
            raise ValueError(f'Invalid existing attempt directory: {folder}')
        existing.append(int(suffix))
    if existing and (sorted(existing)!=list(range(max(existing)+1)) or max(existing)>=attempt_limit):
        raise ValueError('Existing attempts must be contiguous and inside the explicit total budget')
    completed={};attempts=[]
    for attempt in range(attempt_limit):
        pending=[seed for seed in seeds if seed not in completed]
        if not pending:break
        folder=task_dir/f'attempt_{attempt}'
        result_path=folder/'evaluation.json'
        if not folder.exists():
            folder.mkdir()
            cmd=[sys.executable,'-u','scripts/robocasa365/evaluate_policy.py',
                 '--checkpoint',str(a.checkpoint),'--dataset',task['path'],'--task',task['task'],
                 '--output',str(folder),'--sim-python',a.sim_python,'--execute-horizon','8',
                 '--scene-protocol',a.scene_protocol,'--seeds',*map(str,pending)]
            with (folder/'driver.log').open('w') as logfile:
                code=subprocess.run(cmd,stdout=logfile,stderr=subprocess.STDOUT,pass_fds=(lock_fd,),env=environment).returncode
            save(folder/'process.json',dict(returncode=code,command=cmd))
        if result_path.exists():
            verification=json.loads((folder/'checkpoint_verification.json').read_text())
            if verification['policy_artifacts']!=artifacts:
                raise RuntimeError(f'Policy artifacts differ from benchmark identity: {folder}')
            result=json.loads(result_path.read_text())
            validate_protocol_records(dict(scene_protocol=protocol_spec(a.scene_protocol)),verification,result)
            assert result['task']==task['task'] and result['initialization']=='fresh_gym_target'
            assert result['horizon']==result['official_horizon']==task['horizon']
            if result['episodes']:
                audit=audit_phase(folder,stats);assert audit['passed']
                save(folder/'independent_action_audit.json',audit)
            for trial in result['episodes']:
                seed=trial['seed'];assert seed in pending and seed not in completed
                assert trial['demo_episode'] is None
                completed[seed]=dict(**trial,artifact=str(folder/f"trial_{trial['trial']:03d}"))
            attempts.append(dict(attempt=attempt,status=result['status'],error=result.get('error'),
                                 completed_trials=len(result['episodes'])))
        else:
            attempts.append(dict(attempt=attempt,status='infrastructure_failure',
                                 error='No simulator evaluation report; inspect driver.log and process.json'))
    success=sum(trial['success'] for trial in completed.values())
    complete=set(completed)==set(seeds)
    row=dict(task=task['task'],complete=complete,planned_trials=len(seeds),completed_trials=len(completed),
        successes=success,success_rate=success/len(seeds) if complete else None,
        wilson_95=wilson(success,len(seeds)) if complete else None,
        missing_seeds=[seed for seed in seeds if seed not in completed],
        attempts=attempts,trials=[completed[seed] for seed in seeds if seed in completed])
    save(task_dir/'summary.json',row)
    return row


if __name__=='__main__':main()
