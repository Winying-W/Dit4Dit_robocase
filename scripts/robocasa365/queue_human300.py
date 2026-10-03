"""Wait for idle local GPUs, verify recovery at the available GPU count, then run authorized 10k."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SETUP = ROOT/'runs/robocasa365_human300_setup'
QUEUE = SETUP/'gpu_queue'


def idle_gpus():
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                    '--format=csv,noheader'], text=True)
    occupied = {line.strip() for line in apps.splitlines()}
    free = []
    for line in gpu_rows.splitlines():
        index, uuid, memory, utilization = [part.strip() for part in line.split(',')]
        if uuid not in occupied and int(memory) < 512 and int(utilization) == 0:
            free.append(index)
    return free


def status(phase, **extra):
    path = QUEUE/'status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(phase=phase, pid=os.getpid(), updated=time.time(), **extra), indent=2))
    temp.replace(path)


def select_gpus(free, count=None):
    if count is None:
        count = next((n for n in (4,2,1) if len(free) >= n), 0)
    return free[:count] if count and len(free) >= count else []


def wait_for_gpus(phase, count=None):
    while True:
        free = idle_gpus()
        status(phase, free_gpus=free, required=count or 1)
        selected = select_gpus(free, count)
        if selected:
            time.sleep(3)
            if set(selected).issubset(idle_gpus()):
                return selected
        time.sleep(10)


def execute(command, env, log):
    with log.open('a') as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    QUEUE.mkdir(parents=True, exist_ok=True)
    with (QUEUE/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = SETUP/'full_data/manifest.json'
        acceptance = json.loads((SETUP/'full_data_acceptance/acceptance.json').read_text())
        assert acceptance['complete'] and not acceptance['diagnostic'] and len(acceptance['tasks']) == 300
        assert acceptance['manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
        preflight = SETUP/'gpu_available_rank_preflight_v1'
        formal = ROOT/'runs/robocasa365_human300_10k'
        if preflight.exists() or formal.exists():
            raise FileExistsError('Existing run requires inspection and explicit recovery; refusing duplicate launch')
        env = dict(os.environ)
        for key in list(env):
            if key in {'RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'} or key.startswith('TORCHELASTIC_'):
                env.pop(key)
        env.update(PYTHONPATH=str(ROOT), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   NO_ALBUMENTATIONS_UPDATE='1', OMP_NUM_THREADS='2', NCCL_DEBUG='WARN',
                   TMPDIR=str(ROOT/'.cache/human300-tmp'))
        gpus = wait_for_gpus('waiting_for_any_gpu')
        world = len(gpus)
        accumulation = 64 // world
        env['CUDA_VISIBLE_DEVICES'] = ','.join(gpus)
        status('available_rank_preflight', gpus=gpus, world_size=world, accumulation=accumulation)
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', f'--nproc_per_node={world}',
                   '-m', 'scripts.robocasa365.train_human300', '--config', str(ROOT/'DiT4DiT/config/robocasa/dit4dit_human300_10k.yaml'),
                   '--manifest', str(SETUP/'filtered_preflight_data/manifest.json'), '--output', str(preflight),
                   '--preflight', '--steps', '2', '--warmup', '1', '--microbatch', '1', '--accumulation', str(accumulation),
                   '--save-every', '1', '--validate-every', '2', '--validation-examples', '4', '--prediction-examples', '4']
        execute(command+['--stop-after','1'], env, QUEUE/'preflight.log')
        execute(command+['--resume',str(preflight/'joint_step_000001.pt')], env, QUEUE/'preflight_resume.log')
        result = json.loads((preflight/'status.json').read_text())
        latest = json.loads((preflight/'latest.json').read_text())
        assert result['status'] == 'complete' and result['step'] == 2 and result['diagnostic']
        assert latest['step'] == 2 and latest['checkpoint_readback_passed']
        assert 'sampled_actions' in result['validation']
        # Preflight processes have exited; check again before launching the formal job.
        gpus = wait_for_gpus('preflight_passed_waiting_for_formal_gpus', count=world)
        env['CUDA_VISIBLE_DEVICES'] = ','.join(gpus)
        status('formal_training', gpus=gpus, world_size=world, accumulation=accumulation, planned_steps=10000)
        execute(['bash',str(ROOT/'scripts/robocasa365/run_human300_10k.sh')], env, QUEUE/'formal_training.log')
        result = json.loads((formal/'status.json').read_text())
        assert result['status'] == 'complete' and result['step'] == 10000 and not result['diagnostic']
        status('training_finished_pending_final_audit', gpus=gpus, steps=10000)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # A duplicate process must not overwrite the live owner's status.
        if not isinstance(exc, BlockingIOError):
            status('failed', error=repr(exc))
        raise
