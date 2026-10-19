"""Recover only committed optimizer steps; retain evidence of discarded work."""
import json
from pathlib import Path
import torch


def atomic_json(path, value):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def committed_step(output):
    path=Path(output)/'action_latest.pt'
    if not path.exists():return 0
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    required={'optimizer','scheduler','distributed_rng','trained_state','training_schedule','global_step'}
    if not required.issubset(saved):raise ValueError('Not a full resumable checkpoint')
    if len(saved['distributed_rng'])!=4:raise ValueError('Expected four saved rank RNG states')
    return int(saved['global_step'])


def prepare_attempt(output, token):
    output=Path(output);step=committed_step(output)
    if not 0<=step<=5000:raise ValueError(f'Unexpected checkpoint step {step}')
    metrics=output/'metrics.jsonl';rows=[]
    if metrics.exists():
        lines=metrics.read_text().splitlines()
        for i,line in enumerate(lines):
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:
                if i!=len(lines)-1:raise
        archive=output/'recovery';archive.mkdir(exist_ok=True)
        (archive/f'metrics_before_{token}.jsonl').write_bytes(metrics.read_bytes())
    if step:
        saved=torch.load(output/'action_latest.pt',map_location='cpu',weights_only=False,mmap=True)
        kept={int(r['global_step']):r for r in rows if int(r['global_step'])<=step}
        if saved.get('last_metrics'):
            assert int(saved['last_metrics']['global_step'])==step
            kept[step]=saved['last_metrics']
        if sorted(kept)!=list(range(1,step+1)):
            raise ValueError('Committed checkpoint has missing metric history; inspect before resuming')
        temp=metrics.with_suffix('.tmp');temp.write_text(''.join(json.dumps(kept[i])+'\n' for i in range(1,step+1)));temp.replace(metrics)
    elif metrics.exists():
        if (output/'health_100.json').exists():raise ValueError('Health checkpoint disappeared; do not silently restart from GR1')
        metrics.unlink()  # The complete prior file was archived above; no committed step existed.
    result=dict(token=token,global_step=step,resume=bool(step),discarded_uncommitted_steps=[r['global_step'] for r in rows if r['global_step']>step])
    atomic_json(output/f'prepared_{token}.json',result)
    return result


def ensure_policy_snapshot(output, step):
    output=Path(output);target=output/f'action_step_{step:06d}.pt'
    if target.exists():return target
    saved=torch.load(output/'action_latest.pt',map_location='cpu',weights_only=False,mmap=True)
    if saved['global_step']!=step:raise ValueError('Cannot reconstruct policy from a different step')
    temp=target.with_suffix('.tmp')
    torch.save({k:v for k,v in saved.items() if k not in ('optimizer','scheduler','distributed_rng')},temp);temp.replace(target)
    return target
