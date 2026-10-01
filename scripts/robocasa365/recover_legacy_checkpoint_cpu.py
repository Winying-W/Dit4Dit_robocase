"""Read-only NAS recovery of the exact historical16-demo checkpoint.

Copies only the declared source and optional metadata into a new local recovery
directory, validates the original identity, then relocates the GR1 base reference.
No simulator, model inference, training, GPU allocation, or NAS write occurs.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.robocasa365.stage_legacy_ab_checkpoint import sha256, validate


def save(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--base-checkpoint',type=Path,required=True)
    parser.add_argument('--legacy-metadata',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Recovery must hide GPUs'
    args.output.mkdir(parents=True,exist_ok=False)
    status=args.output/'status.json'
    record=dict(started_utc=datetime.now(timezone.utc).isoformat(),source=str(args.source),
        status='checking_source',gpu_count=0,training_updates=0,new_policy_trials=0,nas_writes=0)
    save(status,record)
    try:
        if not args.source.is_file():
            raise FileNotFoundError(f'Original checkpoint not accessible through the requested NAS mount: {args.source}')
        import torch
        assert not torch.cuda.is_available()
        torch.set_num_threads(2)
        payload=torch.load(args.source,map_location='cpu',weights_only=False,mmap=True)
        validate(payload,args.source,args.legacy_metadata)
        assert len(payload['trained_state'])==247
        assert all(torch.isfinite(value).all() for value in payload['trained_state'].values())
        del payload
        raw=args.output/'original';raw.mkdir()
        checkpoint=raw/'action_step_002000.pt'
        before=args.source.stat()
        shutil.copyfile(args.source,checkpoint)
        after=args.source.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Source changed during recovery'
        original_sha=sha256(args.source)
        assert sha256(checkpoint)==original_sha
        sidecars={}
        for name in ['normalization.json','split.json','run_config.json','data_config.yaml','action_trainable.json']:
            source=args.source.parent/name
            if source.is_file():
                target=raw/name;shutil.copyfile(source,target)
                assert sha256(target)==sha256(source)
                sidecars[name]=sha256(target)
        save(args.output/'original_copy.json',dict(source=str(args.source),
            recovered_checkpoint=str(checkpoint),sha256=original_sha,bytes=checkpoint.stat().st_size,
            sidecar_sha256=sidecars,identity_validated=True,all247_action_tensors_finite=True))
        record.update(status='original_copied_relocating',original_sha256=original_sha)
        save(status,record)
        command=[sys.executable,str(Path(__file__).with_name('stage_legacy_ab_checkpoint.py')),
            '--source',str(checkpoint),'--base-checkpoint',str(args.base_checkpoint),
            '--legacy-metadata',str(args.legacy_metadata),'--output',str(args.output/'staged')]
        with (args.output/'relocation.log').open('w') as stream:
            subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=480)
        relocation=json.loads((args.output/'staged/relocation.json').read_text())
        assert relocation['status']=='complete' and relocation['source_sha256']==original_sha
        record.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
            relocation=str(args.output/'staged/relocation.json'),historical_ab_executed=False)
        save(status,record);print(json.dumps(record),flush=True)
        return 0
    except Exception as error:
        record.update(status='failed',completed_utc=datetime.now(timezone.utc).isoformat(),
            error_type=type(error).__name__,error=str(error),historical_ab_executed=False)
        save(status,record);print(json.dumps(record),flush=True)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
