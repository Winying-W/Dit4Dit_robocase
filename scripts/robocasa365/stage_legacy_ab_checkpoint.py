"""Validate and relocate the original16-demo step2000 for the historical A/B.

Same numeric step is insufficient: the Cabinet or overfit checkpoint is rejected.
This script needs the original file; it never reconstructs or invents its weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(state):
    digest=hashlib.sha256()
    for name in sorted(state):
        tensor=state[name].detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode());digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate(payload,source,legacy):
    expected_split=json.loads((legacy/'split.json').read_text())
    expected_config=json.loads((legacy/'run_config.json').read_text())
    expected_shapes=json.loads((legacy/'action_trainable.json').read_text())
    if payload.get('phase')!='action' or payload.get('global_step')!=2000 or payload.get('phase_step')!=2000:
        raise ValueError('Expected the original action-phase step2000')
    if payload.get('split')!=expected_split or payload.get('warm_start') is not None:
        raise ValueError('Not the original16-demo split/lineage; a Cabinet or overfit step2000 is not interchangeable')
    if payload['base_checkpoint']!=expected_config['base_checkpoint']:
        raise ValueError('Source does not retain the recorded original GR1 base reference')
    actual={name:list(tensor.shape) for name,tensor in payload['trained_state'].items()}
    if actual!=expected_shapes:
        raise ValueError('Original247 action tensor names/shapes do not match')
    expected_hash='77d32db8bb1b30e24ec7680585858118e4bda30a6f3102eb2392ef7d1011408a'
    source_statistics=source.parent/'normalization.json'
    if sha256(legacy/'normalization.json')!=expected_hash:
        raise ValueError('Original fixed16-demo normalization is missing or changed')
    if source_statistics.exists() and sha256(source_statistics)!=expected_hash:
        raise ValueError('Source checkpoint sidecar disagrees with the preserved original normalization')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--base-checkpoint',type=Path,required=True)
    parser.add_argument('--legacy-metadata',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    if not args.source.is_file():raise FileNotFoundError(f'Original trained checkpoint remains unavailable: {args.source}')
    payload=torch.load(args.source,map_location='cpu',weights_only=False,mmap=True)
    validate(payload,args.source,args.legacy_metadata)
    if args.validate_only:
        print('ORIGINAL_STEP2000_METADATA_VALIDATED');return
    expected_base='fc65e78ab7c3de040d2df9f41416640e49befec58c784f550ecdaf41ee167098'
    if sha256(args.base_checkpoint)!=expected_base:raise ValueError('Recovered base is not the recorded released GR1')
    if args.output.exists():raise FileExistsError('Choose a new staging directory; never overwrite staged evidence')
    args.output.mkdir(parents=True)
    original_digest=tensor_digest(payload['trained_state'])
    old_reference=payload['base_checkpoint'];payload['base_checkpoint']=str(args.base_checkpoint.resolve())
    target=args.output/'action_step_002000.pt';temporary=target.with_suffix('.tmp')
    torch.save(payload,temporary);temporary.replace(target)
    restored=torch.load(target,map_location='cpu',weights_only=False,mmap=True)
    assert tensor_digest(restored['trained_state'])==original_digest
    assert restored['split']==payload['split']
    for name in ['normalization.json','split.json']:
        shutil.copyfile(args.legacy_metadata/name,args.output/name)
    report=dict(status='complete',source_checkpoint=str(args.source.resolve()),source_sha256=sha256(args.source),
        staged_checkpoint=str(target.resolve()),staged_sha256=sha256(target),trained_tensors_sha256=original_digest,
        original_global_step=2000,train_episodes=16,validation_episodes=4,trainable_tensors=247,
        original_base_reference=old_reference,relocated_base_reference=payload['base_checkpoint'],base_sha256=expected_base,
        scope='Only the base-file reference is relocated; trained tensors are preserved and read back. '
              'Both A/B arms will warm-start these same weights and reset their optimizers identically. '
              'This preparation is not an executed A/B experiment.')
    (args.output/'relocation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
