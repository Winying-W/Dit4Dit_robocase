"""Load independent human300 joint checkpoints without a GR1 base policy."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from DiT4DiT.model.framework import build_framework


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def load_checkpoint(checkpoint,device='cpu',*,allow_preflight=False,verify_base=True):
    checkpoint=Path(checkpoint);folder=checkpoint.parent
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    schedule=payload['schedule']
    if schedule['phase']!='joint':raise ValueError('Not a human300 joint checkpoint')
    if schedule['preflight'] and not allow_preflight:raise ValueError('A diagnostic checkpoint cannot be used as a formal human300 policy')
    if not 0<payload['step']<=schedule['steps']:raise ValueError('Invalid checkpoint step')
    manifest_path=folder/'dataset_manifest.json'
    if sha256(manifest_path)!=payload['split']['manifest_sha256']:raise ValueError('Data manifest mismatch')
    manifest=json.loads(manifest_path.read_text());stats_path=folder/'normalization.json'
    if sha256(stats_path)!=manifest['normalization_sha256']:raise ValueError('Normalization mismatch')
    run=json.loads((folder/'run_config.json').read_text())
    if run['schedule']!=schedule:raise ValueError('Run configuration and checkpoint disagree')
    cfg=OmegaConf.load(folder/'config.yaml');cfg.datasets.vla_data=OmegaConf.load(folder/'data_config.yaml')
    base=Path(cfg.framework.cosmos25.base_model);receipt=base/'CONVERSION_VERIFIED.json'
    if sha256(receipt)!=schedule['initialization_sha256']:raise ValueError('Cosmos initialization mismatch')
    provenance=json.loads(receipt.read_text())
    if verify_base:
        for rel,record in provenance['files'].items():
            file=base/rel
            if file.stat().st_size!=record['bytes'] or sha256(file)!=record['sha256']:raise ValueError(f'Changed base component: {file}')
    model=build_framework(cfg)
    model.action_model.float();model.backbone_interface.extractor.transformer.float()
    expected={n for n,_ in model.named_parameters() if n.startswith(('action_model.','backbone_interface.extractor.transformer.'))}
    if set(payload['trained_state'])!=expected:raise ValueError('Missing or extra trained tensors')
    result=model.load_state_dict(payload['trained_state'],strict=False)
    if result.unexpected_keys:raise ValueError(result.unexpected_keys)
    report=dict(checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha256(checkpoint),step=payload['step'],diagnostic=schedule['preflight'],
        initialization_sha256=schedule['initialization_sha256'],normalization_sha256=manifest['normalization_sha256'],trained_tensors=len(expected),
        source='Original Cosmos frozen components plus saved full video/action branches; no GR1 base policy',device=str(device))
    del payload
    model.requires_grad_(False);model.eval();model.config.framework.cosmos25.training='action';model.to(device)
    return model,cfg,json.loads(stats_path.read_text()),report


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-preflight',action='store_true');a=p.parse_args()
    _,_,_,report=load_checkpoint(a.checkpoint,allow_preflight=a.allow_preflight)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(passed=True,scope='Independent inference loader strict trained-key and provenance verification',**report),indent=2));print(json.dumps(report))


if __name__=='__main__':main()
