"""Load the genuine Atomic checkpoint through the trainer's warm-start helper.

Instantiates the entire real Action DiT branch and initializes it from released
GR1 weights. No Cosmos/video forward, optimizer update or GPU test is claimed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from DiT4DiT.model.framework.share_tools import read_mode_config, dict_to_namespace
from DiT4DiT.model.modules.action_model.ActionDiT import get_action_model
from scripts.robocasa365.action_warm_start import load_action_warm_start, sha


def digest_state(model):
    value=hashlib.sha256()
    for name,param in model.named_parameters():
        value.update(name.encode());value.update(param.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--base-checkpoint',type=Path,required=True)
    parser.add_argument('--target-prepared',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    config,_=read_mode_config(args.base_checkpoint)
    model=torch.nn.Module();model.action_model=get_action_model(config=dict_to_namespace(config))
    released=torch.load(args.base_checkpoint,map_location='cpu',weights_only=False,mmap=True)
    model.action_model.load_state_dict({k.removeprefix('action_model.'):v for k,v in released.items() if k.startswith('action_model.')},strict=True)
    del released
    before=digest_state(model.action_model)
    model.non_action_sentinel=torch.nn.Parameter(torch.tensor([1.,2.,3.]))
    sentinel=model.non_action_sentinel.detach().clone();rng=torch.get_rng_state().clone()
    lineage=load_action_warm_start(model,args.checkpoint,source_run=args.source_run,base_checkpoint=args.base_checkpoint,
        target_config=OmegaConf.load(args.target_prepared/'data_config.yaml'),
        target_stats=json.loads((args.target_prepared/'normalization.json').read_text()),
        target_split=json.loads((args.target_prepared/'split.json').read_text()))
    after=digest_state(model.action_model)
    assert before!=after and torch.equal(model.non_action_sentinel,sentinel) and torch.equal(rng,torch.get_rng_state())
    assert lineage['transferred_tensors']==247 and lineage['transferred_parameters']==163276320
    optimizer=torch.optim.AdamW(model.action_model.parameters(),lr=1e-4,betas=(.9,.95))
    assert not optimizer.state
    report=dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),lineage=lineage,
        released_gr1_action_digest=before,warm_started_action_digest=after,non_action_sentinel_unchanged=True,
        rng_unchanged=True,new_optimizer_state_empty=True,local_updates=0,new_policy_trials=0,source_sha256=sha(__file__),
        scope='Actual full Action DiT/State Encoder/Action Encoder/Decoder loaded from GR1 and then genuine audited Atomic50k through production helper. Cosmos was not instantiated. No model forward, distributed CUDA execution, optimizer step or success-rate claim.')
    (args.output/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='lineage'}),flush=True)


if __name__=='__main__':main()
