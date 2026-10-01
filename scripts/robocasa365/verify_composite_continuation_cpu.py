"""Load a genuine Composite checkpoint through the new guarded initializer.

Checks the real entire Action DiT branch on CPU, without Cosmos instantiation,
optimizer updates, GPU allocation or policy evaluation.
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
from scripts.robocasa365.composite_continuation import load_composite_action_start
from scripts.robocasa365.action_warm_start import sha


def digest(model):
    value=hashlib.sha256()
    for name,param in model.named_parameters():
        value.update(name.encode());value.update(param.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--checkpoint-audit',type=Path,required=True)
    parser.add_argument('--expected-step',type=int,required=True)
    parser.add_argument('--base-checkpoint',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()
    torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=False)
    config,_=read_mode_config(args.base_checkpoint)
    model=torch.nn.Module();model.action_model=get_action_model(config=dict_to_namespace(config)).float()
    assert all(p.device.type=='cpu' for p in model.parameters())
    model.non_action_sentinel=torch.nn.Parameter(torch.tensor([1.,2.,3.]))
    before=digest(model.action_model);sentinel=model.non_action_sentinel.detach().clone()
    optimizer=torch.optim.AdamW(model.action_model.parameters(),lr=1e-4,betas=(.9,.95))
    assert not optimizer.state
    rng=torch.get_rng_state().clone()
    lineage=load_composite_action_start(model,args.checkpoint,source_run=args.source_run,
        checkpoint_audit=args.checkpoint_audit,expected_step=args.expected_step,base_checkpoint=args.base_checkpoint,
        target_config=OmegaConf.load(args.prepared/'data_config.yaml'),
        target_stats=json.loads((args.prepared/'normalization.json').read_text()),
        target_split=json.loads((args.prepared/'split.json').read_text()))
    after=digest(model.action_model)
    assert before!=after and torch.equal(sentinel,model.non_action_sentinel)
    assert torch.equal(rng,torch.get_rng_state()) and not optimizer.state
    assert lineage['transferred_tensors']==247 and lineage['transferred_parameters']==163276320
    report=dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),lineage=lineage,
        action_digest_before=before,action_digest_after=after,non_action_sentinel_unchanged=True,
        torch_rng_unchanged=True,new_optimizer_state_empty=True,global_step_for_new_experiment=0,
        trained_tensors_loaded=247,trained_parameters_loaded=163276320,
        verifier_sha256=sha(Path(__file__)),initializer_sha256=sha(Path(__file__).with_name('composite_continuation.py')),
        new_optimizer_updates=0,new_policy_trials=0,
        scope='Real complete action branch initialized through production Composite contract on CPU. No Cosmos instantiation, model forward, CUDA training/resume, or success-rate claim.')
    (args.output/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
