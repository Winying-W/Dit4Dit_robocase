"""Opt-in independent RNG streams for a single-process frozen/thawed comparison.

The published model and loss math are unchanged. The auxiliary video branch
cannot advance the action branch's noise stream. Contexts restore caller RNGs
and the original forward method, including on exceptions.
"""
from contextlib import contextmanager
import hashlib

import torch


PROTOCOL='paired_video_action_rng_v1'


def paired_schedule(seed):
    return dict(paired_ab_protocol=PROTOCOL, paired_ab_seed=int(seed),
                gradient_clipping='per_optimizer_group')


def validate_paired_resume(previous, current):
    """Even an LR extension cannot change a resumed experiment's RNG protocol."""
    keys=('paired_ab_protocol','paired_ab_seed','gradient_clipping','sampling')
    if any(key in previous or key in current for key in keys):
        if any(previous.get(key)!=current.get(key) for key in keys):
            raise ValueError('Resume cannot change paired A/B randomness, seed, gradient clipping or sampling')
    for key,default in [('microbatch_size',1),('execution_device','cuda')]:
        if previous.get(key,default)!=current.get(key,default):
            raise ValueError(f'Resume cannot change {key}, including during LR extension')


def seed_generators(value, devices):
    torch.default_generator.manual_seed(value)
    for device in devices:
        with torch.cuda.device(device):torch.cuda.manual_seed(value)


def stream_seed(seed, step, micro, stream):
    if step<0 or micro<0 or stream not in ('backbone','action'):
        raise ValueError('Invalid paired RNG coordinates')
    value=f'{PROTOCOL}:{int(seed)}:{int(step)}:{int(micro)}:{stream}'.encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8],'little') & ((1<<63)-1)


@contextmanager
def paired_microbatch(model, *, seed, step, micro):
    """Not reentrant on one model; intended for the single-GPU A/B trainer."""
    action=model.action_model
    if getattr(action,'_paired_rng_active',False):
        raise RuntimeError('Overlapping paired contexts on one action module')
    devices=sorted({p.device.index for p in model.parameters() if p.device.type=='cuda'})
    if len(devices)>1:
        raise ValueError('The paired A/B context supports one model device')
    had_override='forward' in action.__dict__
    original=action.forward
    def seeded_forward(*args,**kwargs):
        with torch.random.fork_rng(devices=devices):
            seed_generators(stream_seed(seed,step,micro,'action'),devices)
            return original(*args,**kwargs)
    with torch.random.fork_rng(devices=devices):
        seed_generators(stream_seed(seed,step,micro,'backbone'),devices)
        action._paired_rng_active=True
        action.forward=seeded_forward
        try:
            yield
        finally:
            if had_override:action.forward=original
            else:del action.forward
            del action._paired_rng_active


def clip_paired_groups(groups, max_norm=1.):
    """Adding video parameters must not rescale the action branch's gradients."""
    norms=[torch.nn.utils.clip_grad_norm_(group['params'],max_norm,error_if_nonfinite=True) for group in groups]
    if not norms:raise ValueError('No optimizer groups to clip')
    return torch.linalg.vector_norm(torch.stack(norms))
