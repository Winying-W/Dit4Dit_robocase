"""Small, testable primitives used by the distributed RoboCasa365 trainer."""
import random
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist


def ensure_process_group(backend, rank, world_size):
    """Reuse the group created by DiT4DiT's import-time Accelerate logger.

    Under torchrun, importing baseframework initializes PartialState already.
    Initializing again raises before any dataset or model is loaded.
    """
    reused=dist.is_initialized()
    if not reused:
        dist.init_process_group(backend,timeout=timedelta(minutes=30))
    if dist.get_backend()!=backend or dist.get_rank()!=rank or dist.get_world_size()!=world_size:
        raise RuntimeError('Existing distributed process group does not match this run')
    return reused


def masked_ddp_scale(valid_elements, global_elements, world_size):
    """Undo DDP's rank mean to obtain the global valid-element mean gradient.

    The upstream loss is already divided by the local microbatch's mask sum.
    global_elements covers *all* ranks and accumulation microbatches in one update.
    """
    if valid_elements <= 0 or global_elements <= 0 or world_size < 1:
        raise ValueError('Mask counts and world size must be positive')
    return world_size * valid_elements / global_elements


def capture_rng(sampler, device=None):
    return dict(sampler=sampler.bit_generator.state, python=random.getstate(),
        numpy=np.random.get_state(), torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(device) if device is not None else None)


def restore_rng(state, sampler, device=None):
    sampler.bit_generator.state=state['sampler']
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if device is not None:
        torch.cuda.set_rng_state(state['cuda'],device)


def validate_resume_schedule(saved, requested):
    if saved != requested:
        raise ValueError('Exact distributed resume requires the same schedule, world size, microbatch and accumulation')
