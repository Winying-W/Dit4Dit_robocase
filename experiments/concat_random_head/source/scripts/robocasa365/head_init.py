"""Replace the GR1-loaded action head with a freshly constructed DiT4DiT head.

Only `model.action_model` changes. Cosmos, VAE, text encoder and the GR1 config
(action 32 / state 64 / horizon 16 / DiT-B) remain exactly as loaded. The fresh head
is built by the same constructor DiT4DiT uses when training from scratch, under a
fixed seed, inside a forked CPU RNG so the baseline's training RNG streams are untouched
and every rank gets identical weights.
"""
import hashlib

import torch


def tensor_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def build_fresh_action_head(config, seed):
    from DiT4DiT.model.modules.action_model.ActionDiT import get_action_model
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return get_action_model(config=config)


@torch.no_grad()
def reinitialize_action_head(model, seed):
    head = model.action_model
    old = {k: v.detach().clone() for k, v in head.state_dict().items()}
    if any(v.is_cuda for v in old.values()):
        raise RuntimeError('Re-initialize the head before moving the model to GPU')
    fresh = build_fresh_action_head(model.config, seed).state_dict()
    if set(fresh) != set(old):
        raise ValueError(f'Fresh head keys differ from GR1 head: {sorted(set(fresh) ^ set(old))[:5]}')
    for key, value in fresh.items():
        if value.shape != old[key].shape:
            raise ValueError(f'Shape mismatch for {key}: {tuple(value.shape)} vs {tuple(old[key].shape)}')
        fresh[key] = value.to(dtype=old[key].dtype)
    result = head.load_state_dict(fresh, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    loaded = head.state_dict()
    for key, value in fresh.items():  # read back
        if not torch.equal(loaded[key], value):
            raise AssertionError(f'Readback mismatch for {key}')
    # Every learned matrix must now differ from GR1; otherwise the head was not actually reset.
    unchanged = [k for k, v in loaded.items() if v.ndim >= 2 and torch.equal(v, old[k])]
    if unchanged:
        raise AssertionError(f'Matrices still equal to GR1 after re-initialization: {unchanged[:5]}')
    return dict(
        mode='random', seed=int(seed), constructor='DiT4DiT get_action_model(config) / FlowmatchingActionHead',
        tensors=len(loaded), parameters=int(sum(p.numel() for p in head.parameters())),
        tensors_equal_to_gr1=sorted(k for k, v in loaded.items() if torch.equal(v, old[k])),
        gr1_head_sha256=tensor_digest(old), fresh_head_sha256=tensor_digest(loaded),
        action_dim=int(model.config.framework.action_model.action_dim),
        state_dim=int(model.config.framework.action_model.state_dim),
        action_horizon=int(model.config.framework.action_model.action_horizon),
    )
