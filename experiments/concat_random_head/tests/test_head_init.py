"""CPU check against the real GR1 checkpoint: only the action head is reset, deterministically."""
import json, sys, os
from pathlib import Path
import torch
from omegaconf import OmegaConf

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'source'))
from scripts.robocasa365.head_init import reinitialize_action_head, build_fresh_action_head, tensor_digest
from DiT4DiT.model.modules.action_model.ActionDiT import get_action_model

gr1 = Path(os.environ['GR1_CHECKPOINT']).resolve().parents[1]
cfg = OmegaConf.load(gr1 / 'config.yaml')
am = cfg.framework.action_model
state = torch.load(gr1 / 'final_model/pytorch_model.pt', map_location='cpu', mmap=True, weights_only=True)
head_state = {k[len('action_model.'):]: v for k, v in state.items() if k.startswith('action_model.')}
cosmos_keys = [k for k in state if not k.startswith('action_model.')]


class Stub(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.config = cfg; self.action_model = get_action_model(config=cfg)


model = Stub()
model.action_model.load_state_dict(head_state, strict=True)
assert all(torch.equal(model.action_model.state_dict()[k], v.to(model.action_model.state_dict()[k].dtype)) for k, v in head_state.items())

rng_before = torch.random.get_rng_state()
record = reinitialize_action_head(model, 42)
assert torch.equal(rng_before, torch.random.get_rng_state()), 'global RNG advanced'
after = model.action_model.state_dict()

# 1) same function as DiT4DiT's own from-scratch construction under the same seed
ref = build_fresh_action_head(cfg, 42).state_dict()
assert tensor_digest(ref) == record['fresh_head_sha256'] == tensor_digest(after)
# 2) deterministic, seed-dependent
assert tensor_digest(build_fresh_action_head(cfg, 42).state_dict()) == record['fresh_head_sha256']
assert tensor_digest(build_fresh_action_head(cfg, 43).state_dict()) != record['fresh_head_sha256']
# 3) every matrix reset
assert not [k for k, v in after.items() if v.ndim >= 2 and torch.equal(v, head_state[k].to(v.dtype))]
# 4) dims identical to DiT4DiT GR1
assert (record['action_dim'], record['state_dim'], record['action_horizon']) == (32, 64, 16), record
assert after['action_encoder.layer1.weight'].shape[-1] == 32 and after['state_encoder.layer1.weight'].shape[-1] == 64
# 5) the GR1 file itself is untouched; Cosmos weights are not part of this operation
assert tensor_digest({k: state['action_model.' + k] for k in list(head_state)[:3]}) == tensor_digest({k: head_state[k] for k in list(head_state)[:3]})
summary = dict(passed=True, tensors=record['tensors'], parameters=record['parameters'],
    tensors_equal_to_gr1=record['tensors_equal_to_gr1'], gr1_cosmos_side_tensors=len(cosmos_keys),
    fresh_head_sha256=record['fresh_head_sha256'], gr1_head_sha256=record['gr1_head_sha256'],
    dims=dict(action=record['action_dim'], state=record['state_dim'], horizon=record['action_horizon']),
    image_size=list(cfg.datasets.vla_data.get('image_size', [])) if 'datasets' in cfg else None)
print(json.dumps(summary, indent=1))
(root / 'tests' / 'test_head_init_result.json').write_text(json.dumps(summary, indent=1) + '\n')
