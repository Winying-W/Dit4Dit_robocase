"""Validate an explicit change in GPU count without relaxing data/model identity."""
import copy
import hashlib
import json


def scaleup_plan(saved, requested, step, *, source_transition=None):
    """Return audited restart metadata; never alter a checkpoint or running job.

    A later launcher must restore model/optimizer/scheduler, preserve existing
    rank RNG states and seed additional ranks using this plan. A changed rank
    count preserves the objective/global batch, not the exact future samples.
    """
    previous = copy.deepcopy(saved)
    target = copy.deepcopy(requested)
    before = previous['world_size']
    after = target['world_size']
    if before not in (1, 2) or after not in (2, 4) or after <= before:
        raise ValueError('Scale-up requires 1->2, 1->4 or 2->4 GPUs')
    def batch(config):
        return config['world_size'] * config['microbatch'] * config['accumulation']
    if batch(previous) != 64 or batch(target) != 64:
        raise ValueError('Scale-up must preserve global batch 64')
    if previous['microbatch'] != target['microbatch']:
        raise ValueError('Keep the verified microbatch size unchanged')
    if not 0 < step < previous['steps']:
        raise ValueError('Checkpoint step must be inside the original training budget')
    for key in ('world_size', 'accumulation'):
        previous.pop(key)
        target.pop(key)
    if source_transition is not None:
        # Any new restart entry point must name the exact old AND new hashes.
        for name, pair in source_transition.items():
            if previous['source_sha256'].get(name) != pair['before'] or target['source_sha256'].get(name) != pair['after']:
                raise ValueError('Source transition does not match the declared hashes')
            previous['source_sha256'][name] = pair['after']
    if previous != target:
        changed = sorted(k for k in set(previous) | set(target) if previous.get(k) != target.get(k))
        raise ValueError(f'Unexpected training change during scale-up: {changed}')
    seeds = {}
    for rank in range(before, after):
        digest = hashlib.sha256(f"human300-scaleup:{saved['seed']}:{step}:{before}:{after}:{rank}".encode()).digest()
        seeds[str(rank)] = int.from_bytes(digest[:4], 'little')
    return dict(parent_schedule_sha256=hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest(),
                parent_step=step, previous_world_size=before, world_size=after,
                accumulation=requested['accumulation'], global_batch=64,
                preserve_rank_rng=list(range(before)), new_rank_seeds=seeds,
                restore=['trained_state', 'optimizer', 'scheduler', 'step'],
                source_transition=source_transition or {},
                exact_sample_sequence_preserved=False)
