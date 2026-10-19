"""Explicit LR extension after a completed short experiment (no optimizer reset)."""
import math


def continuation_lrs(step, spec):
    start, end = spec['start_step'], spec['end_step']
    warmup = min(20, end-start)
    if not start < step <= end:
        raise ValueError((step, start, end))
    offset = step-start-1
    if offset < warmup:
        ratio = offset/max(1, warmup-1)
        return [old+(peak-old)*ratio for old,peak in zip(spec['initial_lrs'],spec['peak_lrs'])]
    progress = (step-start-warmup)/max(1,end-start-warmup)
    factor = .1+.9*.5*(1+math.cos(math.pi*progress))
    return [peak*factor for peak in spec['peak_lrs']]
