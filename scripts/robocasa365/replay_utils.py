"""Restore the collection startup sequence for RoboCasa365 action playback.

collect_demos.py executes an unrecorded zero-action step after the collection
wrapper snapshots state[0]. Consequently states[1] is TWO control periods after
states[0]; subsequent states are one period apart. This helper recognizes only
that known signature. It never injects a recorded state during action playback.
"""
import math


def infer_unrecorded_startup_steps(state_times, control_timestep):
    """Recognize standard recordings or one omitted collection initialization step.

    Fail on underspecified / irregular timing rather than silently inventing
    actions. Times refer to MuJoCo states, not LeRobot frame timestamps.
    """
    times = [float(t) for t in state_times]
    dt = float(control_timestep)
    if len(times) < 3:
        raise ValueError('Need at least three simulation states to identify startup timing')
    if not math.isfinite(dt) or dt <= 0 or not all(map(math.isfinite, times)):
        raise ValueError('Simulation timestamps and positive control timestep must be finite')
    close = lambda a, b: math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-8)
    if any(not close(b-a, dt) for a,b in zip(times[1:-1], times[2:])):
        raise ValueError('Nonuniform recorded control intervals: no supported startup repair')
    first_delta = times[1]-times[0]
    if close(first_delta, dt):
        return 0
    if close(first_delta, 2*dt):
        return 1
    raise ValueError(f'Unsupported first state interval {first_delta}; expected {dt} or {2*dt}')


def apply_collection_startup(env, states, mode='auto', step_fn=None):
    """Apply startup immediately after official reset_to, before the GT actions.

    mode='none' retains original official behavior for a diagnostic baseline.
    This is for recorded-episode playback; it is not a blanket modification to
    fresh Gym resets or a policy rollout. step_fn may bypass a diagnostic wrapper.
    """
    if mode not in ('auto', 'none'):
        raise ValueError(f'Unknown startup mode: {mode}')
    count = 0 if mode == 'none' else infer_unrecorded_startup_steps(states[:, 0], env.control_timestep)
    before = float(env.sim.data.time)
    if not math.isclose(before, float(states[0, 0]), rel_tol=0, abs_tol=1e-8):
        raise ValueError('Startup must be applied immediately after restoring recorded state[0]')
    if count:
        import numpy as np
        step = step_fn or env.step
        # Exactly match collect_demos.py, including control mode=0.
        step(np.zeros(env.action_dim, dtype=np.float64))
    return dict(mode=mode, unrecorded_zero_steps=count, time_before=before,
                time_after=float(env.sim.data.time), control_timestep=float(env.control_timestep))
