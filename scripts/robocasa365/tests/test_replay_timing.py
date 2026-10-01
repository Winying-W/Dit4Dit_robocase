"""Regression checks for the observed missing collection step and safe detection."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from scripts.robocasa365.replay_utils import infer_unrecorded_startup_steps, apply_collection_startup


class RecordedTimingTests(unittest.TestCase):
    def test_actual_collection_signature(self):
        self.assertEqual(infer_unrecorded_startup_steps([.5000000000000003, .6000000000000004, .6500000000000005, .7000000000000005], .05), 1)

    def test_regular_recording_is_not_modified(self):
        self.assertEqual(infer_unrecorded_startup_steps([.5, .55, .6, .65], .05), 0)

    def test_rejects_missing_frames_elsewhere(self):
        with self.assertRaises(ValueError):
            infer_unrecorded_startup_steps([.5, .6, .7, .75], .05)

    def test_rejects_unexplained_initial_gap(self):
        with self.assertRaises(ValueError):
            infer_unrecorded_startup_steps([.5, .65, .7], .05)

    def test_rejects_unknown_or_invalid_timing(self):
        for times, dt in [([.5,.6],.05), ([.5,.6,.65],0), ([.5,float('nan'),.65],.05), ([.5,.4,.45],.05)]:
            with self.subTest(times=times,dt=dt), self.assertRaises(ValueError):
                infer_unrecorded_startup_steps(times,dt)


class StartupApplicationTests(unittest.TestCase):
    class States:
        def __init__(self, times): self.times = times
        def __getitem__(self, index):
            row, column = index
            assert column == 0
            return self.times[row]

    def make_env(self):
        env = SimpleNamespace(control_timestep=.05, action_dim=12,
                              sim=SimpleNamespace(data=SimpleNamespace(time=.5)), actions=[])
        def step(action):
            env.actions.append(action)
            env.sim.data.time += env.control_timestep
        env.step = step
        return env

    def test_one_real_zero_step_and_no_state_injection(self):
        env = self.make_env()
        numpy_stub = SimpleNamespace(zeros=lambda n,dtype: [0.0]*n, float64=float)
        states = self.States([.5,.6,.65])
        with patch.dict('sys.modules', {'numpy':numpy_stub}):
            report = apply_collection_startup(env, states)
        self.assertEqual(env.actions, [[0.0]*12])
        self.assertAlmostEqual(env.sim.data.time, .55)
        self.assertEqual(report['unrecorded_zero_steps'], 1)
        with self.assertRaises(ValueError):
            apply_collection_startup(env, states)

    def test_normal_recording_and_explicit_baseline_do_not_step(self):
        for times,mode in [([.5,.55,.6],'auto'),([.5,.6,.65],'none')]:
            env = self.make_env()
            report = apply_collection_startup(env, self.States(times), mode)
            self.assertEqual(env.actions, [])
            self.assertEqual(report['unrecorded_zero_steps'], 0)


if __name__ == '__main__': unittest.main()

