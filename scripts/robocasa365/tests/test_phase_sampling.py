"""Check command windows and the exact resume/task-pairing invariants."""
import copy
import unittest

import numpy as np

from scripts.robocasa365.phase_sampling import PhaseBalancedTasks, switch_starts


def sampler(mode,empty=False):
    value=PhaseBalancedTasks.__new__(PhaseBalancedTasks)
    value.mode=mode;value.datasets=[object(),object(),object()]
    value.indices=[np.arange(100,200),np.arange(200,350),np.arange(350,510)]
    value.pools=[dict(initial=indices[:16],switch=np.array([],dtype=np.int64) if empty else indices[40:55]) for indices in value.indices]
    return value


class PhaseSamplingTest(unittest.TestCase):
    def test_switch_pool_matches_direct_valid_prefix_enumeration(self):
        rng=np.random.default_rng(24)
        sequences=[[-1]*20,[-1]*4+[1]*16,[1,-1],[1],[-1,1,-1,1]]
        sequences.extend(rng.choice([-1,1],size=n).tolist() for n in [9,17,31])
        for sequence in sequences:
            for horizon in [2,8,16]:
                expected=[i for i in range(len(sequence)) if len(set(sequence[i:i+horizon]))>1]
                np.testing.assert_array_equal(switch_starts(sequence,horizon),expected)

    def test_task_order_and_parent_rng_match_between_sampling_modes(self):
        left,right=sampler('paired_uniform'),sampler('initial_switch')
        a,b=np.random.default_rng(2809),np.random.default_rng(2809)
        left_rows=[left.sample_index(a) for _ in range(4000)]
        right_rows=[right.sample_index(b) for _ in range(4000)]
        self.assertEqual([r[0] for r in left_rows],[r[0] for r in right_rows])
        self.assertEqual(a.bit_generator.state,b.bit_generator.state)
        self.assertNotEqual(left_rows,right_rows)
        for task,index in right_rows:self.assertIn(index,right.indices[task])

    def test_restoring_parent_rng_exactly_restores_sampling(self):
        value=sampler('initial_switch');rng=np.random.default_rng(3)
        for _ in range(13):value.sample_index(rng)
        state=copy.deepcopy(rng.bit_generator.state)
        expected=[value.sample_index(rng) for _ in range(100)]
        resumed=np.random.default_rng();resumed.bit_generator.state=state
        self.assertEqual(expected,[sampler('initial_switch').sample_index(resumed) for _ in range(100)])

    def test_empty_switch_pool_falls_back_without_changing_task_stream(self):
        normal,empty=sampler('initial_switch'),sampler('initial_switch',empty=True)
        a,b=np.random.default_rng(42),np.random.default_rng(42)
        for _ in range(200):
            expected,actual=normal.sample_index(a),empty.sample_index(b)
            self.assertEqual(expected[0],actual[0]);self.assertIn(actual[1],empty.indices[actual[0]])
        self.assertEqual(a.bit_generator.state,b.bit_generator.state)


if __name__=='__main__':unittest.main()
