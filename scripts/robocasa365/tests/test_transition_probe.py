import unittest

import numpy as np

from scripts.robocasa365.probe_atomic_cpu import transition_windows


class TransitionSelectionTest(unittest.TestCase):
    def test_navigation_ignores_short_switches(self):
        actions = np.zeros((40, 12))
        actions[5, 4] = 1
        actions[12:25, 4] = 1
        rows = transition_windows(actions, 'NavigateKitchen')
        self.assertEqual(rows[1]['transition_step'], 12)
        labels = actions[rows[1]['step']:rows[1]['step'] + 8, 4]
        np.testing.assert_array_equal(labels, [0] * 4 + [1] * 4)

    def test_cabinet_uses_last_sustained_open(self):
        actions = np.zeros((36, 12))
        actions[4:12, 11] = 1
        actions[18:32, 11] = 1
        actions[24, 11] = 0
        rows = transition_windows(actions, 'PickPlaceCounterToCabinet')
        self.assertEqual(rows[1]['transition_step'], 32)
        # Selected window reaches beyond the episode; the adapter must mask
        # padding, not drop the meaningful final transition.
        self.assertGreater(rows[1]['step'] + 16, len(actions))

    def test_no_transition_is_not_invented(self):
        actions = np.ones((24, 12))
        self.assertEqual(len(transition_windows(actions, 'NavigateKitchen')), 1)
        self.assertEqual(len(transition_windows(actions, 'PickPlaceCounterToCabinet')), 1)


if __name__ == '__main__':
    unittest.main()
