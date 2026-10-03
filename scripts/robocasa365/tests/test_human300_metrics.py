import unittest
import numpy as np
from scripts.robocasa365.human300_metrics import sampled_action_metrics


class SampledMetricsTest(unittest.TestCase):
    def test_padding_is_excluded_and_chunk_steps_are_separate(self):
        target=np.zeros((2,16,12));target[...,4]=-1;target[...,11]=-1
        prediction=target.copy();valid=np.ones_like(target,dtype=bool)
        valid[1,-1]=False;prediction[1,-1]=1000
        prediction[0,0,0]=.5;prediction[0,7,4]=1
        result=sampled_action_metrics(prediction,target,valid)
        self.assertAlmostEqual(result['overall']['base_command_mae'],.5/(31*4))
        self.assertAlmostEqual(result['overall']['mode_accuracy_at_0_5'],30/31)
        self.assertEqual(result['by_chunk_step']['8']['mode_accuracy_at_0_5'],.5)
        self.assertEqual(result['by_chunk_step']['16']['base_command_mae'],0)
        self.assertEqual(result['overall']['gripper_accuracy'],1)

    def test_no_valid_final_chunk_has_no_fabricated_score(self):
        target=np.zeros((1,16,12));valid=np.ones_like(target,dtype=bool);valid[:,-1]=False
        result=sampled_action_metrics(target,target,valid)
        self.assertEqual(result['by_chunk_step']['16']['valid_elements'],0)
        self.assertIsNone(result['by_chunk_step']['16']['mode_accuracy_at_0_5'])

    def test_invalid_shape_rejected(self):
        with self.assertRaises(ValueError):sampled_action_metrics(np.zeros((1,16,32)),np.zeros((1,16,12)),np.ones((1,16,12)))


if __name__=='__main__':unittest.main()
