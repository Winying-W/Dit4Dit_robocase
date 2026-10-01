import unittest
import numpy as np
from scripts.robocasa365.prediction_metrics import action_metrics

class PredictionMetricsTest(unittest.TestCase):
    def test_physical_component_units_and_padding_mask(self):
        truth=np.zeros((2,16,12));truth[...,11]=-1
        pred=truth.copy();pred[...,5:8]=.2;pred[...,8:11]=.4
        mask=np.ones_like(truth,dtype=bool);mask[1,8:]=False
        pred[1,8:]=100
        r=action_metrics(pred,truth,mask)
        self.assertAlmostEqual(r['position_scaled_component_m']['mae'],.01)
        self.assertAlmostEqual(r['rotation_scaled_component_rad']['mae'],.2)
        self.assertEqual(r['position_command']['count'],24*3)
        self.assertEqual(r['clipped_arm_component_fraction'],0)
        self.assertEqual(r['gripper_accuracy'],1)

    def test_official_binary_threshold_and_arm_clipping(self):
        truth=np.zeros((1,2,12));truth[0,:,11]=[-1,1]
        pred=truth.copy();pred[0,:,11]=[.49,.5];pred[0,0,5]=2
        r=action_metrics(pred,truth,np.ones_like(truth,dtype=bool))
        self.assertEqual(r['gripper_balanced_accuracy'],1)
        self.assertAlmostEqual(r['position_command']['mae'],1/6)
        self.assertAlmostEqual(r['clipped_arm_component_fraction'],1/12)

if __name__=='__main__':unittest.main()
