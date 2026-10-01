"""Protect small FP32 checkpoint updates across training/inference reconstruction."""
from types import SimpleNamespace
import unittest

import torch

from scripts.robocasa365.video_precision import apply_saved_video_precision


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks=torch.nn.ModuleList([torch.nn.Linear(2,2,bias=False).to(torch.bfloat16) for _ in range(18)])
        self.backbone_interface=SimpleNamespace(extractor=SimpleNamespace(
            transformer=SimpleNamespace(transformer_blocks=self.blocks)))
        with torch.no_grad():
            for block in self.blocks:block.weight.fill_(1)


class VideoPrecisionTest(unittest.TestCase):
    def test_saved_small_updates_survive_model_reconstruction(self):
        # 2e-5 at magnitude1 is below a BF16 ULP, yet must survive in the
        # trainable parameters so subsequent optimizer steps can accumulate it.
        state={f'blocks.{i}.weight':torch.full((2,2),1.00002,dtype=torch.float32) for i in (16,17)}
        baseline=SmallModel();baseline.load_state_dict(state,strict=False)
        self.assertTrue(torch.equal(baseline.blocks[16].weight.float(),torch.ones(2,2)))
        rebuilt=SmallModel()
        precision=apply_saved_video_precision(rebuilt,dict(training_schedule={'selected_video_parameter_precision':'float32'}))
        rebuilt.load_state_dict(state,strict=False)
        self.assertEqual(precision,'float32')
        for index in (16,17):self.assertTrue(torch.equal(rebuilt.blocks[index].weight,state[f'blocks.{index}.weight']))
        self.assertEqual(rebuilt.blocks[15].weight.dtype,torch.bfloat16)

    def test_historical_checkpoint_preserves_base_precision(self):
        model=SmallModel()
        self.assertEqual(apply_saved_video_precision(model,{}),'base_dtype')
        self.assertTrue(all(block.weight.dtype==torch.bfloat16 for block in model.blocks))

    def test_unknown_precision_rejected_before_weights_change(self):
        model=SmallModel()
        with self.assertRaises(ValueError):
            apply_saved_video_precision(model,dict(training_schedule={'selected_video_parameter_precision':'float16'}))
        self.assertTrue(all(block.weight.dtype==torch.bfloat16 for block in model.blocks))


if __name__=='__main__':unittest.main()
