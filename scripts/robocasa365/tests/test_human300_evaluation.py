import unittest
from types import SimpleNamespace
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from scripts.robocasa365.human300_evaluation import joint_evaluation_mode


class CheckpointVideo(nn.Module):
    def __init__(self):
        super().__init__();self.layer=nn.Linear(2,2);self.gradient_checkpointing=True
    def disable_gradient_checkpointing(self):self.gradient_checkpointing=False
    def enable_gradient_checkpointing(self):self.gradient_checkpointing=True
    def forward(self,x):
        with torch.enable_grad():
            return checkpoint(self.layer,x,use_reentrant=False) if self.gradient_checkpointing else self.layer(x)


class EvaluationModeTest(unittest.TestCase):
    def make_model(self):
        model=nn.Module();model.video=CheckpointVideo();model.action_model=nn.Linear(2,2);model.frozen=nn.Linear(2,2).requires_grad_(False)
        model.backbone_interface=SimpleNamespace(extractor=SimpleNamespace(transformer=model.video))
        model.eval();model.video.train();model.action_model.train();return model
    def test_inference_tensor_checkpoint_conflict_is_prevented_and_training_restored(self):
        model=self.make_model()
        with self.assertRaisesRegex(RuntimeError,'Inference tensors'):
            with torch.inference_mode():model.video(torch.ones(1,2))
        with joint_evaluation_mode(model),torch.inference_mode():
            result=model.video(torch.ones(1,2));self.assertTrue(torch.isfinite(result).all());self.assertFalse(result.requires_grad)
        self.assertTrue(model.video.gradient_checkpointing)
        self.assertTrue(model.video.training);self.assertTrue(model.action_model.training);self.assertFalse(model.frozen.training)
        self.assertTrue(model.video.layer.weight.requires_grad);self.assertFalse(model.frozen.weight.requires_grad)
        model.video(torch.ones(1,2)).sum().backward();self.assertIsNotNone(model.video.layer.weight.grad)
    def test_exception_restores_state(self):
        model=self.make_model()
        with self.assertRaisesRegex(ValueError,'probe'):
            with joint_evaluation_mode(model):raise ValueError('probe')
        self.assertTrue(model.video.training);self.assertTrue(model.video.gradient_checkpointing)
        self.assertTrue(model.action_model.weight.requires_grad);self.assertFalse(model.frozen.weight.requires_grad)


if __name__=='__main__':unittest.main()
