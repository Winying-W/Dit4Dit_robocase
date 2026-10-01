"""Controlled randomness, cleanup and clipping semantics for the single-GPU A/B."""
import unittest

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from scripts.robocasa365.paired_randomness import (
    paired_microbatch, stream_seed, clip_paired_groups, paired_schedule, validate_paired_resume,
)


class Head(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.ones(8))

    def forward(self, features):
        noise=torch.randn_like(features)
        return checkpoint(lambda x:x*self.weight+noise,features,use_reentrant=False).sum()


class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.action_model=Head()

    def forward(self, auxiliary=False):
        features=torch.randn(8)
        if auxiliary:torch.randn(937)
        return features,self.action_model(features)


class PairedRandomnessTest(unittest.TestCase):
    def test_auxiliary_draws_do_not_change_action_loss_or_gradients(self):
        model=Toy();results=[]
        for auxiliary in (False,True):
            model.zero_grad(set_to_none=True)
            with paired_microbatch(model,seed=42,step=13,micro=2):features,loss=model(auxiliary)
            loss.backward()
            results.append((features,loss.detach(),model.action_model.weight.grad.clone()))
        for first,second in zip(*results):torch.testing.assert_close(first,second,rtol=0,atol=0)

    def test_context_restores_rng_and_forward_on_exception(self):
        model=Toy();torch.manual_seed(81);before=torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with paired_microbatch(model,seed=42,step=1,micro=0):
                model(True);raise RuntimeError('intentional')
        self.assertNotIn('forward',model.action_model.__dict__)
        self.assertFalse(hasattr(model.action_model,'_paired_rng_active'))
        torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)

    def test_existing_override_is_preserved_and_nested_calls_are_rejected(self):
        model=Toy();override=lambda x:x.sum();model.action_model.forward=override
        with paired_microbatch(model,seed=42,step=1,micro=0):
            with self.assertRaisesRegex(RuntimeError,'Overlapping'):
                with paired_microbatch(model,seed=42,step=1,micro=0):pass
        self.assertIs(model.action_model.forward,override)

    def test_update_micro_and_domains_have_distinct_reproducible_seeds(self):
        keys=[stream_seed(42,step,micro,stream) for step in (1,2) for micro in (0,1) for stream in ('action','backbone')]
        self.assertEqual(len(set(keys)),8)
        self.assertEqual(stream_seed(42,2,1,'action'),stream_seed(42,2,1,'action'))

    def test_video_gradient_does_not_change_clipped_action_gradient(self):
        action=nn.Parameter(torch.zeros(2));video=nn.Parameter(torch.zeros(2))
        action.grad=torch.tensor([3.,4.]);clip_paired_groups([dict(params=[action])]);reference=action.grad.clone()
        action.grad=torch.tensor([3.,4.]);video.grad=torch.tensor([60.,80.])
        norm=clip_paired_groups([dict(params=[action]),dict(params=[video])])
        torch.testing.assert_close(action.grad,reference,rtol=0,atol=0)
        self.assertGreater(float(norm),100.)

    def test_resume_allows_lr_extension_but_rejects_randomness_or_clipping_changes(self):
        old=dict(action_steps=1000,**paired_schedule(42))
        new=dict(action_steps=2000,**paired_schedule(42))
        validate_paired_resume(old,new)
        validate_paired_resume(dict(action_steps=1000),dict(action_steps=2000))
        for changed in ({},paired_schedule(43),dict(paired_schedule(42),gradient_clipping='global')):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_paired_resume(old,changed)
        with self.assertRaises(ValueError):validate_paired_resume({},new)

    def test_lr_extension_cannot_change_sampling_microbatch_or_device(self):
        old=dict(action_steps=2000,microbatch_size=4,sampling=dict(mode='initial_switch'),**paired_schedule(42))
        extended=dict(old,action_steps=3000)
        validate_paired_resume(old,extended)
        for change in [dict(sampling=dict(mode='paired_uniform')),dict(microbatch_size=1),
                       dict(execution_device='cpu_preflight')]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_paired_resume(old,dict(extended,**change))
        validate_paired_resume(paired_schedule(42),dict(paired_schedule(42),execution_device='cuda',microbatch_size=1))


if __name__=='__main__':unittest.main()
