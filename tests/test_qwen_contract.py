"""Qwen feature/attention contracts without downloading or loading a language model."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/concat_random_head/source'))
from DiT4DiT.model.framework.qwen_context import QwenProjection, current_views
from DiT4DiT.model.modules.action_model import ActionDiT

class QwenContract(unittest.TestCase):
    def test_projection_and_gradient_isolation(self):
        p=QwenProjection(20,16)
        cosmos=torch.randn(2,12,16)
        qwen=torch.randn(2,5,20,requires_grad=True)
        valid=torch.tensor([[1,1,1,0,0],[1,1,1,1,1]],dtype=torch.bool)
        context,mask=p(cosmos,qwen,valid)
        self.assertTrue(torch.equal(context[:,:12],cosmos))
        self.assertEqual(tuple(mask.shape),(2,1,17))
        self.assertTrue(torch.isneginf(mask[0,0,-2:]).all())
        context.square().sum().backward()
        self.assertIsNone(qwen.grad)
        self.assertTrue(all(x.grad is not None and torch.isfinite(x.grad).all() for x in p.parameters()))

    def test_future_frames_never_enter_qwen(self):
        current=np.random.default_rng(42).integers(0,256,(3,224,672),dtype=np.uint8)
        example={'image':[current,np.zeros_like(current)]}
        views=current_views(example)
        example['image'][1][:]=255
        for a,b in zip(views,current_views(example)):
            np.testing.assert_array_equal(a,b)
            self.assertEqual(a.shape,(224,224,3))

    def test_attention_padding_is_invisible(self):
        ActionDiT.DiTConfig['portable_test']={'input_embedding_dim':32,'attention_head_dim':8,'num_attention_heads':4}
        cfg=OmegaConf.create({'framework':{'action_model':dict(action_model_type='portable_test',hidden_size=64,
            state_dim=64,action_dim=32,future_action_window_size=15,action_horizon=16,add_pos_embed=True,max_seq_len=64,
            num_inference_timesteps=4,noise_beta_alpha=1.5,noise_beta_beta=1.,noise_s=.999,num_timestep_buckets=1000,
            diffusion_model_cfg=dict(cross_attention_dim=16,num_layers=4,output_dim=32,dropout=.2,final_dropout=True,
                                    interleave_self_attention=True,norm_type='ada_norm',positional_embeddings=None))}})
        model=ActionDiT.FlowmatchingActionHead(cfg).eval()
        cosmos=torch.randn(2,12,16);state=torch.randn(2,1,64)
        extra=torch.randn(2,5,16);mask=torch.zeros(2,1,17);mask[:,:,12:]=-torch.inf
        torch.manual_seed(21);base=model.predict_action(cosmos,state)
        torch.manual_seed(21);padded=model.predict_action(torch.cat((cosmos,extra),1),state,encoder_attention_mask=mask)
        torch.testing.assert_close(base,padded,rtol=1e-5,atol=2e-6)

if __name__=='__main__':unittest.main()
