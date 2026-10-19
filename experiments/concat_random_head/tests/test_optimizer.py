import copy,unittest
import torch
from scripts.robocasa365.master_adamw import MasterAdamW
class MasterTest(unittest.TestCase):
    def test_small_updates_and_exact_resume(self):
        torch.manual_seed(2)
        real=torch.nn.Parameter(torch.ones(7,dtype=torch.bfloat16))
        head=torch.nn.Parameter(torch.randn(7))
        opt=MasterAdamW([dict(params=[real],lr=1e-5),dict(params=[head],lr=1e-4)],betas=(.9,.95),eps=1e-8,weight_decay=1e-8,foreach=False)
        ref=torch.nn.Parameter(real.detach().float());rh=torch.nn.Parameter(head.detach().clone())
        base=torch.optim.AdamW([dict(params=[ref],lr=1e-5),dict(params=[rh],lr=1e-4)],betas=(.9,.95),eps=1e-8,weight_decay=1e-8,foreach=False)
        for _ in range(3):
            grad=torch.randn(7).bfloat16();hg=torch.randn(7)
            real.grad=grad;ref.grad=grad.float();head.grad=hg;rh.grad=hg.clone();opt.step();base.step()
            self.assertTrue(torch.equal(opt.pairs[0][1],ref));self.assertTrue(torch.equal(head,rh))
        self.assertFalse(torch.equal(ref,torch.ones_like(ref)))
        opt.zero_grad();self.assertIsNone(real.grad);self.assertIsNone(head.grad)
        state=copy.deepcopy(opt.state_dict());r2=torch.nn.Parameter(real.detach().clone());h2=torch.nn.Parameter(head.detach().clone())
        resumed=MasterAdamW([dict(params=[r2],lr=1e-5),dict(params=[h2],lr=1e-4)],betas=(.9,.95),eps=1e-8,weight_decay=1e-8,foreach=False)
        resumed.load_state_dict(state)
        for _ in range(5):
            grad=torch.randn(7).bfloat16();hg=torch.randn(7)
            real.grad=grad;r2.grad=grad.clone();head.grad=hg;h2.grad=hg.clone();opt.step();resumed.step()
        self.assertTrue(torch.equal(real,r2));self.assertTrue(torch.equal(head,h2));self.assertTrue(torch.equal(opt.pairs[0][1],resumed.pairs[0][1]))
if __name__=='__main__':unittest.main()
