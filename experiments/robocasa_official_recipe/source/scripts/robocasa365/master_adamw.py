"""AdamW with FP32 master weights for the BF16 video DiT, FP32 action head unchanged.

Matches mixed-precision optimizer practice: the small warmup/video updates must not
be rounded away by updating BF16 parameters directly. Both experimental arms use it.
"""
import torch


class MasterAdamW(torch.optim.AdamW):
    def __init__(self, groups, **kwargs):
        self.pairs=[]
        master_groups=[]
        for group in groups:
            params=[]
            for real in group['params']:
                if real.dtype in (torch.bfloat16,torch.float16):
                    master=torch.nn.Parameter(real.detach().float().clone())
                    self.pairs.append((real,master))
                    params.append(master)
                else:params.append(real)
            master_groups.append({**group,'params':params})
        super().__init__(master_groups,**kwargs)

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        for real,_ in self.pairs:
            if set_to_none:real.grad=None
            elif real.grad is not None:real.grad.zero_()

    def step(self, closure=None):
        if closure is not None:raise ValueError('Closure unsupported in this explicit training loop')
        for real,master in self.pairs:
            master.grad=real.grad.detach().float() if real.grad is not None else None
        result=super().step()
        with torch.no_grad():
            for real,master in self.pairs:
                real.copy_(master);master.grad=None
        return result

    def state_dict(self):
        state=super().state_dict()
        state['fp32_master_weights']=[master.detach().cpu().clone() for _,master in self.pairs]
        return state

    def load_state_dict(self, state_dict):
        values=state_dict['fp32_master_weights']
        if len(values)!=len(self.pairs):raise ValueError('FP32 master parameter count changed')
        super().load_state_dict({k:v for k,v in state_dict.items() if k!='fp32_master_weights'})
        with torch.no_grad():
            for (real,master),value in zip(self.pairs,values):
                if master.shape!=value.shape:raise ValueError('FP32 master parameter shape changed')
                master.copy_(value);real.copy_(master)
