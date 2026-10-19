"""CPU four-process numerical oracle for DDP accumulation and resume state."""
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from scripts.robocasa365.distributed_support import (
    masked_ddp_scale,capture_rng,restore_rng,validate_resume_schedule,
)


def batches(rank):
    generator=np.random.default_rng(142+rank)
    result=[]
    for micro in range(2):
        inputs=torch.tensor(generator.normal(size=(4,3)),dtype=torch.float64)
        labels=torch.tensor(generator.normal(size=(4,2)),dtype=torch.float64)
        mask=torch.zeros(4,2,dtype=torch.float64)
        mask[:(rank+micro)%4+1]=1
        result.append((inputs,labels,mask))
    return result


def worker(rank,folder):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',rank=rank,world_size=4,
        init_method='file://'+str(Path(folder)/'rendezvous'),timeout=timedelta(seconds=90))
    torch.manual_seed(9)
    module=torch.nn.Linear(3,2,bias=False,dtype=torch.float64)
    ddp=DDP(module,broadcast_buffers=False,gradient_as_bucket_view=True)
    optimizer=torch.optim.AdamW(ddp.parameters(),lr=.03,betas=(.9,.95),eps=1e-8,weight_decay=1e-8)
    local=batches(rank)
    total=torch.tensor(sum(float(mask.sum()) for _,_,mask in local),dtype=torch.float64)
    dist.all_reduce(total)
    optimizer.zero_grad()
    for micro,(inputs,labels,mask) in enumerate(local):
        with ddp.no_sync() if micro==0 else nullcontext():
            loss=(((ddp(inputs)-labels)**2)*mask).sum()/mask.sum()
            (loss*masked_ddp_scale(float(mask.sum()),float(total),4)).backward()
    gradient=module.weight.grad.detach().clone()
    optimizer.step()
    random.seed(71+rank);np.random.seed(81+rank);torch.manual_seed(91+rank)
    sampler=np.random.default_rng(np.random.SeedSequence([42,rank]))
    sampler.integers(0,100000,size=13)
    states=[None]*4;dist.all_gather_object(states,capture_rng(sampler))
    if rank==0:
        torch.save(dict(weights=module.state_dict(),optimizer=optimizer.state_dict(),rng=states),Path(folder)/'checkpoint.pt')
    dist.barrier()
    expected=(sampler.integers(0,100000,size=16),torch.rand(8),np.random.rand(8),random.random())
    saved=torch.load(Path(folder)/'checkpoint.pt',map_location='cpu',weights_only=False)
    restored=torch.nn.Linear(3,2,bias=False,dtype=torch.float64);restored.load_state_dict(saved['weights'])
    restored_optimizer=torch.optim.AdamW(restored.parameters(),lr=.03,betas=(.9,.95),eps=1e-8,weight_decay=1e-8)
    restored_optimizer.load_state_dict(saved['optimizer'])
    restored_sampler=np.random.default_rng(999)
    restore_rng(saved['rng'][rank],restored_sampler)
    actual=(restored_sampler.integers(0,100000,size=16),torch.rand(8),np.random.rand(8),random.random())
    for x,y in zip(actual,expected):np.testing.assert_array_equal(x,y)
    assert torch.equal(restored.weight,module.weight)
    for name,value in optimizer.state[module.weight].items():
        assert torch.equal(restored_optimizer.state[restored.weight][name],value)
    torch.save(dict(gradient=gradient,weights=module.weight.detach(),next_indices=actual[0]),Path(folder)/f'rank_{rank}.pt')
    dist.destroy_process_group()


class DistributedTrainingTest(unittest.TestCase):
    def test_four_rank_masked_accumulation_matches_concatenated_batch_and_resumes(self):
        with tempfile.TemporaryDirectory(prefix='robo365-ddp-test-') as folder:
            mp.spawn(worker,args=(folder,),nprocs=4,join=True)
            torch.manual_seed(9)
            reference=torch.nn.Linear(3,2,bias=False,dtype=torch.float64)
            samples=[sample for rank in range(4) for sample in batches(rank)]
            numerator=sum((((reference(x)-target)**2)*mask).sum() for x,target,mask in samples)
            denominator=sum(mask.sum() for _,_,mask in samples)
            (numerator/denominator).backward()
            expected_gradient=reference.weight.grad.clone()
            optimizer=torch.optim.AdamW(reference.parameters(),lr=.03,betas=(.9,.95),eps=1e-8,weight_decay=1e-8);optimizer.step()
            probes=[]
            for rank in range(4):
                actual=torch.load(Path(folder)/f'rank_{rank}.pt',map_location='cpu',weights_only=False)
                torch.testing.assert_close(actual['gradient'],expected_gradient,rtol=1e-12,atol=1e-12)
                torch.testing.assert_close(actual['weights'],reference.weight,rtol=1e-12,atol=1e-12)
                probes.append(tuple(actual['next_indices']))
            self.assertEqual(len(set(probes)),4)

    def test_resume_rejects_changed_batch_schedule(self):
        old=dict(world_size=4,microbatch=4,accumulation=4,steps=50000)
        validate_resume_schedule(old,dict(old))
        for key in old:
            changed=dict(old);changed[key]+=1
            with self.assertRaises(ValueError):validate_resume_schedule(old,changed)


if __name__=='__main__':unittest.main()
