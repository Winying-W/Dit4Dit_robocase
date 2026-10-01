"""Reject mismatched data/lineage when branching Atomic50k into a new paired A/B."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from omegaconf import OmegaConf

from scripts.robocasa365.action_warm_start import SEMANTIC_SOURCES, sha
from scripts.robocasa365.atomic_continuation import load_atomic_action_start


class AtomicContinuationTest(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.folder=Path(temp.name);self.run=self.folder/'run';self.run.mkdir()
        self.checkpoint=self.folder/'action_step_050000.pt';self.base=self.folder/'base.pt'
        self.model=torch.nn.Module();self.model.action_model=torch.nn.Linear(2,2);self.model.video=torch.nn.Linear(2,2)
        self.cfg=OmegaConf.create(dict(dataset_py='robocasa365_datasets',action_horizon=16,max_state_dim=64,
            max_action_dim=32,image_size=[128,128],video_delta_indices=list(range(17)),action_video_freq_ratio=2,lerobot_version='v2.0'))
        self.stats={'action':{'min':[-1.]*12,'max':[1.]*12,'q01':[-.8]*12,'q99':[.8]*12}}
        self.split=dict(task_set='atomic_seen',tasks=[dict(task=f'atomic{i}',train_episodes=[0,1],validation_episodes=[2]) for i in range(18)],manifest_sha256='original')
        (self.folder/'normalization.json').write_text(json.dumps(self.stats));OmegaConf.save(self.cfg,self.folder/'data_config.yaml')
        root=Path(__file__).resolve().parents[3];hashes={}
        for name in SEMANTIC_SOURCES:
            target=self.run/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((root/name).read_bytes());hashes[name]=sha(target)
        (self.run/'source_hashes.json').write_text(json.dumps(hashes));(self.run/'checkpoint_audits').mkdir()
        self.weights={name:torch.full_like(p,3.) for name,p in self.model.named_parameters() if name.startswith('action_model.')}
        self.payload=dict(phase='action',global_step=50000,base_checkpoint=str(self.base),trained_state=self.weights,
            split=self.split,normalization_sha256=sha(self.folder/'normalization.json'),optimizer={'old_state':'must_not_load'})
        self.save()

    def save(self):
        torch.save(self.payload,self.checkpoint)
        (self.run/'checkpoint_audits/step_050000.json').write_text(json.dumps(dict(status='complete',global_step=50000,checkpoint_sha256=sha(self.checkpoint))))

    def load(self,**changes):
        kwargs=dict(source_run=self.run,base_checkpoint=self.base,target_config=self.cfg,target_stats=self.stats,target_split=self.split)
        kwargs.update(changes)
        return load_atomic_action_start(self.model,self.checkpoint,**kwargs)

    def test_action_only_load_preserves_video_rng_and_fresh_optimizer(self):
        video={name:p.detach().clone() for name,p in self.model.video.named_parameters()}
        rng=torch.get_rng_state().clone();optimizer=torch.optim.AdamW(self.model.action_model.parameters())
        lineage=self.load()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertFalse(optimizer.state)
        for name,p in self.model.named_parameters():
            if name.startswith('action_model.'):self.assertTrue(torch.equal(p,self.weights[name]))
        for name,p in self.model.video.named_parameters():self.assertTrue(torch.equal(p,video[name]))
        self.assertTrue(lineage['optimizer_reset']);self.assertTrue(lineage['local_steps_start_at_zero'])
        self.assertFalse(lineage['original16_demo_step2000_ab'])

    def test_changed_split_or_manifest_rejected(self):
        for field in ['episodes','manifest']:
            split=copy.deepcopy(self.split)
            if field=='episodes':split['tasks'][0]['train_episodes']=[0]
            else:split['manifest_sha256']='different'
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'original Atomic18 split'):
                self.load(target_split=split)

    def test_changed_statistics_or_input_contract_rejected(self):
        stats=copy.deepcopy(self.stats);stats['action']['q01'][0]=-.5
        with self.assertRaisesRegex(ValueError,'identical complete normalization'):self.load(target_stats=stats)
        cfg=OmegaConf.merge(self.cfg,dict(image_size=[224,224]))
        with self.assertRaisesRegex(ValueError,'input contract'):self.load(target_config=cfg)

    def test_wrong_phase_step_or_unverified_checkpoint_rejected(self):
        for phase,step in [('action',2000),('partial_joint',50000)]:
            self.payload.update(phase=phase,global_step=step);self.save()
            with self.subTest(phase=phase,step=step),self.assertRaisesRegex(ValueError,'action step50000'):self.load()
        self.payload.update(phase='action',global_step=50000);self.save()
        (self.run/'checkpoint_audits/step_050000.json').write_text(json.dumps(dict(status='complete',global_step=50000,checkpoint_sha256='wrong')))
        with self.assertRaisesRegex(ValueError,'audited50k'):self.load()

    def test_foreign_or_nonfinite_weights_rejected(self):
        self.payload['trained_state']=dict(self.weights,**{'video.bias':torch.ones(2)});self.save()
        with self.assertRaisesRegex(ValueError,'complete action branch only'):self.load()
        weights=copy.deepcopy(self.weights);weights['action_model.bias'][0]=float('nan');self.payload['trained_state']=weights;self.save()
        with self.assertRaisesRegex(ValueError,'Invalid Atomic continuation tensor'):self.load()

    def test_changed_adapter_source_rejected(self):
        p=self.run/'source'/SEMANTIC_SOURCES[0];p.write_text(p.read_text()+'\n# changed\n')
        with self.assertRaisesRegex(ValueError,'adapter implementation'):self.load()


if __name__=='__main__':unittest.main()
