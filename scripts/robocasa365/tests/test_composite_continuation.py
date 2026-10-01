"""Reject identity/data mismatches before transferring a new Composite branch."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from omegaconf import OmegaConf

from scripts.robocasa365.action_warm_start import SEMANTIC_SOURCES, sha
from scripts.robocasa365.composite_continuation import load_composite_action_start


class CompositeContinuationTest(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.folder=Path(temporary.name);self.run=self.folder/'run';self.run.mkdir()
        self.checkpoint=self.folder/'action_step_010000.pt';self.audit_path=self.folder/'audit.json'
        self.base=self.folder/'base.pt'
        self.model=torch.nn.Module();self.model.action_model=torch.nn.Linear(2,2);self.model.video=torch.nn.Linear(2,2)
        self.cfg=OmegaConf.create(dict(dataset_py='robocasa365_datasets',action_horizon=16,max_state_dim=64,
            max_action_dim=32,image_size=[128,128],video_delta_indices=list(range(17)),action_video_freq_ratio=2,lerobot_version='v2.0'))
        self.stats={'action':{'min':[-1.]*12,'max':[1.]*12,'q01':[-.8]*12,'q99':[.8]*12}}
        self.split=dict(task_set='composite_seen',tasks=[dict(task=f'composite{i}',train_episodes=[0,1],validation_episodes=[2]) for i in range(16)],manifest_sha256='same-manifest')
        (self.folder/'normalization.json').write_text(json.dumps(self.stats));OmegaConf.save(self.cfg,self.folder/'data_config.yaml')
        root=Path(__file__).resolve().parents[3];hashes={}
        for name in SEMANTIC_SOURCES:
            target=self.run/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((root/name).read_bytes());hashes[name]=sha(target)
        (self.run/'source_hashes.json').write_text(json.dumps(hashes))
        self.weights={name:torch.full_like(p,3.) for name,p in self.model.named_parameters() if name.startswith('action_model.')}
        self.origin=dict(protocol='atomic_to_composite_action_warmstart_v1',global_step=50000,checkpoint_sha256='atomic50k',
                         optimizer_reset=True,scheduler_reset=True,sampler_reset=True,local_steps_start_at_zero=True)
        self.payload=dict(format_version=2,phase='action',phase_step=10000,global_step=10000,base_checkpoint=str(self.base),
            trained_state=self.weights,split=self.split,normalization_sha256=sha(self.folder/'normalization.json'),
            warm_start=self.origin,training_schedule={'initialization_checkpoint_sha256':'atomic50k'},
            optimizer={'old_state':'must_not_load'},distributed_rng=['old sampler must not load'])
        self.save()

    def save(self,wrapped=False):
        torch.save(self.payload,self.checkpoint)
        audit=dict(status='complete',global_step=10000,checkpoint_sha256=sha(self.checkpoint),
            checkpoint_bytes=self.checkpoint.stat().st_size,trained_tensors=2,trained_parameters=6,
            all_trained_weights_finite=True,all_adam_moments_finite=True,all_optimizer_steps_match=True,
            scheduler_step_matches=True,split_and_normalization_match=True,
            initialization=dict(protocol=self.origin['protocol'],checkpoint_sha256='atomic50k',source_global_step=50000,
                                local_initial_step=0,optimizer_scheduler_sampler_reset=True))
        if wrapped:audit=dict(status='complete',step=10000,cpu_readback=audit)
        self.audit_path.write_text(json.dumps(audit))

    def load(self,**overrides):
        values=dict(source_run=self.run,checkpoint_audit=self.audit_path,expected_step=10000,base_checkpoint=self.base,
                    target_config=self.cfg,target_stats=self.stats,target_split=self.split)
        values.update(overrides);return load_composite_action_start(self.model,self.checkpoint,**values)

    def test_action_only_initialization_keeps_rng_video_and_empty_optimizer(self):
        self.save(wrapped=True)
        video={name:p.detach().clone() for name,p in self.model.video.named_parameters()}
        rng=torch.get_rng_state().clone();optimizer=torch.optim.AdamW(self.model.action_model.parameters())
        lineage=self.load()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertFalse(optimizer.state)
        for name,p in self.model.action_model.named_parameters():self.assertTrue(torch.equal(p,self.weights['action_model.'+name]))
        for name,p in self.model.video.named_parameters():self.assertTrue(torch.equal(p,video[name]))
        self.assertEqual(lineage['global_step'],10000);self.assertTrue(lineage['local_steps_start_at_zero'])
        self.assertFalse(lineage['original16_demo_step2000_ab'])

    def test_wrong_step_phase_or_unverified_audit_rejected_before_mutation(self):
        before={name:p.detach().clone() for name,p in self.model.named_parameters()}
        with self.assertRaisesRegex(ValueError,'selected step'):self.load(expected_step=2000)
        for key,value in [('phase','partial_joint'),('phase_step',2000)]:
            old=self.payload[key];self.payload[key]=value;self.save()
            with self.assertRaisesRegex(ValueError,'selected step'):self.load()
            self.payload[key]=old
        self.save();audit=json.loads(self.audit_path.read_text());audit['checkpoint_sha256']='wrong'
        self.audit_path.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError,'audited weight'):self.load()
        for name,p in self.model.named_parameters():self.assertTrue(torch.equal(p,before[name]))

    def test_wrong_split_stats_or_input_contract_rejected(self):
        split=copy.deepcopy(self.split);split['tasks'][0]['train_episodes']=[0]
        with self.assertRaisesRegex(ValueError,'original Composite16 split'):self.load(target_split=split)
        stats=copy.deepcopy(self.stats);stats['action']['q01'][0]=-.4
        with self.assertRaisesRegex(ValueError,'complete normalization'):self.load(target_stats=stats)
        with self.assertRaisesRegex(ValueError,'input contract'):self.load(target_config=OmegaConf.merge(self.cfg,dict(image_size=[224,224])))

    def test_wrong_origin_or_incomplete_audit_rejected(self):
        self.payload['warm_start']=dict(self.origin,protocol='legacy16demo');self.save()
        with self.assertRaisesRegex(ValueError,'Atomic50k-to-Composite'):self.load()
        self.payload['warm_start']=dict(self.origin,sampler_reset=False);self.save()
        with self.assertRaisesRegex(ValueError,'local-step/reset'):self.load()
        self.payload['warm_start']=self.origin;self.save()
        audit=json.loads(self.audit_path.read_text());audit['all_optimizer_steps_match']=False
        self.audit_path.write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError,'successful check'):self.load()

    def test_foreign_and_nonfinite_weights_rejected(self):
        self.payload['trained_state']=dict(self.weights,**{'video.bias':torch.ones(2)});self.save()
        with self.assertRaisesRegex(ValueError,'complete action branch only'):self.load()
        weights=copy.deepcopy(self.weights);weights['action_model.bias'][0]=float('nan')
        self.payload['trained_state']=weights;self.save()
        with self.assertRaisesRegex(ValueError,'Invalid Composite continuation tensor'):self.load()

    def test_adapter_source_and_task_set_changes_rejected(self):
        self.payload['split']=dict(self.split,task_set='atomic_seen');self.save()
        with self.assertRaisesRegex(ValueError,'all16 Composite'):self.load(target_split=self.payload['split'])
        self.payload['split']=self.split;self.save()
        path=self.run/'source'/SEMANTIC_SOURCES[0];path.write_text(path.read_text()+'\n# changed\n')
        with self.assertRaisesRegex(ValueError,'adapter implementation'):self.load()


if __name__=='__main__':unittest.main()
