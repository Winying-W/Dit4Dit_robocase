"""Synthetic checkpoint tests for semantic compatibility and initialization isolation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from omegaconf import OmegaConf

from scripts.robocasa365.action_warm_start import (
    SEMANTIC_SOURCES, sha, validate_contract, load_action_warm_start,
)


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.action_model=torch.nn.Linear(2,2)
        self.backbone=torch.nn.Linear(2,2)


class ActionWarmStartTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name);self.run=self.folder/'source_run';self.run.mkdir()
        self.checkpoint=self.folder/'action_step_050000.pt';self.base=self.folder/'gr1.pt'
        self.config=dict(dataset_py='robocasa365_datasets',action_horizon=16,max_state_dim=64,max_action_dim=32,
            image_size=[128,128],video_delta_indices=list(range(17)),action_video_freq_ratio=2,lerobot_version='v2.0')
        self.stats={'action':{'min':[-1.]*12,'max':[1.]*12,'q01':[-.8]*12,'q99':[.8]*12}}
        self.source_split=dict(task_set='atomic_seen',tasks=[dict(task=f'atomic{i}') for i in range(18)])
        self.target_split=dict(task_set='composite_seen',tasks=[dict(task=f'composite{i}') for i in range(16)])
        self.model=SmallModel();self.action={name:torch.full_like(param,3.) for name,param in self.model.named_parameters() if name.startswith('action_model.')}
        (self.folder/'normalization.json').write_text(json.dumps(self.stats))
        OmegaConf.save(OmegaConf.create(self.config),self.folder/'data_config.yaml')
        root=Path(__file__).resolve().parents[3];hashes={}
        for name in SEMANTIC_SOURCES:
            target=self.run/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((root/name).read_bytes());hashes[name]=sha(target)
        (self.run/'source_hashes.json').write_text(json.dumps(hashes))
        self.payload=dict(phase='action',global_step=50000,base_checkpoint=str(self.base),
            normalization_sha256=sha(self.folder/'normalization.json'),split=self.source_split,trained_state=self.action,
            optimizer={'synthetic_source_state_must_not_be_loaded':True},scheduler={'last_epoch':50000})
        self.save_checkpoint()

    def save_checkpoint(self):
        torch.save(self.payload,self.checkpoint)
        audits=self.run/'checkpoint_audits';audits.mkdir(exist_ok=True)
        (audits/'step_050000.json').write_text(json.dumps(dict(status='complete',global_step=50000,checkpoint_sha256=sha(self.checkpoint))))

    def load(self,stats=None):
        return load_action_warm_start(self.model,self.checkpoint,source_run=self.run,base_checkpoint=self.base,
            target_config=OmegaConf.create(self.config),target_stats=stats or self.stats,target_split=self.target_split)

    def test_only_action_weights_change_rng_and_fresh_optimizer_are_preserved(self):
        backbone={name:value.detach().clone() for name,value in self.model.backbone.named_parameters()}
        optimizer=torch.optim.AdamW(self.model.action_model.parameters());rng=torch.get_rng_state().clone()
        lineage=self.load()
        for name,value in self.model.named_parameters():
            if name.startswith('action_model.'):torch.testing.assert_close(value,self.action[name],rtol=0,atol=0)
        for name,value in self.model.backbone.named_parameters():torch.testing.assert_close(value,backbone[name],rtol=0,atol=0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertFalse(optimizer.state)
        self.assertTrue(lineage['optimizer_reset']);self.assertTrue(lineage['local_steps_start_at_zero'])
        self.assertEqual(lineage['global_step'],50000)

    def test_unused_quantile_differences_are_accepted(self):
        stats=copy.deepcopy(self.stats);stats['action']['q01']=[-.1]*12;stats['action']['q99']=[.2]*12
        self.load(stats)

    def test_changed_effective_action_scale_is_rejected(self):
        stats=copy.deepcopy(self.stats);stats['action']['min'][0]=-.5
        with self.assertRaisesRegex(ValueError,'normalization'):self.load(stats)

    def test_incomplete_action_branch_or_foreign_weights_are_rejected(self):
        for invalid in [{k:v for k,v in self.action.items() if k!='action_model.bias'},dict(self.action,**{'backbone.bias':torch.ones(2)})]:
            self.payload['trained_state']=invalid;self.save_checkpoint()
            with self.subTest(keys=list(invalid)),self.assertRaisesRegex(ValueError,'entire action branch'):self.load()

    def test_unaudited_weight_and_changed_adapter_source_are_rejected(self):
        (self.run/'checkpoint_audits/step_050000.json').write_text(json.dumps(dict(status='complete',global_step=50000,checkpoint_sha256='wrong')))
        with self.assertRaisesRegex(ValueError,'audited'):self.load()
        self.save_checkpoint()
        path=self.run/'source'/SEMANTIC_SOURCES[0];path.write_text(path.read_text()+'\n# changed\n')
        with self.assertRaisesRegex(ValueError,'implementation'):self.load()

    def test_wrong_input_contract_or_overlapping_tasks_are_rejected(self):
        altered=dict(self.config,image_size=[224,224])
        with self.assertRaisesRegex(ValueError,'input contract'):
            validate_contract(self.config,altered,self.stats,self.stats,self.source_split,self.target_split)
        overlap=copy.deepcopy(self.target_split);overlap['tasks'][0]['task']='atomic0'
        with self.assertRaisesRegex(ValueError,'disjoint'):
            validate_contract(self.config,self.config,self.stats,self.stats,self.source_split,overlap)


if __name__=='__main__':unittest.main()
