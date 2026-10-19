import json,tempfile,unittest
from pathlib import Path
import torch
from scripts.robocasa365.recovery import prepare_attempt,committed_step,ensure_policy_snapshot
class RecoveryTest(unittest.TestCase):
    def test_reconcile_committed_step_and_rebuild_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            saved=dict(global_step=3,optimizer={'fp32_master_weights':[torch.ones(2)]},scheduler={'last_epoch':3},distributed_rng=[{} for _ in range(4)],trained_state={'weight':torch.ones(2)},training_schedule={'steps':50000},last_metrics={'global_step':3,'loss':.2})
            torch.save(saved,out/'action_latest.pt')
            (out/'metrics.jsonl').write_text('\n'.join(json.dumps({'global_step':i,'loss':.3}) for i in [1,2,4])+'\n{"global_')
            result=prepare_attempt(out,'test')
            self.assertEqual(result['global_step'],3);self.assertEqual(result['discarded_uncommitted_steps'],[4])
            rows=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['global_step'] for r in rows],[1,2,3]);self.assertEqual(rows[-1]['loss'],.2)
            policy=ensure_policy_snapshot(out,3);p=torch.load(policy,weights_only=False)
            self.assertNotIn('optimizer',p);self.assertEqual(p['global_step'],3)
            self.assertEqual(committed_step(out),3)
            self.assertTrue((out/'recovery/metrics_before_test.jsonl').exists())
    def test_uncommitted_work_retained_without_fake_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'metrics.jsonl').write_text('{"global_step":1}\n')
            r=prepare_attempt(out,'fresh')
            self.assertEqual(r['global_step'],0);self.assertFalse(r['resume']);self.assertFalse((out/'metrics.jsonl').exists())
            self.assertTrue((out/'recovery/metrics_before_fresh.jsonl').exists())
    def test_policy_only_checkpoint_cannot_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);torch.save(dict(global_step=100,trained_state={}),out/'action_latest.pt')
            with self.assertRaises(ValueError):committed_step(out)
if __name__=='__main__':unittest.main()
