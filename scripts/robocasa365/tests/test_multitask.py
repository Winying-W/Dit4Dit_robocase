import unittest
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from scripts.robocasa365.prepare_multitask import deterministic_split
from scripts.robocasa365.evaluate_multitask import wilson, summarize, benchmark_lock, policy_worker_environment, select_evaluation_tasks
from scripts.robocasa365.artifact_identity import policy_identity


class MultitaskProtocolTest(unittest.TestCase):
    def test_development_subset_keeps_full_manifest_unchanged_and_has_explicit_scope(self):
        import copy
        manifest=dict(tasks=[dict(task=name,horizon=10) for name in ['A','B','C']],task_scope='full_official_task_set')
        original=copy.deepcopy(manifest)
        rows,scope=select_evaluation_tasks(manifest,['C','A'],development=True,allow_subset=True)
        self.assertEqual([row['task'] for row in rows],['A','C'])
        self.assertEqual(scope,'declared_task_subset');self.assertEqual(manifest,original)
        report=summarize([dict(task=row['task'],complete=True,successes=0,completed_trials=20) for row in rows],
                         ['A','C'],list(range(100,120)),purpose='development',task_scope=scope)
        self.assertEqual(report['planned_trials'],40);self.assertEqual(report['task_scope'],'declared_task_subset')
        self.assertEqual(select_evaluation_tasks(manifest),(manifest['tasks'],'full_official_task_set'))

    def test_subset_cannot_silently_select_final_tasks_or_unknown_or_duplicate_names(self):
        manifest=dict(tasks=[dict(task='A'),dict(task='B')])
        for kwargs in [dict(requested=['A']),dict(requested=['A'],development=True),
                       dict(requested=['A'],allow_subset=True),
                       dict(requested=['X'],development=True,allow_subset=True),
                       dict(requested=['A','A'],development=True,allow_subset=True),
                       dict(requested=[],development=True,allow_subset=True)]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):select_evaluation_tasks(manifest,**kwargs)

    def test_independent_policy_workers_do_not_inherit_launch_ranks_or_ports(self):
        parent=dict(RANK='0',LOCAL_RANK='0',WORLD_SIZE='4',MASTER_ADDR='localhost',
            MASTER_PORT='1234',TORCHELASTIC_RUN_ID='run',CUDA_VISIBLE_DEVICES='0,1,2,3',
            MUJOCO_EGL_DEVICE_ID='0',PYTHONPATH='source',HF_HUB_OFFLINE='1')
        for device in map(str,range(4)):
            env=policy_worker_environment(device,parent)
            self.assertEqual(env,dict(CUDA_VISIBLE_DEVICES=device,MUJOCO_EGL_DEVICE_ID=device,
                PYTHONPATH='source',HF_HUB_OFFLINE='1'))
        self.assertEqual(parent['WORLD_SIZE'],'4')

    def test_same_paths_with_changed_weights_or_normalization_are_different_policies(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            incremental=root/'train'/'action.pt';base=root/'base'/'final_model'/'pytorch_model.pt'
            incremental.parent.mkdir();base.parent.mkdir(parents=True)
            files=[incremental,base,incremental.parent/'normalization.json',incremental.parent/'data_config.yaml',
                   base.parent.parent/'config.yaml',base.parent.parent/'dataset_statistics.json']
            for file in files:file.write_bytes(b'original')
            original=policy_identity(incremental,base)
            for file in files:
                file.write_bytes(b'replaced')  # Same size, same path: a path/size check is insufficient.
                self.assertNotEqual(policy_identity(incremental,base),original)
                file.write_bytes(b'original')
            self.assertEqual(policy_identity(incremental,base),original)

    def test_live_worker_keeps_lock_after_parent_context_closes(self):
        with tempfile.TemporaryDirectory() as folder:
            process=None
            try:
                with benchmark_lock(folder) as fd:
                    process=subprocess.Popen([sys.executable,'-c','import sys; sys.stdin.read()'],
                        stdin=subprocess.PIPE,pass_fds=(fd,))
                self.assertIsNone(process.poll())
                with self.assertRaises(RuntimeError):
                    with benchmark_lock(folder):pass
                process.communicate(b'',timeout=10)
                self.assertEqual(process.returncode,0)
                with benchmark_lock(folder):pass
            finally:
                if process is not None and process.poll() is None:
                    process.terminate();process.communicate(timeout=10)

    def test_split_is_disjoint_reproducible_and_order_independent(self):
        episodes=list(range(501))
        train,val=deterministic_split(episodes,'StirVegetables',20260924)
        self.assertEqual(len(train),451)
        self.assertEqual(len(val),50)
        self.assertFalse(set(train)&set(val))
        self.assertEqual(set(train)|set(val),set(episodes))
        self.assertEqual((train,val),deterministic_split(episodes[::-1],'StirVegetables',20260924))
        self.assertNotEqual((train,val),deterministic_split(episodes,'DeliverStraw',20260924))

    def test_incomplete_task_never_becomes_full_benchmark_rate(self):
        rows=[dict(task='A',complete=True,successes=10,completed_trials=20)]
        result=summarize(rows,['A','B'],list(range(20)))
        self.assertEqual(result['planned_trials'],40)
        self.assertEqual(result['completed_trials'],20)
        self.assertIsNone(result['pooled_success_rate'])
        rows.append(dict(task='B',complete=False,successes=1,completed_trials=19))
        self.assertIsNone(summarize(rows,['A','B'],list(range(20)))['equal_task_macro_success_rate'])
        rows[-1].update(complete=True,completed_trials=20)
        result=summarize(rows,['A','B'],list(range(20)))
        self.assertEqual(result['status'],'complete')
        self.assertAlmostEqual(result['pooled_success_rate'],11/40)
        self.assertAlmostEqual(result['equal_task_macro_success_rate'],11/40)

    def test_zero_success_still_has_positive_uncertainty_upper_bound(self):
        low,high=wilson(0,20)
        self.assertAlmostEqual(low,0)
        self.assertAlmostEqual(high,.16112515805281938)
        self.assertLess(wilson(0,320)[1],high)
        with self.assertRaises(ValueError):wilson(0,0)

    def test_final_100_trials_per_task_has_full_1600_denominator(self):
        names=[f'Task{i}' for i in range(16)]
        rows=[dict(task=name,complete=True,successes=50,completed_trials=100) for name in names]
        report=summarize(rows,names,list(range(1000,1100)))
        self.assertEqual(report['planned_trials'],1600)
        self.assertEqual(report['completed_trials'],1600)
        self.assertEqual(report['successes'],800)
        self.assertEqual(report['pooled_success_rate'],.5)
        self.assertEqual(report['purpose'],'final')

    def test_atomic_development_subset_is_explicitly_labelled(self):
        name='PickPlaceCounterToCabinet'
        rows=[dict(task=name,complete=True,successes=7,completed_trials=20)]
        report=summarize(rows,[name],list(range(100,120)),purpose='development',
            task_set='atomic_seen',task_scope='declared_task_subset')
        self.assertEqual(report['planned_trials'],20)
        self.assertEqual(report['task_set'],'atomic_seen')
        self.assertEqual(report['task_scope'],'declared_task_subset')
        self.assertEqual(report['purpose'],'development')
        self.assertIn('excluded from the final result',report['scope'])


if __name__=='__main__':unittest.main()
