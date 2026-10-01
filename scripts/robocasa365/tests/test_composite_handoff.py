"""Submission-state adversaries with synthetic reports; no real cloud mutations."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import yaml

from scripts.robocasa365.queue_composite_after_atomic import Cloud, advance, candidate, sha


def put(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


class FakeCloud:
    def __init__(self):
        self.status='Running'
        self.active=[]
        self.matches=[]
        self.calls=[]
        self.submission_error=None
        self.observation_error=None

    def get(self, task_id):
        if self.observation_error:
            raise self.observation_error
        return dict(Id=task_id,Name='dit4dit-target' if task_id=='new' else 'dit4dit-source',
                    Status='Queue' if task_id=='new' else self.status,ExitCode=0)

    def find(self, name, active_only=False):
        return self.active if active_only else self.matches

    def submit(self, config):
        self.calls.append(config)
        if self.submission_error:
            raise self.submission_error
        return dict(Id='new')


class CloudOutputTest(unittest.TestCase):
    def call(self, output, operation='list', returncode=0):
        result = subprocess.CompletedProcess([], returncode, output, '')
        with patch('scripts.robocasa365.queue_composite_after_atomic.subprocess.run', return_value=result):
            return Cloud().call([operation, '--output', 'json'])

    def test_empty_list_notice(self):
        self.assertEqual(self.call('没有匹配条件的任务\n\n[]\n'), [])

    def test_plain_json(self):
        self.assertEqual(self.call('[{"Id":"existing"}]\n'), [{'Id': 'existing'}])

    def test_notice_does_not_mask_unknown_output(self):
        for output, operation in [('没有匹配条件的任务\n\n[]\n', 'submit'),
                                  ('没有匹配条件的任务\n\n[{"Id":"existing"}]', 'list'),
                                  ('unexpected warning\n[]', 'list')]:
            with self.subTest(output=output, operation=operation):
                with self.assertRaises(json.JSONDecodeError):
                    self.call(output, operation)

    def test_nonzero_cli_exit_still_fails(self):
        with self.assertRaises(RuntimeError):
            self.call('没有匹配条件的任务\n\n[]\n', returncode=1)


class CompositeHandoffTest(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        root=Path(tmp.name);self.source=root/'atomic';self.target=root/'composite'
        self.observer=root/'observer';self.output=root/'handoff'
        self.output.mkdir();self.target.mkdir()
        self.cloud=FakeCloud()
        weights=root/'atomic.pt';weights.write_bytes(b'synthetic test checkpoint')
        entry='scripts/entry.sh';p=self.target/'source'/entry;p.parent.mkdir(parents=True);p.write_text('exit 0\n')
        scene=dict(name='stable_counter_v1')
        put(self.target/'source_hashes.json',{entry:sha(p)})
        put(self.target/'plan.json',dict(task_set='composite_seen',task_count=16,planned_updates=50000,
            planned_final_trials=1600,gpu_count=4,entrypoint_script=entry,scene_protocol=scene,
            warm_start=dict(checkpoint=str(weights),checkpoint_sha256=sha(weights),source_global_step=50000,local_initial_step=0)))
        config=dict(TaskName='dit4dit-target',Entrypoint=f'bash {self.target}/source/{entry}',
                    TaskRoleSpecs=[dict(RoleReplicas=1,ResourceSpec=dict(GPUNum=4))],Storages=[])
        (self.target/'task.yaml').write_text(yaml.safe_dump(config))
        put(self.target/'prequeue_acceptance.json',dict(status='complete',checks=dict(warm_start=dict(passed=True)),
            scene_protocol=scene,source_hashes_sha256=sha(self.target/'source_hashes.json'),
            evidence_sha256={n:sha(self.target/n) for n in ['plan.json','task.yaml','source_hashes.json']}))
        self.accepted=candidate(self.target)
        put(self.source/'plan.json',dict(task_set='atomic_seen',task_count=18,final_checkpoint=50000,
                                         planned_final_trials=1800,development_checkpoints=[10000,25000]))
        tasks=[dict(task=f'Task{i}') for i in range(18)]
        put(self.source/'prepared/manifest.json',dict(tasks=tasks))
        (self.source/'stage.txt').write_text('complete\n')
        for name,step,purpose,count in [('development_10000',10000,'development',360),
                ('development_25000',25000,'development',360),('evaluation_1800',50000,'final',1800)]:
            report=self.source/name/'report.json';put(report,dict(status='complete',completed_trials=count,successes=0))
            identity=self.source/name/'identity.json';put(identity,dict(checkpoint_step=step))
            review=dict(status='complete',report_sha256=sha(report),manifest_sha256=sha(self.source/'prepared/manifest.json'),
                identity_sha256=sha(identity),checkpoint_step=step,purpose=purpose,task_set='atomic_seen',
                saved_action_chain_passed=True,planned_trials=count,completed_trials=count,successes=0,
                tasks=[dict(**t,complete=True,action_chain_passed=True,completed_trials=count//18) for t in tasks])
            put(self.observer/name/f'review_{sha(report)}.json',review)

    def call(self,submit=True):
        return advance(self.cloud,dependency_task='old',dependency_run=self.source,observer=self.observer,
                       target=self.target,output=self.output,accepted=self.accepted,submit=submit)

    def test_running_dependency_never_submits_even_with_complete_local_files(self):
        self.assertEqual(self.call()['status'],'waiting_for_dependency')
        self.assertEqual(self.cloud.calls,[])
        self.assertFalse((self.target/'submission_intent.json').exists())

    def test_terminal_failure_is_not_resource_ready_success(self):
        for status in ['Failed','Killed']:
            self.cloud.status=status
            self.assertEqual(self.call()['status'],'dependency_failed')
        self.assertEqual(self.cloud.calls,[])

    def test_observation_timeout_cannot_trigger_submission(self):
        self.cloud.observation_error=subprocess.TimeoutExpired('volc',30)
        with self.assertRaises(subprocess.TimeoutExpired):self.call()
        self.assertEqual(self.cloud.calls,[])

    def test_complete_cloud_without_current_review_waits(self):
        self.cloud.status='Success'
        report=self.source/'evaluation_1800/report.json'
        put(report,dict(status='complete',completed_trials=1800,successes=1))
        self.assertEqual(self.call()['status'],'waiting_for_complete_reviews')
        self.assertEqual(self.cloud.calls,[])

    def test_wrong_scene_identity_in_review_refuses(self):
        self.cloud.status='Success'
        put(self.source/'evaluation_1800/identity.json',dict(changed=True))
        with self.assertRaisesRegex(ValueError,'mismatched independent audit'):self.call()
        self.assertEqual(self.cloud.calls,[])

    def test_another_project_allocation_waits(self):
        self.cloud.status='Success';self.cloud.active=[dict(Id='another',Status='Queue')]
        self.assertEqual(self.call()['status'],'waiting_for_project_capacity')
        self.assertEqual(self.cloud.calls,[])

    def test_existing_exact_target_is_never_duplicated(self):
        self.cloud.status='Success';self.cloud.matches=[dict(Id='existing',Name='dit4dit-target')]
        self.assertEqual(self.call()['status'],'existing_target_requires_reconciliation')
        self.assertEqual(self.cloud.calls,[])

    def test_dry_ready_does_not_submit_or_create_intent(self):
        self.cloud.status='Success'
        self.assertEqual(self.call(submit=False)['status'],'ready')
        self.assertEqual(self.cloud.calls,[])
        self.assertFalse((self.target/'submission_intent.json').exists())

    def test_valid_handoff_submits_only_once_across_calls(self):
        self.cloud.status='Success'
        self.assertEqual(self.call()['status'],'submitted')
        self.assertEqual(self.call()['status'],'already_submitted')
        self.assertEqual(len(self.cloud.calls),1)
        self.assertEqual(json.loads((self.target/'submission.json').read_text())['Id'],'new')

    def test_uncertain_submit_persists_intent_and_never_retries(self):
        self.cloud.status='Success';self.cloud.submission_error=subprocess.TimeoutExpired('submit',120)
        self.assertEqual(self.call()['status'],'submission_requires_reconciliation')
        self.assertEqual(self.call()['status'],'submission_requires_reconciliation')
        self.assertEqual(len(self.cloud.calls),1)
        self.assertTrue((self.target/'submission_intent.json').exists())

    def test_changed_source_or_weights_refuses(self):
        self.cloud.status='Success'
        original=(self.target/'source/scripts/entry.sh').read_text()
        (self.target/'source/scripts/entry.sh').write_text('exit 1\n')
        with self.assertRaisesRegex(ValueError,'Frozen source changed'):self.call()
        (self.target/'source/scripts/entry.sh').write_text(original)
        weights=Path(json.loads((self.target/'plan.json').read_text())['warm_start']['checkpoint'])
        weights.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checkpoint changed'):self.call()
        self.assertEqual(self.cloud.calls,[])


if __name__=='__main__':unittest.main()
