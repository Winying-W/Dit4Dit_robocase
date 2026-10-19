"""Verify and summarize the bounded single-demo overfit cloud experiment."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from scripts.robocasa365.audit_saved_actions import audit_phase


def read(path):return json.loads(path.read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--train',type=Path,required=True)
    a=p.parse_args();root=a.run
    gt=read(root/'gt/result.json');assert gt['success'] and gt['episode']==5
    status=read(a.train/'status.json');assert status['status']=='complete' and status['global_steps']==2000
    split=read(a.train/'split.json');assert split['train_episodes']==[5] and split['validation_episodes']==[157,386,480,489]
    norm=read(a.train/'normalization.json')
    assert norm==read(Path('runs/robocasa365_dual_20260923/live/normalization.json'))
    tensors=read(a.train/'action_trainable.json');assert tensors and all(k.startswith('action_model.') for k in tensors)
    lineage=read(a.train/'warm_start.json');assert lineage['global_step']==2000 and lineage['optimizer_reset']
    baseline=read(root/'baseline/checkpoint_verification.json');final=read(root/'overfit_eval/checkpoint_verification.json')
    assert baseline['normalization_sha256']==final['normalization_sha256'] and final['warm_start']==lineage
    training=read(a.train/'summary.json')
    assert abs(training['phases'][-1]['final_validation']-final['validation_action_fm_loss'])<1e-6
    phases={}
    for label,count,horizon in [('baseline',1,8),('overfit_eval',3,8),('overfit_h1',1,1)]:
        folder=root/label
        if label=='overfit_h1' and not folder.exists():continue
        evaluation=read(folder/'evaluation.json');assert evaluation['status']=='complete' and len(evaluation['episodes'])==count
        assert evaluation['horizon']==2400 and evaluation['execution_horizon']==horizon
        assert all(x['demo_episode']==5 for x in evaluation['episodes'])
        inference=read(folder/'inference_summary.json');assert inference['status']=='complete'
        assert read(folder/'action_chain_audit.json')['passed']
        audit=audit_phase(folder,norm);assert audit['passed']
        (folder/'saved_prediction_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        for trial in evaluation['episodes']:
            assert trial['success'] or trial['steps']==2400
            video=folder/f"trial_{trial['trial']:03d}"/'rollout.mp4'
            cap=cv2.VideoCapture(str(video));count_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ok,frame=cap.read();assert ok and frame.shape==(256,768,3)
            assert count_frames==1+trial['steps']//4
            cap.set(cv2.CAP_PROP_POS_FRAMES,count_frames-1);ok,_=cap.read();cap.release();assert ok
        phases[label]=dict(trials=count,successes=evaluation['successes'],execution_horizon=horizon,
            scope='Training demo initial scene; NOT held-out success',audited_steps=audit['audited_steps'])
    for phase in ('baseline','overfit_eval'):
        trial=root/phase/'trial_000'
        with np.load(root/'gt/trajectory.npz') as gt_trace,np.load(trial/'trajectory.npz') as policy:
            np.testing.assert_array_equal(gt_trace['initial_state'],policy['initial_state'])
        assert (root/'gt/initial_model.xml').read_bytes()==(trial/'initial_model.xml').read_bytes()
    probes={label:read(root/label/'offline_prediction_metrics.json') for label in ('baseline','probe_1000','overfit_eval')}
    with np.load(root/'baseline/offline_predictions.npz') as before,np.load(root/'overfit_eval/offline_predictions.npz') as after:
        for name in ('target','valid','dataset_index','step','rng_seed','state'):np.testing.assert_array_equal(before[name],after[name])
    output=dict(status='complete',episode=5,additional_optimizer_steps=2000,warm_start=lineage,
        normalization_unchanged=True,trainable_tensor_count=len(tensors),independent_restore_loss=final['validation_action_fm_loss'],
        gt_replay=gt,policy_results=phases,offline_prediction_metrics=probes,
        limitations=['Single training demonstration; no generalization claim.','Original16-demo training normalization intentionally retained.','Scaled command errors do not measure achieved end-effector pose.'])
    (root/'report.json').write_text(json.dumps(output,indent=2)+'\n')
    rows=['# Single-demo overfit result','',f"Episode5; {lineage['global_step']} warm-start steps + 2000 additional updates. Original normalization unchanged.",'',
          '| Phase | Train-scene successes | Execute horizon |','| --- | --- | --- |']
    for name,record in phases.items():rows.append(f"| {name} | {record['successes']}/{record['trials']} | {record['execution_horizon']} |")
    rows+=['','These are training-scene diagnostic results, not held-out success rates.','',
           '| Probe | Arm command MAE | Position component MAE (m) | Rotation component MAE (rad) | Gripper accuracy |',
           '| --- | --- | --- | --- | --- |']
    for name,probe in probes.items():
        m=probe['overall'];rows.append(f"| {name} | {m['arm_command']['mae']:.6f} | {m['position_scaled_component_m']['mae']:.6f} | {m['rotation_scaled_component_rad']['mae']:.6f} | {m['gripper_accuracy']:.4f} |")
    rows+=['','Full metrics, per-chunk-step errors, action audits and limitations: [report.json](report.json).']
    (root/'report.md').write_text('\n'.join(rows)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('gt_replay','offline_prediction_metrics')},indent=2))

if __name__=='__main__':main()
