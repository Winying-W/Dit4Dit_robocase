"""Evidence-based summary of completed training, replay diagnosis and policy evaluation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import cv2
from scripts.robocasa365.audit_saved_actions import audit_phase


def read(path):return json.loads(path.read_text())

def wilson(successes,n,z=1.959963984540054):
    fraction=successes/n;denom=1+z*z/n
    mid=(fraction+z*z/(2*n))/denom
    half=z*math.sqrt(fraction*(1-fraction)/n+z*z/(4*n*n))/denom
    return [max(0.,mid-half),min(1.,mid+half)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--train-run',type=Path,required=True);p.add_argument('--eval-run',type=Path,required=True)
    p.add_argument('--initial-training-mirror',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    training=read(a.train_run/'live/status.json')
    assert training['status']=='complete' and training['global_steps']==2000,training
    latest=read(a.train_run/'live/latest.json');assert latest['global_step']==2000 and latest['checkpoint_readback_passed']
    split=read(a.train_run/'live/split.json');norm=read(a.train_run/'live/normalization.json')
    assert split==read(a.initial_training_mirror/'split.json')
    assert norm==read(a.initial_training_mirror/'normalization.json')
    trainable=read(a.train_run/'live/action_trainable.json');assert all(k.startswith('action_model.') for k in trainable)
    phases=[];validation=[]
    for line in (a.train_run/'cloud.log').read_text().splitlines():
        if not line.startswith('{'):continue
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        if row.get('event')=='validation':phases.append(row)
        if 'validation_action_fm_loss' in row:validation.append(dict(step=row['global_step'],loss=row['validation_action_fm_loss']))
    restored=[x for x in phases if x.get('phase_step')==2000]
    assert restored and validation[-1]['step']==2000
    assert abs(restored[-1]['action_fm_loss']-validation[-1]['loss'])<1e-6
    policies={};evaluations={}
    for label,count in [('step2000_target',10),('step300_target',3),('step2000_heldout_nonmobile',4)]:
        directory=a.eval_run/label;evaluation=read(directory/'evaluation.json');evaluations[label]=evaluation
        assert evaluation['status']=='complete' and len(evaluation['episodes'])==count
        assert evaluation['completed_trials']==count
        assert evaluation['successes']==sum(bool(x['success']) for x in evaluation['episodes'])
        assert evaluation['success_rate']==evaluation['successes']/count
        assert [x['trial'] for x in evaluation['episodes']]==list(range(count))
        assert [x['seed'] for x in evaluation['episodes']]==list(range(100,100+count))
        assert evaluation['horizon']==evaluation['official_horizon']==2400
        assert evaluation['initialization']==('demo_initial_state' if label=='step2000_heldout_nonmobile' else 'fresh_gym_target')
        if label=='step2000_heldout_nonmobile':
            assert [x['demo_episode'] for x in evaluation['episodes']]==split['validation_episodes']
        inference=read(directory/'inference_summary.json');assert inference['status']=='complete'
        assert inference['global_step']==(300 if label=='step300_target' else 2000)
        assert inference['normalization_sha256']==hashlib.sha256((a.train_run/'live/normalization.json').read_bytes()).hexdigest()
        assert read(directory/'action_chain_audit.json')['passed']
        prediction_audit=audit_phase(directory,norm)
        assert prediction_audit['passed'] and prediction_audit['completed_trials']==count
        (directory/'saved_prediction_audit.json').write_text(json.dumps(prediction_audit,indent=2)+'\n')
        for trial in evaluation['episodes']:
            assert trial['action_chain_runtime_passed'] and trial['controller_input_type']=='delta'
            if not trial['success']:assert trial['steps']==2400,trial
            folder=directory/f"trial_{trial['trial']:03d}"
            trace=np.load(folder/'trajectory.npz')
            sent=trace['actions_dataset_order'];received=trace['actions_actually_received_by_simulator']
            assert sent.shape==received.shape==(trial['steps'],12)
            expected=sent.copy();expected[:,4]=np.where(expected[:,4]<.5,-1.,1.);expected[:,11]=np.where(expected[:,11]<.5,-1.,1.)
            np.testing.assert_allclose(received,expected[:,[5,6,7,8,9,10,11,0,1,2,3,4]],atol=1e-7,rtol=0)
            assert np.isfinite(sent).all() and np.all(sent[:,:4]==0)
            assert np.all(np.abs(sent)<=1)
            assert trace['normalized_chunks'].shape==((trial['steps']+7)//8,16,32)
            assert trace['query_states'].shape==((trial['steps']+7)//8,16)
            assert np.isfinite(trace['normalized_chunks']).all() and np.isfinite(trace['query_states']).all()
            cap=cv2.VideoCapture(str(folder/'rollout.mp4'))
            frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));ok,frame=cap.read()
            assert ok and frame.shape==(256,768,3) and frames==1+trial['steps']//4
            cap.set(cv2.CAP_PROP_POS_FRAMES,frames-1);ok,last_frame=cap.read();cap.release()
            assert ok and last_frame.shape==frame.shape
        policies[label]=dict(initialization=evaluation['initialization'],trials=count,successes=evaluation['successes'],success_rate=evaluation['success_rate'],
            wilson_95_interval=wilson(evaluation['successes'],count),checkpoint=inference['checkpoint'],
            normalized_action_roundtrip_max_error=read(directory/'action_chain_audit.json')['max_roundtrip_error'],
            full_runtime_action_chain_passed=True,
            independently_audited_simulator_steps=prediction_audit['audited_steps'],
            prediction_to_simulator_max_error=max(x['prediction_to_simulator_max_error'] for x in prediction_audit['trials']))
    pairs=[]
    for i in range(3):
        early=a.eval_run/'step300_target'/f'trial_{i:03d}';late=a.eval_run/'step2000_target'/f'trial_{i:03d}'
        first=np.load(early/'trajectory.npz')['initial_state'];second=np.load(late/'trajectory.npz')['initial_state']
        exact=bool(np.array_equal(first,second))
        same_xml=(early/'initial_model.xml').read_bytes()==(late/'initial_model.xml').read_bytes()
        pairs.append(dict(seed=read(early/'result.json')['seed'],initial_state_exact=exact,model_xml_exact=same_xml,
            step300_success=read(early/'result.json')['success'],step2000_success=read(late/'result.json')['success']))
    replay=read(a.train_run/'replay/report.json');assert {x['episode'] for x in replay}=={0,3}
    output=dict(status='complete',training_global_steps=2000,train_split_unchanged=True,normalization_unchanged=True,
        frozen_backbone=True,trainable_tensor_count=len(trainable),trainable_parameter_count=sum(math.prod(shape) for shape in trainable.values()),independent_resume_validation=restored[-1],validation_trace=validation,
        policy_results=policies,paired_initializations=pairs,replay=replay,
        limitations=['Single task; only16 training demos and8 fixed validation windows.','GT action replay drift remains; failure mechanisms identified, not eliminated.',
            'Exploratory target success rates are small-sample results; no365-task benchmark claim.',
            'Fresh target scenes are not filtered for non-mobile feasibility; four held-out non-mobile demo initializations are reported separately.'])
    (a.output/'report.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('replay','validation_trace')},indent=2))

if __name__=='__main__':main()
