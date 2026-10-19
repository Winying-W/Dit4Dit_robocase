"""Predeclared selection using paired development trials, never final benchmark seeds."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--train-root',type=Path,required=True);a=p.parse_args()
    reports={};checkpoints={};train_configs={}
    for arm in ['frozen','partial_joint']:
        folder=a.run/'ab'/arm/'eval'
        report=json.loads((folder/'evaluation.json').read_text())
        checkpoint=json.loads((folder/'checkpoint_verification.json').read_text())
        audit=json.loads((a.run/'ab'/arm/'action_audit.json').read_text())
        config=json.loads((a.train_root/arm/'run_config.json').read_text())
        assert report['status']=='complete' and report['completed_trials']==10
        assert report['initialization']=='fresh_gym_target' and report['horizon']==report['official_horizon']==2400
        assert [row['seed'] for row in report['episodes']]==list(range(100,110))
        assert audit['passed'] and audit['completed_trials']==10
        assert checkpoint['global_step']==1000 and config['accumulation']==4
        assert config['warm_start']['global_step']==2000 and config['warm_start']['optimizer_reset']
        reports[arm]=report;checkpoints[arm]=checkpoint;train_configs[arm]=config
    assert checkpoints['frozen']['normalization_sha256']==checkpoints['partial_joint']['normalization_sha256']
    for key in ['warm_start','split','accumulation','action_lr','seed']:
        assert train_configs['frozen'][key]==train_configs['partial_joint'][key], key
    for trial in range(10):
        folders=[a.run/'ab'/arm/'eval'/f'trial_{trial:03d}' for arm in ['frozen','partial_joint']]
        initial=[np.load(folder/'trajectory.npz')['initial_state'] for folder in folders]
        np.testing.assert_array_equal(*initial)
        assert (folders[0]/'initial_model.xml').read_bytes()==(folders[1]/'initial_model.xml').read_bytes()
    f,b=reports['frozen']['successes'],reports['partial_joint']['successes']
    fv,bv=checkpoints['frozen']['validation_action_fm_loss'],checkpoints['partial_joint']['validation_action_fm_loss']
    select_b=b>f or (b==f and bv < .95*fv)
    result=dict(status='complete',recipe='partial_joint' if select_b else 'frozen',
        development_successes=dict(frozen=f,partial_joint=b),trials_per_arm=10,
        validation_action_fm_loss=dict(frozen=fv,partial_joint=bv),
        rule='Prefer more successes on paired development seeds100..109. If tied, partial_joint must lower fixed held-out FM loss by over5%; otherwise retain frozen video.',
        caveat='Small development comparison selects a training recipe, not evidence of statistical superiority. Final 320 trials are independent and not used here.')
    (a.run/'ab_selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
