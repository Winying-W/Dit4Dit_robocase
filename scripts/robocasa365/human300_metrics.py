"""Masked sampled-action metrics in RoboCasa365 dataset order."""
import numpy as np
from scripts.robocasa365.prediction_metrics import action_metrics


def sampled_action_metrics(prediction,target,valid):
    prediction=np.asarray(prediction);target=np.asarray(target);valid=np.asarray(valid,dtype=bool)
    if prediction.shape!=target.shape or prediction.shape!=valid.shape or prediction.shape[-1]!=12:
        raise ValueError('Expected matching [...,16,12] arrays')
    def summarize(p,t,v):
        if not v.any():return dict(valid_elements=0,base_command_mae=None,mode_accuracy_at_0_5=None)
        report=action_metrics(p,t,v)
        mask=v[...,:4];errors=np.abs(np.clip(p[...,:4],-1,1)-t[...,:4])
        report['base_command_mae']=float(errors[mask].mean()) if mask.any() else None
        modes=v[...,4];report['mode_accuracy_at_0_5']=float(((p[...,4]>=.5)==(t[...,4]>=.5))[modes].mean()) if modes.any() else None
        return report
    return dict(scope='Sampled actions on fixed held-out episode windows; not task success or achieved pose error',
        overall=summarize(prediction,target,valid),by_chunk_step={str(i+1):summarize(prediction[:,i:i+1],target[:,i:i+1],valid[:,i:i+1]) for i in (0,7,15)})
