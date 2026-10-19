"""Exhaustively check real train/validation action rows through the actual transform."""
import json
import numpy as np
import pandas as pd


def audit_dataset_actions(ds,dataset,split,require_fixed_base=True):
    modality=json.loads((dataset/'meta/modality.json').read_text())['action']
    meta=json.loads((dataset/'extras/dataset_meta.json').read_text())['env_args']
    arm=meta['env_kwargs']['controller_configs']['body_parts']['right']
    assert arm['type']=='OSC_POSE' and arm['input_type']=='delta' and arm['input_ref_frame']=='base'
    np.testing.assert_allclose(arm['output_max'],[.05,.05,.05,.5,.5,.5],rtol=0,atol=0)
    expected_ranges=dict(base_motion=(0,4),control_mode=(4,5),end_effector_position=(5,8),end_effector_rotation=(8,11),gripper_close=(11,12))
    for key,(start,end) in expected_ranges.items():
        assert (modality[key]['start'],modality[key]['end'])==(start,end)
    ep,t=ds.dataset.all_steps[0];template=ds.dataset.get_step_data(ep,t)
    reports=[];max_error=0.
    for ep in split['train_episodes']+split['validation_episodes']:
        frame=pd.read_parquet(next((dataset/'data').glob(f'*/episode_{ep:06d}.parquet')),columns=['action'])
        raw=np.stack(frame['action'])
        assert raw.shape[1]==12 and np.isfinite(raw).all() and np.max(np.abs(raw))<=1+1e-6
        assert set(np.unique(raw[:,4])).issubset({-1.,1.})
        if require_fixed_base:
            assert np.all(raw[:,:4]==0) and np.all(raw[:,4]==-1)
        assert set(np.unique(raw[:,11])).issubset({-1.,1.})
        data={key:raw[:,expected_ranges[key.removeprefix('action.')][0]:expected_ranges[key.removeprefix('action.')][1]].astype(np.asarray(template[key]).dtype) for key in ds.robot.action_keys}
        transformed=ds.dataset.transforms(data)
        normalized=np.concatenate([np.asarray(transformed[key]) for key in ds.robot.action_keys],axis=-1).astype(np.float32)
        padded=np.pad(normalized,((0,0),(0,20)))
        restored=ds.decode_actions(padded,env_order=False)
        error=float(np.max(np.abs(restored-raw)));max_error=max(max_error,error)
        np.testing.assert_allclose(restored,raw,rtol=0,atol=1e-6)
        np.testing.assert_allclose(ds.decode_actions(padded),raw[:,[5,6,7,8,9,10,11,0,1,2,3,4]],rtol=0,atol=1e-6)
        np.testing.assert_array_equal(normalized[:,[4,11]],raw[:,[4,11]])
        if require_fixed_base:
            np.testing.assert_array_equal(normalized[:,:4],0.)
        reports.append(dict(episode=ep,frames=len(raw),max_roundtrip_error=error,
            nonzero_base_frames=int(np.any(raw[:,:4]!=0,axis=1).sum()),base_mode_frames=int((raw[:,4]==1).sum()),
            normalized_components_outside_unit_range=int(np.sum(np.abs(normalized)>1+1e-6))))
    return dict(passed=True,require_fixed_base=require_fixed_base,episodes=reports,frames=sum(x['frames'] for x in reports),max_roundtrip_error=max_error,
        dataset_order=list(expected_ranges),input_type='delta',input_reference_frame='base',rotation='axis-angle delta; unchanged representation',
        continuous_normalization='training min/max across all continuous dimensions; constant dimensions decode to their training constant',
        binary_normalization='mode and gripper preserve raw -1/+1; official Gym thresholds predictions at 0.5',
        physical_output_scale=[.05,.05,.05,.5,.5,.5],physical_scale_applied='inside OSC only',
        note='Held-out actions can exceed fitted normalized [-1,1]; min-max transform deliberately does not clip them.')
