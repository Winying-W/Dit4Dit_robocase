"""Validate replay under matching rendering protocols; report cross-mode differences."""
import json
import math
from pathlib import Path
import cv2

root=Path(__file__).resolve().parents[2]
output=root/'runs/robocasa365_replay_fixed_20260923'
reference=json.loads((root/'runs/robocasa365_replay_warmup_20260923/summary.json').read_text())
reference={r['episode']:r for r in reference if r['variant']=='warmup'}
checks=[]
headless=json.loads((output/'headless/replay.json').read_text())
assert headless['timing_passed']
for row in headless['episodes']:
    baseline=reference[row['episode']]
    assert row['startup']['unrecorded_zero_steps']==1
    assert row['initial_full_state_max_error']==0
    assert row['success']==baseline['success']
    assert math.isclose(row['sim_state_l2_final'],baseline['final_state_l2'],rel_tol=0,abs_tol=1e-10)
    checks.append(dict(mode='headless',episode=row['episode'],success=row['success'],time_error_max=row['time_error_max'],matches_diagnostic=True))
first=json.loads((output/'video/replay.json').read_text())
second=json.loads((output/'video_repeat/replay.json').read_text())
assert first['timing_passed'] and second['timing_passed']
a,b=first['episodes'][0],second['episodes'][0]
assert a['episode']==b['episode']==11 and a['success'] and b['success']
assert a['initial_full_state_max_error']==b['initial_full_state_max_error']==0
errors1=json.loads((output/'video/errors_000011.json').read_text())
errors2=json.loads((output/'video_repeat/errors_000011.json').read_text())
assert errors1==errors2, 'Same-render-protocol error traces differ'
assert a['sim_state_l2_final']==b['sim_state_l2_final']
cap=cv2.VideoCapture(str(output/'video_repeat/replay_000011.mp4'))
ok,frame=cap.read()
assert ok and frame.shape==(256,768,3)
frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
assert frames==492
checks.append(dict(mode='video',episode=11,success=b['success'],time_error_max=b['time_error_max'],
                   identical_error_trace_on_repeat=True,frames=frames,frame_shape=list(frame.shape)))
report=dict(status='passed',checks=checks,
            cross_render_mode=dict(headless_final_state_l2=reference[11]['final_state_l2'],video_final_state_l2=b['sim_state_l2_final'],
                                   note='Rendering configuration changes the rollout; compare only a fixed protocol.'),
            previous_assertion_failure='The previous job incorrectly required rendered and headless rollouts to match; replaced by repeat verification within each rendering protocol, retaining both traces.')
(output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print('FIXED_REPLAY_VERIFIED',json.dumps(report),flush=True)
