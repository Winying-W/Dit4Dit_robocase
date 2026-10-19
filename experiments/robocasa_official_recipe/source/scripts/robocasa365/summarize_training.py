"""Summarize captured cloud logs and mirrored training artifacts without GPU access."""
import csv,json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
setup=root/'runs/robocasa365_train_setup';live=setup/'live'
rows={};validations=[];events=[]
for line in (setup/'cloud_final.log').read_text().splitlines():
    try:row=json.loads(line)
    except (ValueError,TypeError):continue
    if not isinstance(row,dict):continue
    if 'global_step' in row and 'action_loss' in row:rows[row['global_step']]=row
    if row.get('event')=='validation':validations.append(row)
    if row.get('event'):events.append(row)
fields=['global_step','phase','phase_step','action_loss','future_video_loss','validation_action_fm_loss','gradient_norm','seconds','peak_allocated_gib']
with (setup/'metrics_logged.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows[k] for k in sorted(rows))
summary=json.loads((live/'summary.json').read_text()) if (live/'summary.json').exists() else {'status':'incomplete'}
summary['resume_verification_elapsed_seconds']=summary.pop('elapsed_seconds',None)
summary['resume_verification_phases']=summary.pop('phases',[])
summary.update(cloud=json.loads((setup/'cloud_status.json').read_text()),validation_baselines=validations,
    logged_optimizer_steps=len(rows),checkpoints=[x for x in events if x.get('event')=='checkpoint'],
    peak_allocated_gib=max((r['peak_allocated_gib'] for r in rows.values()),default=None),
    inference_limitation='Offline FM metrics only; no closed-loop success claim')
recovery=setup/'recovery_status.json'
if recovery.exists():summary['recovery_cloud']=json.loads(recovery.read_text())
summary['initial_validation_action_fm_loss']=next((r['action_fm_loss'] for r in validations if r.get('phase')=='action' and r.get('phase_step')==0),None)
summary['final_phase_validation']={r['phase']:r['validation_action_fm_loss'] for _,r in sorted(rows.items()) if 'validation_action_fm_loss' in r}
summary['validation_windows']=8
summary['original_task_failure']='After completed optimizer step400 and checkpoint readback, final summary hit NameError(start); fixed to start_time and resumed for finalization without extra updates.'
(setup/'training_report.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
