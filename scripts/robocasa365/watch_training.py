"""Read-only cloud monitor; never resubmits or cancels jobs."""
import json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
setup=root/'runs/robocasa365_train_setup'
task=json.loads((setup/'submission.json').read_text())['Id']
env=os.environ.copy();env['PATH']=str(Path.home()/'.volc/bin')+':'+env['PATH']
last=None
while True:
    try:
        cmd=['volc','ml_task','get','--id',task,'--output','json','--format','Id,Status,Elapsed,ExitCode']
        proc=subprocess.run(cmd,env=env,text=True,capture_output=True,check=True,timeout=30)
        status=json.loads(proc.stdout);(setup/'cloud_status.json').write_text(json.dumps(status,indent=2)+'\n')
        live=setup/'live/status.json';report=json.loads(live.read_text()) if live.exists() else {}
        key=(status[0]['Status'],report.get('phase'),report.get('global_step',report.get('global_steps')))
        if key!=last:
            print(json.dumps({'cloud':status[0],'training':report}),flush=True)
            with (setup/'progress.jsonl').open('a') as f:f.write(json.dumps({'cloud':status[0],'training':report})+'\n')
            last=key
        if status[0]['Status'] in ('Success','Failed','Killed'):
            p=subprocess.run(['volc','ml_task','logs','--task',task,'--instance','worker_0','--lines','500'],env=env,text=True,capture_output=True,timeout=45)
            (setup/'cloud_final.log').write_text(p.stdout+'\n'+p.stderr)
            break
    except Exception as exc:print(f'Monitor read retry: {type(exc).__name__}: {exc}',flush=True)
    time.sleep(30)
