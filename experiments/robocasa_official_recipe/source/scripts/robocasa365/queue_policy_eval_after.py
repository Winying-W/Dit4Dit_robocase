"""Submit one evaluation job only after the authorized training job succeeds."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--training-task',required=True)
p.add_argument('--conf',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
os.environ['PATH']=str(Path.home()/'.volc/bin')+os.pathsep+os.environ['PATH']
with (a.output/'queue.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    destination=a.output/'submission.json'
    if destination.exists():
        print('ALREADY_SUBMITTED',destination.read_text(),flush=True);raise SystemExit(0)
    deadline=time.monotonic()+6*3600
    while time.monotonic()<deadline:
        response=subprocess.run(['volc','ml_task','get','--id',a.training_task,'--output','json','--format','Id,Status,Elapsed,ExitCode'],capture_output=True,text=True,timeout=40)
        if response.returncode:
            print('STATUS_QUERY_RETRY',response.stderr[-500:],flush=True);time.sleep(20);continue
        info=json.loads(response.stdout)[0]
        (a.output/'training_dependency.json').write_text(json.dumps(info,indent=2)+'\n')
        if info['Status']=='Success':break
        if info['Status'] in ('Failed','Canceled','Cancelled','Stopped'):
            print('TRAINING_DEPENDENCY_FAILED',json.dumps(info),flush=True);raise SystemExit(2)
        print('WAIT_TRAINING',json.dumps(info),flush=True);time.sleep(30)
    else:raise TimeoutError('Training dependency exceeded six hours')
    response=subprocess.run(['volc','ml_task','submit','--conf',str(a.conf),'--output','json'],capture_output=True,text=True,timeout=120)
    # Never retry an ambiguous submission automatically: prevent duplicate GPU jobs.
    (a.output/'submission_response.txt').write_text(response.stdout+'\n'+response.stderr)
    response.check_returncode()
    submitted=json.loads(response.stdout)
    assert submitted.get('Id'),submitted
    destination.write_text(json.dumps(submitted)+'\n')
    print('POLICY_EVAL_SUBMITTED',json.dumps(submitted),flush=True)
