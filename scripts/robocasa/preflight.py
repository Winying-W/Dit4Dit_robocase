"""Import once, then reset/render/step every requested task using the real simulator."""
import argparse
import json
from pathlib import Path
import time
from scripts.robocasa.evaluate import main as evaluate
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--tasks',nargs='+',type=int,default=list(range(24)))
a=p.parse_args();started=time.monotonic()
passed=[]
for task in a.tasks:
    evaluate(['--task-index',str(task),'--episodes','1','--preflight','--videos','0','--output',str(a.output)])
    passed.append(task)
    print(f'PREFLIGHT PASS: {task} ({len(passed)}/{len(a.tasks)})',flush=True)
(a.output/'preflight_passed.json').write_text(json.dumps(dict(status='passed',tasks=passed,seconds=time.monotonic()-started),indent=2)+'\n')
