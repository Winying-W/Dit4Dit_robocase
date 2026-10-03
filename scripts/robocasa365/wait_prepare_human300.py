"""Prepare the full human300 manifest once all pinned task archives are extracted."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=json.loads(a.source.read_text());assert len(source['tasks'])==300
    a.output.parent.mkdir(parents=True,exist_ok=True)
    previous=None
    while True:
        ready=sum((a.root/'extracted'/Path(r['remote']).parent/'EXTRACTION_VERIFIED.json').exists() for r in source['tasks'])
        if ready!=previous:print('HUMAN300_READY',ready,'/300',flush=True);previous=ready
        if ready==300:break
        time.sleep(30)
    if (a.output/'manifest.json').exists():
        m=json.loads((a.output/'manifest.json').read_text());assert m['task_set']=='pretrain_human300' and len(m['tasks'])==300
        print('FULL_MANIFEST_ALREADY_PRESENT',flush=True);return
    subprocess.run([sys.executable,'-m','scripts.robocasa365.prepare_human300','--source',str(a.source),'--root',str(a.root),'--output',str(a.output)],check=True)
    print('FULL_HUMAN300_PREPARED',flush=True)


if __name__=='__main__':main()
