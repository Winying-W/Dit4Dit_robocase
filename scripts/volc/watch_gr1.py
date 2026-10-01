"""Persist cloud job status and final metrics without resubmitting failed jobs."""
import datetime
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--setup-dir',type=Path,default=ROOT/'results/gr1_setup')
parser.add_argument('--run-dir',type=Path,default=ROOT/'results/gr1_volc_20260920')
args=parser.parse_args()
SETUP=args.setup_dir.resolve()
RUN=args.run_dir.resolve()
run_link=RUN.relative_to(ROOT.resolve()).as_posix() if RUN.is_relative_to(ROOT.resolve()) else str(args.run_dir)
setup_link=SETUP.relative_to(ROOT.resolve()).as_posix() if SETUP.is_relative_to(ROOT.resolve()) else str(args.setup_dir)
TASK=json.loads((SETUP/'submission.json').read_text())['Id']
TERMINAL={'Success','Failed','Killed'}

def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

last=None
while True:
    try:
        result=subprocess.run(['volc','ml_task','get','--id',TASK,'--output','json','--format','Id,Status,Elapsed,ExitCode'],capture_output=True,text=True,check=True,timeout=40)
        status=json.loads(result.stdout)[0]
        rows=[json.loads(p.read_text()) for p in sorted(RUN.glob('[0-9][0-9]/summary.json'))]
        n=sum(r['episodes'] for r in rows)
        report=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),task=status,completed_episodes=n,expected_episodes=1200,completed_tasks=sum(r['complete'] for r in rows))
        if n:
            report['partial_successes']=sum(r['successes'] for r in rows)
            report['partial_success_percent']=100*report['partial_successes']/n
        if (RUN/'summary.json').exists():
            final=json.loads((RUN/'summary.json').read_text())
            report['evaluation_complete']=bool(final['complete'] and final['episodes']==1200 and final['tasks']==24)
            report['result']=final
        atomic(SETUP/'cloud_status.json',report)
        key=(status['Status'],n)
        if key!=last:
            print(json.dumps({k:v for k,v in report.items() if k!='result'}),flush=True);last=key
        if status['Status'] in TERMINAL:
            logs=subprocess.run(['volc','ml_task','logs','--task',TASK,'--instance','worker_0','--lines','150'],capture_output=True,text=True,timeout=50)
            (SETUP/'cloud_final.log').write_text(logs.stdout+'\n'+logs.stderr)
            doc=ROOT/'ROBOCASA_RUN.md'
            text=doc.read_text()+'\n## 云端最终状态（监控自动记录）\n\n'
            text+=f"任务 `{TASK}` 状态：**{status['Status']}**。\n\n"
            if status['Status']=='Success' and report.get('evaluation_complete'):
                f=report['result']
                text+=f"正式评测 **{f['successes']}/{f['episodes']}，成功率 {f['success_percent']:.2f}%**；原作者统计口径 {f['author_success_percent']:.2f}%。\n\n"
                text+=f'[完整汇总]({run_link}/summary.json) · [逐任务表格]({run_link}/summary.md)\n'
            else:
                text+=f'正式完整成功率尚未取得；请查看 [最终日志]({setup_link}/cloud_final.log) 和 [状态记录]({setup_link}/cloud_status.json)。\n'
            doc.write_text(text)
            break
    except Exception as e:
        # A transient API failure must not overwrite the last successful status.
        print(f'Monitor retry: {type(e).__name__}: {e}',flush=True)
    time.sleep(60)
