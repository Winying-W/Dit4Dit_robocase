import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
p.add_argument('--expected-tasks',type=int,default=24)
p.add_argument('--expected-episodes',type=int,default=50)
a=p.parse_args()
rows=[json.loads(f.read_text()) for f in sorted(a.directory.glob('[0-9][0-9]/summary.json'))]
assert rows, 'No evaluation results'
assert not any(r['preflight'] for r in rows), 'Preflight is not a policy evaluation'
complete=(len(rows)==a.expected_tasks and all(r['complete'] and r['episodes']==a.expected_episodes for r in rows))
n=sum(r['episodes'] for r in rows); successes=sum(r['successes'] for r in rows)
report=dict(complete=complete, tasks=len(rows), episodes=n, successes=successes,
            success_rate=successes/n, success_percent=100*successes/n,
            author_success_percent=100*sum(r['author_successes'] for r in rows)/n,
            paper_reference_percent=50.8, released_checkpoint_reference_percent=56.7,
            per_task=[{k:v for k,v in r.items() if k!='results'} for r in rows])
(a.directory/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['| Task | Success | Rate |','|---|---:|---:|']
for r in rows: lines.append(f"| {r['task'].split('/')[1]} | {r['successes']}/{r['episodes']} | {100*r['success_rate']:.1f}% |")
lines.append(f'| **Total** | **{successes}/{n}** | **{100*successes/n:.2f}%** |')
(a.directory/'summary.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='per_task'},indent=2))
if not complete: raise SystemExit('Evaluation incomplete')
