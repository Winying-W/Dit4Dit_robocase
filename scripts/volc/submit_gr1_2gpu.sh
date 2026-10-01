#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
"$ROOT/.conda/bin/python" - <<'PY'
import json
from pathlib import Path
p=Path('results/gr1_preflight_24/preflight_passed.json')
s=json.loads(p.read_text())
assert s['status']=='passed' and sorted(s['tasks'])==list(range(24)), 'All 24 scenes must pass locally'
s=json.loads(Path('results/gr1_smoke_bf16/summary.json').read_text())
assert s['complete'] and s['tasks']==2 and s['episodes']==2, 'Two complete real policy episodes required'
assert sorted(r['task_index'] for r in s['per_task'])==[2,23]
assert all(r['max_episode_steps']==720 and r['n_action_steps']==12 for r in s['per_task'])
s=json.loads(Path('checkpoints/dit4dit-model/dit4dit_robocasa_gr1/final_model/DOWNLOAD_VERIFIED.json').read_text())
assert s['sha256']=='fc65e78ab7c3de040d2df9f41416640e49befec58c784f550ecdaf41ee167098'
print('Local preflight gate passed: assets, all scenes, strict checkpoint load and closed-loop policy')
PY
if [[ -f results/gr1_setup_2gpu/submission.json ]]; then
    echo 'Submission already recorded; inspect results/gr1_setup_2gpu/submission.json before submitting again.' >&2
    exit 1
fi
volc ml_task submit --conf scripts/volc/robocasa_gr1_eval_2gpu.yaml --output json | tee results/gr1_setup_2gpu/submission.json
