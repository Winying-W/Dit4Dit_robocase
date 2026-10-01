#!/usr/bin/env bash
set -Eeuo pipefail
umask 000
if [ ! -e /file_system ]; then ln -s / /file_system; fi
ROOT=/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT
cd "$ROOT"
source scripts/robocasa365/env.sh
export NO_ALBUMENTATIONS_UPDATE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN365_EVAL="$ROOT/runs/robocasa365_policy_eval_20260923"
CHECKPOINT365="$ROOT/results/robocasa365_dual_20260923/train/action_step_002000.pt"
mkdir -p "$RUN365_EVAL"
chmod 777 "$RUN365_EVAL"
exec > >(tee -a "$RUN365_EVAL/cloud.log") 2>&1
trap 'rc=$?; echo "POLICY_EVAL_JOB_EXIT code=$rc"; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv
"$MODEL365_PYTHON" - <<'AUDIT'
import hashlib,json
from pathlib import Path
files=['scripts/robocasa365/evaluate_policy.py','scripts/robocasa365/eval_simulator.py','scripts/robocasa365/eval_protocol.py','scripts/robocasa365/action_audit.py','DiT4DiT/dataloader/robocasa365_datasets.py','DiT4DiT/model/framework/DiT4DiT.py']
records={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
Path('runs/robocasa365_policy_eval_20260923/source_hashes.json').write_text(json.dumps(records,indent=2)+'\n')
AUDIT
# Separate, labelled 32-step integration gate before spending time on full episodes.
"$MODEL365_PYTHON" -u scripts/robocasa365/evaluate_policy.py \
  --checkpoint "$CHECKPOINT365" --dataset "$DATASET365" --sim-python "$SIM365_PYTHON" \
  --output "$RUN365_EVAL/smoke" --seeds 100 --max-steps 32 --execute-horizon 8
"$MODEL365_PYTHON" -u scripts/robocasa365/evaluate_policy.py \
  --checkpoint "$CHECKPOINT365" --dataset "$DATASET365" --sim-python "$SIM365_PYTHON" \
  --output "$RUN365_EVAL/step2000_target" --seeds 100 101 102 103 104 105 106 107 108 109 --execute-horizon 8
# Identical first three seeds for a small step300-vs2000 paired comparison.
"$MODEL365_PYTHON" -u scripts/robocasa365/evaluate_policy.py \
  --checkpoint "$ROOT/results/robocasa365_train_20260922/train/action_latest.pt" \
  --dataset "$DATASET365" --sim-python "$SIM365_PYTHON" \
  --output "$RUN365_EVAL/step300_target" --seeds 100 101 102 --execute-horizon 8
# The 4 held-out non-mobile demonstrations define a second, explicitly diagnostic
# initialization set. Only initial XML/state is restored; no GT actions are used.
"$MODEL365_PYTHON" -u scripts/robocasa365/evaluate_policy.py \
  --checkpoint "$CHECKPOINT365" --dataset "$DATASET365" --sim-python "$SIM365_PYTHON" \
  --output "$RUN365_EVAL/step2000_heldout_nonmobile" --seeds 100 101 102 103 \
  --demo-episodes 157 386 480 489 --execute-horizon 8
