#!/usr/bin/env bash
set -Eeuo pipefail
umask 000
if [ ! -e /file_system ]; then ln -s / /file_system; fi
ROOT=/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT
cd "$ROOT"
source scripts/robocasa365/env.sh
export NO_ALBUMENTATIONS_UPDATE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OVERFIT365_RUN="$ROOT/runs/robocasa365_overfit_ep5_20260923"
OVERFIT365_TRAIN="$ROOT/results/robocasa365_overfit_ep5_20260923/train"
OVERFIT365_BASE="$ROOT/results/robocasa365_dual_20260923/train/action_step_002000.pt"
OVERFIT365_GR1="$ROOT/checkpoints/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt"
mkdir -p "$OVERFIT365_RUN/live" "$OVERFIT365_TRAIN"
chmod 777 "$OVERFIT365_RUN" "$OVERFIT365_TRAIN"
exec 9>"$OVERFIT365_RUN/cloud.lock"
flock -n 9 || { echo 'An overfit task already holds the lock'; exit 2; }
if [ -e "$OVERFIT365_RUN/started" ]; then echo 'Existing run; explicit recovery required'; exit 2; fi
touch "$OVERFIT365_RUN/started"
exec > >(tee -a "$OVERFIT365_RUN/cloud.log") 2>&1
trap 'rc=$?; echo "OVERFIT_JOB_EXIT code=$rc"; exit "$rc"' EXIT
stage() { printf '%s\n' "$1" > "$OVERFIT365_RUN/stage.txt"; echo "OVERFIT_STAGE $1"; }
nvidia-smi --query-gpu=name,memory.total --format=csv
stage gt_replay
"$SIM365_PYTHON" -u scripts/robocasa365/verify_overfit_demo.py --dataset "$DATASET365" --episode 5 --seed 100 --output "$OVERFIT365_RUN/gt"
stage baseline_probe_and_closed_loop
"$MODEL365_PYTHON" -u scripts/robocasa365/evaluate_policy.py --checkpoint "$OVERFIT365_BASE" \
  --dataset "$DATASET365" --sim-python "$SIM365_PYTHON" --output "$OVERFIT365_RUN/baseline" \
  --demo-episodes 5 --seeds 100 --probe-episode 5 --execute-horizon 8
export ROBO365_STATUS_MIRROR="$OVERFIT365_RUN/live"
stage warm_start_train_to_10
"$MODEL365_PYTHON" -u scripts/robocasa365/train_single_gpu.py --dataset "$DATASET365" --checkpoint "$OVERFIT365_GR1" \
  --output "$OVERFIT365_TRAIN" --warm-start "$OVERFIT365_BASE" --overfit-episode 5 \
  --action-steps 2000 --joint-steps 0 --accumulation 4 --stop-after 10 --snapshot-steps 500 1000 2000
stage resume_train_to_1000
"$MODEL365_PYTHON" -u scripts/robocasa365/train_single_gpu.py --dataset "$DATASET365" --checkpoint "$OVERFIT365_GR1" \
  --output "$OVERFIT365_TRAIN" --resume "$OVERFIT365_TRAIN/action_latest.pt" --overfit-episode 5 \
  --action-steps 2000 --joint-steps 0 --accumulation 4 --stop-after 1000 --snapshot-steps 500 1000 2000
stage probe_1000
"$MODEL365_PYTHON" -u scripts/robocasa365/evaluate_policy.py --checkpoint "$OVERFIT365_TRAIN/action_step_001000.pt" \
  --dataset "$DATASET365" --sim-python "$SIM365_PYTHON" --output "$OVERFIT365_RUN/probe_1000" \
  --validation-only --probe-episode 5
stage resume_train_to_2000
"$MODEL365_PYTHON" -u scripts/robocasa365/train_single_gpu.py --dataset "$DATASET365" --checkpoint "$OVERFIT365_GR1" \
  --output "$OVERFIT365_TRAIN" --resume "$OVERFIT365_TRAIN/action_latest.pt" --overfit-episode 5 \
  --action-steps 2000 --joint-steps 0 --accumulation 4 --snapshot-steps 500 1000 2000
stage final_probe_and_closed_loop
"$MODEL365_PYTHON" -u scripts/robocasa365/evaluate_policy.py --checkpoint "$OVERFIT365_TRAIN/action_step_002000.pt" \
  --dataset "$DATASET365" --sim-python "$SIM365_PYTHON" --output "$OVERFIT365_RUN/overfit_eval" \
  --demo-episodes 5 5 5 --seeds 100 101 102 --probe-episode 5 --execute-horizon 8
# Predeclared diagnostic: if H8 fails all three runs, test faster visual feedback
# separately, without relabeling this as the primary result or changing physics.
if "$SIM365_PYTHON" - "$OVERFIT365_RUN/overfit_eval/evaluation.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));assert r['status']=='complete'
sys.exit(0 if r['successes']==0 else 1)
PY
then
  stage diagnostic_execute_horizon_1
  "$MODEL365_PYTHON" -u scripts/robocasa365/evaluate_policy.py --checkpoint "$OVERFIT365_TRAIN/action_step_002000.pt" \
    --dataset "$DATASET365" --sim-python "$SIM365_PYTHON" --output "$OVERFIT365_RUN/overfit_h1" \
    --demo-episodes 5 --seeds 100 --execute-horizon 1
fi
stage artifact_verification
"$SIM365_PYTHON" -u scripts/robocasa365/summarize_overfit.py --run "$OVERFIT365_RUN" --train "$OVERFIT365_TRAIN"
stage complete
