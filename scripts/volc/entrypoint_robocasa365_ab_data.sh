#!/usr/bin/env bash
set -Eeuo pipefail
umask 000
if [ ! -e /file_system ]; then ln -s / /file_system; fi
ROOT=/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT
cd "$ROOT"
source scripts/robocasa365/env.sh
export NO_ALBUMENTATIONS_UPDATE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN365="$ROOT/runs/robocasa365_multitask_20260924"
TRAIN365="$ROOT/results/robocasa365_multitask_20260924/ab"
BASE365="$ROOT/results/robocasa365_dual_20260923/train/action_step_002000.pt"
GR1365="$ROOT/checkpoints/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt"
mkdir -p "$RUN365/data" "$TRAIN365"
exec 9>"$RUN365/cloud.lock"
flock -n 9 || exit 2
if [ -e "$RUN365/started" ]; then echo 'Existing run; explicit recovery required'; exit 2; fi
touch "$RUN365/started"
exec > >(tee -a "$RUN365/cloud.log") 2>&1
trap 'rc=$?; echo "AB_DATA_JOB_EXIT code=$rc"; exit "$rc"' EXIT
cd "$RUN365/source"
export PYTHONPATH="$PWD:$ROOT:$ROBOCASA365_ROOT${PYTHONPATH:+:$PYTHONPATH}"
stage365() { printf '%s\n' "$1" > "$RUN365/stage.txt"; echo "AB_DATA_STAGE $1"; }
nvidia-smi --query-gpu=name,memory.total --format=csv
# All 16 datasets were independently downloaded and verified on vePFS.
test -f "$RUN365/adapter_verification/acceptance.json"
export DATASET365="$ROOT/data/robocasa365/v1.0/target/composite/StirVegetables/20250814"
stage365 ab_on_verified_data
# Predeclared paired experiment: same starting weights, 16 demos, normalization,
# NumPy sampling seed, 1000 updates and accumulation=4. Video loss is B's treatment.
for ARM365 in frozen partial_joint; do
  mkdir -p "$RUN365/ab/$ARM365/live"
  export ROBO365_STATUS_MIRROR="$RUN365/ab/$ARM365/live"
  if [ "$ARM365" = frozen ]; then
    ACTION365=1000; JOINT365=0; PHASE365=action
  else
    ACTION365=0; JOINT365=1000; PHASE365=partial_joint
  fi
  stage365 "train_$ARM365"
  "$MODEL365_PYTHON" -u scripts/robocasa365/train_single_gpu.py \
    --dataset "$DATASET365" --checkpoint "$GR1365" --output "$TRAIN365/$ARM365" \
    --warm-start "$BASE365" --action-steps "$ACTION365" --joint-steps "$JOINT365" \
    --accumulation 4 --seed 42 --snapshot-steps 1000
  stage365 "eval_$ARM365"
  "$MODEL365_PYTHON" -u scripts/robocasa365/evaluate_policy.py \
    --checkpoint "$TRAIN365/$ARM365/${PHASE365}_step_001000.pt" --dataset "$DATASET365" \
    --sim-python "$SIM365_PYTHON" --output "$RUN365/ab/$ARM365/eval" \
    --seeds 100 101 102 103 104 105 106 107 108 109 --execute-horizon 8
  "$SIM365_PYTHON" scripts/robocasa365/audit_saved_actions.py \
    --phase "$RUN365/ab/$ARM365/eval" --normalization "$TRAIN365/$ARM365/normalization.json" \
    --output "$RUN365/ab/$ARM365/action_audit.json"
done
stage365 select_recipe
"$SIM365_PYTHON" scripts/robocasa365/select_ab.py --run "$RUN365" --train-root "$TRAIN365"
stage365 ab_and_data_complete
