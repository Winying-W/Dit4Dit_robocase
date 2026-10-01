#!/usr/bin/env bash
set -Eeuo pipefail
umask 000
if [ ! -e /file_system ]; then ln -s / /file_system; fi
ROOT=/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT
cd "$ROOT"
source scripts/robocasa365/env.sh
export NO_ALBUMENTATIONS_UPDATE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MULTI365_RUN="$ROOT/runs/robocasa365_multitask_20260924"
MULTI365_OUT="$ROOT/results/robocasa365_multitask_20260924/multitask_train"
MULTI365_MANIFEST="$MULTI365_RUN/prepared/manifest.json"
MULTI365_GR1="$ROOT/checkpoints/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt"
# Recipe selection is based only on the prior paired 10-seed StirVegetables
# experiment; never select a recipe/checkpoint using the final 320 trials.
test -f "$MULTI365_RUN/ab_selection.json"
test -f "$MULTI365_RUN/adapter_verification/acceptance.json"
test -f "$MULTI365_GR1"
mkdir -p "$MULTI365_RUN/train_live" "$MULTI365_OUT"
exec 9>"$MULTI365_RUN/multitask.lock"
flock -n 9 || exit 2
if [ -e "$MULTI365_RUN/multitask_started" ]; then echo 'Existing run requires explicit recovery'; exit 2; fi
touch "$MULTI365_RUN/multitask_started"
exec > >(tee -a "$MULTI365_RUN/multitask_cloud.log") 2>&1
cd "$MULTI365_RUN/source_multitask"
export PYTHONPATH="$PWD:$ROOT:$ROBOCASA365_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ROBO365_STATUS_MIRROR="$MULTI365_RUN/train_live"
MULTI365_RECIPE=$("$MODEL365_PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="complete"; print(r["recipe"])' "$MULTI365_RUN/ab_selection.json")
case "$MULTI365_RECIPE" in
  frozen) ACTION365=15000; JOINT365=0; FINAL365=action ;;
  partial_joint) ACTION365=12000; JOINT365=3000; FINAL365=partial_joint ;;
  *) echo "Invalid recipe: $MULTI365_RECIPE"; exit 2 ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv
printf 'train_preflight\n' > "$MULTI365_RUN/multitask_stage.txt"
"$MODEL365_PYTHON" -u scripts/robocasa365/train_single_gpu.py \
  --manifest "$MULTI365_MANIFEST" --checkpoint "$MULTI365_GR1" --output "$MULTI365_OUT" \
  --action-steps "$ACTION365" --joint-steps "$JOINT365" --accumulation 4 --stop-after 10 \
  --validation-interval 1000 --checkpoint-interval 1000 --snapshot-steps 5000 10000 12000 15000
printf 'training\n' > "$MULTI365_RUN/multitask_stage.txt"
"$MODEL365_PYTHON" -u scripts/robocasa365/train_single_gpu.py \
  --manifest "$MULTI365_MANIFEST" --checkpoint "$MULTI365_GR1" --output "$MULTI365_OUT" \
  --resume "$MULTI365_OUT/action_latest.pt" --action-steps "$ACTION365" --joint-steps "$JOINT365" \
  --accumulation 4 --validation-interval 1000 --checkpoint-interval 1000 --snapshot-steps 5000 10000 12000 15000
printf 'evaluation_320\n' > "$MULTI365_RUN/multitask_stage.txt"
"$MODEL365_PYTHON" -u scripts/robocasa365/evaluate_multitask.py \
  --checkpoint "$MULTI365_OUT/${FINAL365}_step_015000.pt" --manifest "$MULTI365_MANIFEST" \
  --output "$MULTI365_RUN/evaluation_320" --sim-python "$SIM365_PYTHON"
printf 'complete\n' > "$MULTI365_RUN/multitask_stage.txt"
