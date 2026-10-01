#!/usr/bin/env bash
set -Eeuo pipefail
umask 000
if [ ! -e /file_system ]; then ln -s / /file_system; fi
ROOT=/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT
cd "$ROOT"
source scripts/robocasa365/env.sh
export NO_ALBUMENTATIONS_UPDATE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN365_DUAL="$ROOT/results/robocasa365_dual_20260923"
MIRROR365="$ROOT/runs/robocasa365_dual_20260923"
mkdir -p "$RUN365_DUAL/train" "$MIRROR365/live" "$MIRROR365/replay"
chmod 777 "$RUN365_DUAL" "$RUN365_DUAL/train" "$MIRROR365"
exec > >(tee -a "$MIRROR365/cloud.log") 2>&1
trap 'rc=$?; echo "DUAL_JOB_EXIT code=$rc"; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv
# No rendering or CUDA in this independent diagnostic process.
CUDA_VISIBLE_DEVICES='' "$SIM365_PYTHON" -u scripts/robocasa365/analyze_replay_failure.py \
  --dataset "$DATASET365" --episodes 0 3 \
  --traces "$ROOT/runs/robocasa365_replay_warmup_20260923" \
  --output "$MIRROR365/replay" > "$MIRROR365/replay.log" 2>&1 &
REPLAY365_PID=$!
export ROBO365_STATUS_MIRROR="$MIRROR365/live"
"$MODEL365_PYTHON" -u scripts/robocasa365/train_single_gpu.py \
  --dataset "$DATASET365" --output "$RUN365_DUAL/train" \
  --checkpoint "$ROOT/checkpoints/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt" \
  --resume "$ROOT/results/robocasa365_train_20260922/train/action_latest.pt" \
  --action-steps 2000 --joint-steps 0 --accumulation 4 \
  --extend-action-schedule --snapshot-steps 500 1000 2000
# A second process proves the incremental checkpoint is independently recoverable.
"$MODEL365_PYTHON" -u scripts/robocasa365/train_single_gpu.py \
  --dataset "$DATASET365" --output "$RUN365_DUAL/resume_verify" \
  --checkpoint "$ROOT/checkpoints/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt" \
  --resume "$RUN365_DUAL/train/action_latest.pt" --action-steps 2000 --joint-steps 0 --accumulation 4
wait "$REPLAY365_PID"
echo DUAL_TRAIN_REPLAY_COMPLETE
