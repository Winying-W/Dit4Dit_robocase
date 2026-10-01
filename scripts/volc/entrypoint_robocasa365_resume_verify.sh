#!/usr/bin/env bash
set -Eeuo pipefail
umask 000
if [ ! -e /file_system ]; then ln -s / /file_system; fi
ROOT=/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT
cd "$ROOT"
source scripts/robocasa365/env.sh
export NO_ALBUMENTATIONS_UPDATE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROBO365_STATUS_MIRROR="$ROOT/runs/robocasa365_train_setup/live"
RUN365_TRAIN="$ROOT/results/robocasa365_train_20260922"
mkdir -p "$RUN365_TRAIN"
chmod 777 "$RUN365_TRAIN"
exec > >(tee -a "$RUN365_TRAIN/resume_verify.log") 2>&1
"$MODEL365_PYTHON" -u scripts/robocasa365/train_single_gpu.py \
 --dataset "$DATASET365" --output "$RUN365_TRAIN/train" \
 --checkpoint "$ROOT/checkpoints/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt" \
 --action-steps 300 --joint-steps 100 --accumulation 4 \
 --resume "$RUN365_TRAIN/train/partial_joint_latest.pt"
