#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$DIT365_ROOT"
RUN365_OUTPUT="${RUN365_OUTPUT:-$DIT365_ROOT/results/robocasa365_interface/model_128}"
mkdir -p "$RUN365_OUTPUT"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-5}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$MODEL365_PYTHON" -u scripts/robocasa365/check_model.py \
  --dataset "$DATASET365" --output "$RUN365_OUTPUT" --image-size 128 \
  --checkpoint "$DIT365_ROOT/checkpoints/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt"
"$MODEL365_PYTHON" -u scripts/robocasa365/verify_adapter.py --output "$RUN365_OUTPUT"
