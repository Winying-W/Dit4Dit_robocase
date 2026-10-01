#!/usr/bin/env bash
set -Eeuo pipefail
umask 000
if [ ! -e /file_system ]; then ln -s / /file_system; fi
ROOT=/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT
cd "$ROOT"
source scripts/robocasa/env.sh
export RUN_DIR="$ROOT/results/gr1_volc_20260920"
mkdir -p "$RUN_DIR"
chmod 777 "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/entrypoint.log") 2>&1
nvidia-smi
# CPU package/import validation and an actual EGL reset/step before policy loading.
GPU_LIST=0 TASK_INDICES=20 PREFLIGHT_ONLY=1 RUN_DIR="$RUN_DIR/startup_preflight" bash scripts/robocasa/run.sh
export GPU_LIST=0,1,2,3,4,5,6,7 EPISODES=50 USE_BF16=1 VIDEOS=1
bash scripts/robocasa/run.sh
