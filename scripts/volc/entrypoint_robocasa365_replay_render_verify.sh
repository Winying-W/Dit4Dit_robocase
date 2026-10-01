#!/usr/bin/env bash
set -Eeuo pipefail
umask 000
if [ ! -e /file_system ]; then ln -s / /file_system; fi
ROOT=/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT
cd "$ROOT"
source scripts/robocasa365/env.sh
export NO_ALBUMENTATIONS_UPDATE=1
OUT="$ROOT/runs/robocasa365_replay_fixed_20260923"
mkdir -p "$OUT"
chmod 777 "$OUT"
exec > >(tee -a "$OUT/render_verify.log") 2>&1
"$SIM365_PYTHON" -u scripts/robocasa365/replay.py --dataset "$DATASET365" --output "$OUT/video_repeat" --episodes 11
"$SIM365_PYTHON" scripts/robocasa365/verify_fixed_replay_outputs.py
