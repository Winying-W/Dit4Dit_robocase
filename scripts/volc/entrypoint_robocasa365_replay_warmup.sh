#!/usr/bin/env bash
set -Eeuo pipefail
umask 000
if [ ! -e /file_system ]; then ln -s / /file_system; fi
ROOT=/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT
cd "$ROOT"
source scripts/robocasa365/env.sh
export NO_ALBUMENTATIONS_UPDATE=1
OUT="$ROOT/runs/robocasa365_replay_warmup_20260923"
mkdir -p "$OUT"
chmod 777 "$OUT"
exec > >(tee -a "$OUT/cloud.log") 2>&1
"$SIM365_PYTHON" -u scripts/robocasa365/diagnose_replay.py --dataset "$DATASET365" --output "$OUT" \
 --episodes 0 3 5 10 11 --variants warmup official
