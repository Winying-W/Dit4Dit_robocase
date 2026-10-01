#!/usr/bin/env bash
set -Eeuo pipefail
umask 000
if [ ! -e /file_system ]; then ln -s / /file_system; fi
ROOT=/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT
cd "$ROOT"
source scripts/robocasa365/env.sh
export NO_ALBUMENTATIONS_UPDATE=1
OUT="$ROOT/runs/robocasa365_replay_probe_20260923"
mkdir -p "$OUT"
chmod 777 "$OUT"
exec > >(tee -a "$OUT/cloud.log") 2>&1
"$SIM365_PYTHON" -u scripts/robocasa365/diagnose_replay.py --dataset "$DATASET365" --output "$OUT/normal" \
 --episodes 0 3 5 --variants warmup warmup_forward warmup_device --max-steps 15
NUMBA_DISABLE_JIT=1 "$SIM365_PYTHON" -u scripts/robocasa365/diagnose_replay.py --dataset "$DATASET365" --output "$OUT/no_jit" \
 --episodes 0 3 5 --variants warmup --max-steps 15
