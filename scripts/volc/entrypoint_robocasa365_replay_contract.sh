#!/usr/bin/env bash
set -Eeuo pipefail
umask 000
if [ ! -e /file_system ]; then ln -s / /file_system; fi
ROOT=/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT
cd "$ROOT"
source scripts/robocasa365/env.sh
export NO_ALBUMENTATIONS_UPDATE=1
OUT="$ROOT/runs/robocasa365_replay_contract_20260923"
mkdir -p "$OUT"
chmod 777 "$OUT"
exec > >(tee -a "$OUT/cloud.log") 2>&1
cp scripts/robocasa365/verify_replay_contract.py "$OUT/verify_replay_contract_source.py"
"$SIM365_PYTHON" -u scripts/robocasa365/verify_replay_contract.py --dataset "$DATASET365" --output "$OUT" --episodes 0 3 11
