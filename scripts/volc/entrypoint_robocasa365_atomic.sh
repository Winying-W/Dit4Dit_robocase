#!/usr/bin/env bash
set -Eeuo pipefail
umask 002
ROOT=/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT
RUN365="$ROOT/runs/robocasa365_atomic_20260924"
OUT365="$ROOT/artifacts/training/robocasa365_atomic_20260924"
BACKUP365=/file_system/efs/checkpoint/intern/haozhe.jia/projects/DiT4DiT
export MODEL365_PYTHON="$BACKUP365/conda-env/bin/python"
export DIT365_CACHE="$ROOT/artifacts/runtime/cache"
export IMAGEIO_FFMPEG_EXE="$BACKUP365/conda-env/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
export TORCHINDUCTOR_CACHE_DIR="$DIT365_CACHE/torchinductor"
export NO_ALBUMENTATIONS_UPDATE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROBO365_STATUS_MIRROR="$RUN365/train_live"
cd "$RUN365/source"
source scripts/robocasa365/env.sh
mkdir -p "$OUT365" "$RUN365/train_live"
exec 9>"$RUN365/run.lock"
flock -n 9 || { echo 'Another worker holds the run lock'; exit 2; }
exec > >(tee -a "$RUN365/cloud.log") 2>&1
trap 'code=$?; printf "failed code=%s line=%s\n" "$code" "$LINENO" > "$RUN365/stage.txt"; exit "$code"' ERR
GR1365="$ROOT/artifacts/checkpoints/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt"
MANIFEST365="$RUN365/prepared/manifest.json"
test -f "$GR1365"
test -f "$ROOT/artifacts/checkpoints/Cosmos-Predict2.5-2B-from-GR1/POLICY_DERIVATION.json"
test -f "$RUN365/adapter_verification/acceptance.json"
test -x "$MODEL365_PYTHON"
test -x "$SIM365_PYTHON"
"$MODEL365_PYTHON" -c 'import hashlib,json,pathlib,sys; root=pathlib.Path(sys.argv[1]); records=json.loads((root/"source_hashes.json").read_text()); assert all(hashlib.sha256((root/"source"/p).read_bytes()).hexdigest()==h for p,h in records.items()); replay=json.loads((root/"gt_replay_summary.json").read_text()); assert replay["status"]=="complete" and all(r["success"] for r in replay["episodes"])' "$RUN365"
nvidia-smi --query-gpu=name,memory.total --format=csv
TRAIN365=("$MODEL365_PYTHON" -u scripts/robocasa365/train_single_gpu.py
  --manifest "$MANIFEST365" --checkpoint "$GR1365" --output "$OUT365"
  --action-steps 10000 --joint-steps 0 --accumulation 4 --warmup-steps 500
  --validation-interval 500 --checkpoint-interval 500 --snapshot-steps 2000 5000 10000)
latest_step() {
  "$MODEL365_PYTHON" -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1])/"latest.json"; print(json.loads(p.read_text())["global_step"] if p.exists() else 0)' "$OUT365"
}
STEP365=$(latest_step)
if [ "$STEP365" -eq 0 ]; then
  printf 'preflight_10_updates\n' > "$RUN365/stage.txt"
  "${TRAIN365[@]}" --stop-after 10
fi
for GATE365 in 2000 5000; do
  STEP365=$(latest_step)
  if [ "$STEP365" -lt "$GATE365" ]; then
    printf 'training_to_%s_of_10000\n' "$GATE365" > "$RUN365/stage.txt"
    "${TRAIN365[@]}" --resume "$OUT365/action_latest.pt" --stop-after "$GATE365"
  fi
  if [ ! -f "$RUN365/development_${GATE365}_complete" ]; then
    printf -v SNAP365 '%06d' "$GATE365"
    printf 'development_%s_selection_seeds\n' "$GATE365" > "$RUN365/stage.txt"
    "$MODEL365_PYTHON" -u scripts/robocasa365/evaluate_multitask.py \
      --checkpoint "$OUT365/action_step_${SNAP365}.pt" --manifest "$MANIFEST365" \
      --output "$RUN365/development_${GATE365}" --sim-python "$SIM365_PYTHON" \
      --development --allow-task-subset
    touch "$RUN365/development_${GATE365}_complete"
  fi
done
STEP365=$(latest_step)
if [ "$STEP365" -lt 10000 ]; then
  printf 'training_to_10000\n' > "$RUN365/stage.txt"
  "${TRAIN365[@]}" --resume "$OUT365/action_latest.pt"
fi
printf 'final_evaluation_100\n' > "$RUN365/stage.txt"
"$MODEL365_PYTHON" -u scripts/robocasa365/evaluate_multitask.py \
  --checkpoint "$OUT365/action_step_010000.pt" --manifest "$MANIFEST365" \
  --output "$RUN365/evaluation_100" --sim-python "$SIM365_PYTHON" --allow-task-subset
printf 'complete\n' > "$RUN365/stage.txt"
