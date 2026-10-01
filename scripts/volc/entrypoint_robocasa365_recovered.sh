#!/usr/bin/env bash
set -Eeuo pipefail
umask 002
ROOT=/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT
RUN365="$ROOT/runs/robocasa365_recovered_20260924"
OUT365="$ROOT/artifacts/training/robocasa365_recovered_20260924"
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
test -f "$RUN365/data_acceptance.json"
test -x "$MODEL365_PYTHON"
test -x "$SIM365_PYTHON"
"$MODEL365_PYTHON" -c 'import hashlib,json,pathlib,sys; root=pathlib.Path(sys.argv[1]); records=json.loads((root/"source_hashes.json").read_text()); assert all(hashlib.sha256((root/"source"/p).read_bytes()).hexdigest()==h for p,h in records.items())' "$RUN365"
nvidia-smi --query-gpu=name,memory.total --format=csv
TRAIN365=("$MODEL365_PYTHON" -u scripts/robocasa365/train_single_gpu.py
  --manifest "$MANIFEST365" --checkpoint "$GR1365" --output "$OUT365"
  --action-steps 50000 --joint-steps 0 --accumulation 4 --warmup-steps 1000
  --validation-interval 1000 --checkpoint-interval 1000
  --snapshot-steps 5000 10000 15000 20000 30000 40000 50000)
latest_step() {
  "$MODEL365_PYTHON" -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1])/"latest.json"; print(json.loads(p.read_text())["global_step"] if p.exists() else 0)' "$OUT365"
}
STEP365=$(latest_step)
if [ "$STEP365" -eq 0 ]; then
  printf 'preflight_10_updates\n' > "$RUN365/stage.txt"
  "${TRAIN365[@]}" --stop-after 10
fi
STEP365=$(latest_step)
if [ "$STEP365" -lt 15000 ]; then
  printf 'training_to_15000_of_50000\n' > "$RUN365/stage.txt"
  "${TRAIN365[@]}" --resume "$OUT365/action_latest.pt" --stop-after 15000
fi
if [ ! -f "$RUN365/development_complete" ]; then
  printf 'development_320_selection_seeds\n' > "$RUN365/stage.txt"
  "$MODEL365_PYTHON" -u scripts/robocasa365/evaluate_multitask.py \
    --checkpoint "$OUT365/action_step_015000.pt" --manifest "$MANIFEST365" \
    --output "$RUN365/development_15000" --sim-python "$SIM365_PYTHON" --development
  touch "$RUN365/development_complete"
fi
STEP365=$(latest_step)
if [ "$STEP365" -lt 50000 ]; then
  printf 'training_to_50000\n' > "$RUN365/stage.txt"
  "${TRAIN365[@]}" --resume "$OUT365/action_latest.pt"
fi
printf 'final_evaluation_1600\n' > "$RUN365/stage.txt"
"$MODEL365_PYTHON" -u scripts/robocasa365/evaluate_multitask.py \
  --checkpoint "$OUT365/action_step_050000.pt" --manifest "$MANIFEST365" \
  --output "$RUN365/evaluation_1600" --sim-python "$SIM365_PYTHON"
printf 'complete\n' > "$RUN365/stage.txt"
