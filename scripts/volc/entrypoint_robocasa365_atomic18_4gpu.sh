#!/usr/bin/env bash
set -Eeuo pipefail
umask 002
ROOT=/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT
RUN365="$ROOT/runs/robocasa365_atomic18_20260924"
OUT365="$ROOT/artifacts/training/robocasa365_atomic18_20260924"
BACKUP365=/file_system/efs/checkpoint/intern/haozhe.jia/projects/DiT4DiT
export MODEL365_PYTHON="$BACKUP365/conda-env/bin/python"
export DIT365_CACHE="$ROOT/artifacts/runtime/cache"
export IMAGEIO_FFMPEG_EXE="$BACKUP365/conda-env/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
export TORCHINDUCTOR_CACHE_DIR="$DIT365_CACHE/torchinductor"
export NO_ALBUMENTATIONS_UPDATE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
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
test -f "$RUN365/prequeue_acceptance.json"
test -x "$MODEL365_PYTHON"
test -x "$SIM365_PYTHON"
"$MODEL365_PYTHON" - "$RUN365" <<'PY'
import hashlib,json,pathlib,sys,torch
root=pathlib.Path(sys.argv[1])
records=json.loads((root/'source_hashes.json').read_text())
assert all(hashlib.sha256((root/'source'/p).read_bytes()).hexdigest()==h for p,h in records.items())
report=json.loads((root/'prequeue_acceptance.json').read_text())
assert report['status']=='complete'
assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in report['evidence_sha256'].items())
assert torch.cuda.device_count()==4
PY
nvidia-smi --query-gpu=name,memory.total --format=csv
TRAIN365=("$MODEL365_PYTHON" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4
  scripts/robocasa365/train_distributed.py --manifest "$MANIFEST365" --checkpoint "$GR1365"
  --output "$OUT365" --steps 50000 --microbatch 4 --accumulation 4 --warmup-steps 1000
  --validation-interval 1000 --checkpoint-interval 1000 --snapshot-steps 10 12 10000 25000 50000
  --stop-file "$RUN365/stop_requested")
latest_step() {
  "$MODEL365_PYTHON" -c 'import pathlib,sys,torch; p=pathlib.Path(sys.argv[1])/"action_latest.pt"; print(torch.load(p,map_location="cpu",mmap=True,weights_only=False)["global_step"] if p.exists() else 0)' "$OUT365"
}
STEP365=$(latest_step)
if [ "$STEP365" -eq 0 ]; then
  printf 'four_gpu_preflight_10_updates\n' > "$RUN365/stage.txt"
  timeout --signal=TERM --kill-after=60s 1200s "${TRAIN365[@]}" --stop-after 10
fi
test ! -f "$RUN365/stop_requested" || exit 0
STEP365=$(latest_step)
if [ "$STEP365" -lt 12 ]; then
  printf 'four_gpu_independent_resume_gate\n' > "$RUN365/stage.txt"
  timeout --signal=TERM --kill-after=60s 1200s "${TRAIN365[@]}" --resume "$OUT365/action_latest.pt" --stop-after 12
fi
test ! -f "$RUN365/stop_requested" || exit 0
"$MODEL365_PYTHON" -c 'import json,pathlib,sys; out=pathlib.Path(sys.argv[1]); assert json.loads((out/"latest.json").read_text())["global_step"]>=12; assert json.loads((out/"rank_weights_step_000010.json").read_text())["passed"]; start=json.loads((out/"distributed_preflight.json").read_text())["resume_step"]; assert start>0; assert json.loads((out/f"rank_weights_step_{start+1:06d}.json").read_text())["passed"]' "$OUT365"
touch "$RUN365/four_gpu_gate_complete"
if [ ! -f "$RUN365/four_gpu_evaluation_gate_complete" ]; then
  printf 'four_gpu_policy_simulator_gate\n' > "$RUN365/stage.txt"
  timeout --signal=TERM --kill-after=60s 1200s "$MODEL365_PYTHON" -u scripts/robocasa365/verify_four_gpu_evaluation.py \
    --checkpoint "$OUT365/action_step_000012.pt" --manifest "$MANIFEST365" \
    --output "$RUN365/evaluation_smoke" --sim-python "$SIM365_PYTHON"
  touch "$RUN365/four_gpu_evaluation_gate_complete"
fi
test ! -f "$RUN365/stop_requested" || exit 0
for GATE365 in 10000 25000; do
  STEP365=$(latest_step)
  if [ "$STEP365" -lt "$GATE365" ]; then
    printf 'training_to_%s_of_50000\n' "$GATE365" > "$RUN365/stage.txt"
    "${TRAIN365[@]}" --resume "$OUT365/action_latest.pt" --stop-after "$GATE365"
  fi
  test ! -f "$RUN365/stop_requested" || exit 0
  if [ ! -f "$RUN365/development_${GATE365}_complete" ]; then
    printf -v SNAP365 '%06d' "$GATE365"
    printf 'development_%s_360_trials\n' "$GATE365" > "$RUN365/stage.txt"
    "$MODEL365_PYTHON" -u scripts/robocasa365/evaluate_multitask.py \
      --checkpoint "$OUT365/action_step_${SNAP365}.pt" --manifest "$MANIFEST365" \
      --output "$RUN365/development_${GATE365}" --sim-python "$SIM365_PYTHON" --development --workers 4
    touch "$RUN365/development_${GATE365}_complete"
  fi
done
STEP365=$(latest_step)
if [ "$STEP365" -lt 50000 ]; then
  printf 'training_to_50000\n' > "$RUN365/stage.txt"
  "${TRAIN365[@]}" --resume "$OUT365/action_latest.pt"
fi
test ! -f "$RUN365/stop_requested" || exit 0
printf 'final_evaluation_1800\n' > "$RUN365/stage.txt"
"$MODEL365_PYTHON" -u scripts/robocasa365/evaluate_multitask.py \
  --checkpoint "$OUT365/action_step_050000.pt" --manifest "$MANIFEST365" \
  --output "$RUN365/evaluation_1800" --sim-python "$SIM365_PYTHON" --workers 4
printf 'complete\n' > "$RUN365/stage.txt"
