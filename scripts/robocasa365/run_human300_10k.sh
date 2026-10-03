#!/usr/bin/env bash
set -euo pipefail
ROOT365=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT365"
: "${CUDA_VISIBLE_DEVICES:?Set available GPU IDs before launching}"
export PYTHONPATH="$ROOT365${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export ACCELERATE_USE_DEEPSPEED=false ACCELERATE_USE_FSDP=false
export TMPDIR="$ROOT365/.cache/human300-tmp"
export HF_HOME="$ROOT365/.cache/huggingface"
export TORCH_HOME="$ROOT365/.cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$ROOT365/.cache/torchinductor"
export CUDA_CACHE_PATH="$ROOT365/.cache/cuda"
mkdir -p "$TMPDIR" "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"
PY365="$ROOT365/.venv-human300/bin/python"
MANIFEST365="${HUMAN300_MANIFEST:-$ROOT365/runs/robocasa365_human300_setup/full_data/manifest.json}"
OUTPUT365="${HUMAN300_OUTPUT:-$ROOT365/runs/robocasa365_human300_10k}"
NPROC365=$("$PY365" -c 'import os; ids=os.environ["CUDA_VISIBLE_DEVICES"].split(","); assert len(ids)==len(set(ids)) and len(ids) in (1,2,4), "Expected 1, 2 or 4 distinct available GPUs"; print(len(ids))')
ACCUM365=$((64 / NPROC365))
exec "$PY365" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC365" \
  -m scripts.robocasa365.train_human300 \
  --config "$ROOT365/DiT4DiT/config/robocasa/dit4dit_human300_10k.yaml" \
  --manifest "$MANIFEST365" --output "$OUTPUT365" \
  --steps 10000 --warmup 500 --microbatch 1 --accumulation "$ACCUM365" \
  --save-every 2000 --validate-every 1000 --validation-examples 120 --prediction-examples 16 "$@"
