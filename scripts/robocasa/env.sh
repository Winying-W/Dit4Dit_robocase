#!/usr/bin/env bash
DIT4DIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DIT4DIT_ROOT
export PYTHONPATH="$DIT4DIT_ROOT:$DIT4DIT_ROOT/third_party/robocasa-gr1${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$DIT4DIT_ROOT/.cache/egl${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export HF_HOME="$DIT4DIT_ROOT/.cache/huggingface" HF_HUB_OFFLINE=1
export MPLCONFIGDIR="$DIT4DIT_ROOT/.cache/matplotlib-gr1"
export NUMBA_CACHE_DIR="$DIT4DIT_ROOT/.cache/numba-gr1"
export XDG_CACHE_HOME="$DIT4DIT_ROOT/.cache/xdg-gr1"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export MODEL_PYTHON="$DIT4DIT_ROOT/.conda/bin/python"
export SIM_PYTHON="$DIT4DIT_ROOT/.sim-gr1/bin/python"
export CKPT="${CKPT:-$DIT4DIT_ROOT/checkpoints/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt}"
