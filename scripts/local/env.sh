#!/usr/bin/env bash
# Source this file from any working directory.
DIT4DIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DIT4DIT_ROOT
source /file_system/vepfs/intern/haozhe.jia/miniforge3/etc/profile.d/conda.sh
conda activate dit4dit
export PYTHONPATH="$DIT4DIT_ROOT:$DIT4DIT_ROOT/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="$DIT4DIT_ROOT/.cache/uv"
export UV_LINK_MODE=hardlink
export HF_HOME="$DIT4DIT_ROOT/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
export LIBERO_CONFIG_PATH="$DIT4DIT_ROOT/.cache/libero"
export LD_LIBRARY_PATH="$DIT4DIT_ROOT/.cache/egl${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MPLCONFIGDIR="$DIT4DIT_ROOT/.cache/matplotlib"
export NUMBA_CACHE_DIR="$DIT4DIT_ROOT/.cache/numba"
export XDG_CACHE_HOME="$DIT4DIT_ROOT/.cache/xdg"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
