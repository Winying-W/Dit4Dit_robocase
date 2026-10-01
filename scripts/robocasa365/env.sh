#!/usr/bin/env bash
DIT365_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DIT365_ROOT
export ROBOCASA365_ROOT="${ROBOCASA365_ROOT:-/file_system/vepfs/intern/haozhe.jia/projects/openpi/robocasa365}"
export PYTHONPATH="$DIT365_ROOT:$ROBOCASA365_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export DIT365_CACHE="${DIT365_CACHE:-$DIT365_ROOT/.cache}"
export LD_LIBRARY_PATH="$DIT365_CACHE/egl:${LD_LIBRARY_PATH:-}"
export IMAGEIO_FFMPEG_EXE="${IMAGEIO_FFMPEG_EXE:-$DIT365_ROOT/.conda/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}"
export HF_HOME="${HF_HOME:-$DIT365_CACHE/huggingface}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export DATASET365="${DATASET365:-$DIT365_ROOT/results/robocasa365_interface/datasets/v1.0/target/composite/StirVegetables/20250814}"
export SIM365_PYTHON="${SIM365_PYTHON:-$ROBOCASA365_ROOT/.venv/bin/python}"
export MODEL365_PYTHON="${MODEL365_PYTHON:-$DIT365_ROOT/.conda/bin/python}"
