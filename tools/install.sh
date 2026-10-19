#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10"'
python -m pip install --upgrade pip setuptools wheel
# Set TORCH_INDEX_URL for a compatible alternative build if required by the host driver.
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
python -m pip install -r requirements.txt
python -m pip install ninja iopath fvcore
# Only pytorch3d.transforms is used; no CUDA extensions are needed by this workflow.
PYTORCH3D_NO_EXTENSION=1 python -m pip install --no-build-isolation 'pytorch3d @ https://github.com/facebookresearch/pytorch3d/archive/refs/tags/v0.7.9.tar.gz'
python -m pip check
python tools/check_environment.py
