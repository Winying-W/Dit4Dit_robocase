#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$DIT4DIT_ROOT"
UV=${UV:-/file_system/vepfs/intern/haozhe.jia/.local/bin/uv}
export UV_CACHE_DIR="$DIT4DIT_ROOT/.cache/uv"
export UV_CONCURRENT_DOWNLOADS=4
export UV_HTTP_TIMEOUT=120
# DiT4DiT only uses pure-Python pytorch3d.transforms.
export PYTORCH3D_NO_EXTENSION=1
# Use the host proxy if HTTPS_PROXY is configured.
export UV_LINK_MODE=hardlink
"$UV" pip install --python .conda/bin/python torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install --python .conda/bin/python pip wheel setuptools==75.8.2 packaging ninja
"$UV" pip install --python .conda/bin/python --no-build-isolation -r scripts/local/requirements-cu128.txt
"$UV" pip install --python .conda/bin/python --no-build-isolation -r scripts/local/requirements-libero.txt -e . -e third_party/LIBERO
python scripts/local/configure_libero.py
"$UV" pip check --python .conda/bin/python
