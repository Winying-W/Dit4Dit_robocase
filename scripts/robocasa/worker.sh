#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$DIT4DIT_ROOT"
GPU=$1; PORT=$2; EPISODES=$3; shift 3
export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU"
SERVER_PID=''
trap '[[ -z "$SERVER_PID" ]] || kill "$SERVER_PID" 2>/dev/null || true' EXIT
BF16=()
# Official recommendation: bf16 on A100/A800, fp32 on RTX/L20.
if [[ ${USE_BF16:-auto} == 1 ]] || { [[ ${USE_BF16:-auto} == auto ]] && nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU" | grep -Eq 'A100|A800|H100|H800'; }; then BF16=(--use_bf16); fi
"$MODEL_PYTHON" -u deployment/model_server/server_policy.py --ckpt_path "$CKPT" --port "$PORT" "${BF16[@]}" \
    > "$RUN_DIR/server_gpu${GPU}.log" 2>&1 &
SERVER_PID=$!
"$MODEL_PYTHON" - "$SERVER_PID" "$PORT" <<'PY'
import os,sys,time
from websockets.sync.client import connect
from websockets.exceptions import WebSocketException
pid,port=map(int,sys.argv[1:]);deadline=time.monotonic()+900
while time.monotonic()<deadline:
    try: os.kill(pid,0)
    except ProcessLookupError: raise SystemExit('Policy server exited; inspect server log')
    try:
        with connect(f'ws://127.0.0.1:{port}',open_timeout=1,proxy=None) as ws: ws.recv(timeout=1)
        break
    except (OSError,TimeoutError,WebSocketException): time.sleep(2)
else: raise SystemExit('Policy startup timed out')
print('Policy ready',flush=True)
PY
for task in "$@"; do
    "$SIM_PYTHON" -u scripts/robocasa/evaluate.py --task-index "$task" --episodes "$EPISODES" \
        --port "$PORT" --output "$RUN_DIR" --videos "${VIDEOS:-1}" \
        > "$RUN_DIR/task_${task}.log" 2>&1
done
