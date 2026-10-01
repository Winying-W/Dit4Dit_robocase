#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$DIT4DIT_ROOT"
GPU_ID=${GPU_ID:-2}
PORT=${PORT:-15694}
TRIALS=${TRIALS:-1}
SUITES=${SUITES:-libero_spatial}
CKPT=${CKPT:-$DIT4DIT_ROOT/checkpoints/dit4dit-model/dit4dit_libero/final_model/pytorch_model.pt}
RUN_DIR=${RUN_DIR:-$DIT4DIT_ROOT/results/libero_$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# MuJoCo uses the physical EGL device index rather than CUDA's remapped index.
export MUJOCO_EGL_DEVICE_ID="$GPU_ID"
python scripts/local/configure_libero.py
python - "$CKPT" <<'PY'
import pathlib, sys, yaml
p=pathlib.Path(sys.argv[1])
assert p.is_file(), f'Missing policy checkpoint: {p}'
config=yaml.safe_load((p.parent.parent/'config.yaml').read_text())
base=pathlib.Path(config['framework']['cosmos25']['base_model'])
assert (base/'model_index.json').is_file(), f'Missing Cosmos backbone: {base}'
PY
# Refuse an occupied port so an unrelated policy cannot supply our evaluation.
python - "$PORT" <<'PY'
import socket,sys
s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(('127.0.0.1', int(sys.argv[1]))); s.close()
PY
python -u deployment/model_server/server_policy.py --ckpt_path "$CKPT" --port "$PORT" > "$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
python - "$SERVER_PID" "$PORT" <<'PY'
import os, sys, time
from websockets.sync.client import connect
from websockets.exceptions import WebSocketException
pid,port=map(int,sys.argv[1:]); deadline=time.monotonic()+900
while time.monotonic()<deadline:
    try: os.kill(pid,0)
    except ProcessLookupError: raise SystemExit('Policy server exited; inspect server.log')
    try:
        with connect(f'ws://127.0.0.1:{port}',open_timeout=1,proxy=None) as ws:
            ws.recv(timeout=1)
            break
    except (OSError,TimeoutError,WebSocketException): time.sleep(2)
else: raise SystemExit('Policy server readiness timed out; inspect server.log')
print('Policy server ready',flush=True)
PY
TASK_ARGS=()
if [[ -n "${TASK_IDS:-}" ]]; then
    read -ra IDS <<< "$TASK_IDS"
    TASK_ARGS=(--args.task-ids "${IDS[@]}")
fi
for SUITE in $SUITES; do
    python -u examples/LIBERO/eval_files/eval_libero.py \
        --args.pretrained-path "$CKPT" --args.host 127.0.0.1 --args.port "$PORT" \
        --args.task-suite-name "$SUITE" --args.num-trials-per-task "$TRIALS" \
        --args.video-out-path "$RUN_DIR/$SUITE" "${TASK_ARGS[@]}" \
        2>&1 | tee "$RUN_DIR/${SUITE}.log"
done
printf 'Results: %s\n' "$RUN_DIR"
