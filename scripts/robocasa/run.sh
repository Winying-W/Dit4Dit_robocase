#!/usr/bin/env bash
set -Eeuo pipefail
umask 000
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$DIT4DIT_ROOT"
GPU_LIST=${GPU_LIST:-0}
EPISODES=${EPISODES:-50}
RUN_DIR=${RUN_DIR:-$DIT4DIT_ROOT/results/gr1_$(date -u +%Y%m%dT%H%M%SZ)}
BASE_PORT=${BASE_PORT:-16398}
TASK_INDICES=${TASK_INDICES:-$(seq -s ' ' 0 23)}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
if [[ -f "$RUN_DIR/summary.json" ]]; then
    echo "Refusing to overwrite an existing evaluated run: $RUN_DIR" >&2
    exit 1
fi
mkdir -p "$RUN_DIR"
chmod 777 "$RUN_DIR"
export RUN_DIR
IFS=',' read -ra GPUS <<< "$GPU_LIST"
read -ra TASKS <<< "$TASK_INDICES"
PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do
        # Each worker owns its policy server and simulation process group.
        kill -- -"$pid" 2>/dev/null || true
    done
    wait || true
}
trap cleanup EXIT
if [[ $PREFLIGHT_ONLY == 1 ]]; then
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" MUJOCO_EGL_DEVICE_ID="${GPUS[0]}" \
        "$SIM_PYTHON" -u scripts/robocasa/preflight.py --output "$RUN_DIR" --tasks "${TASKS[@]}" \
        2>&1 | tee "$RUN_DIR/preflight.log"
    exit 0
fi
# Precheck checkpoint/config/port before starting GPU workers.
"$MODEL_PYTHON" - "$CKPT" "$BASE_PORT" "${#GPUS[@]}" <<'PY'
import json,pathlib,socket,sys,yaml
p=pathlib.Path(sys.argv[1]);assert p.is_file(),p
cfg=yaml.safe_load((p.parent.parent/'config.yaml').read_text())
base=pathlib.Path(cfg['framework']['cosmos25']['base_model'])
for f in ['model_index.json','text_encoder/model.safetensors','transformer/diffusion_pytorch_model.safetensors','vae/diffusion_pytorch_model.safetensors']:
    assert (base/f).is_file(),base/f
stats=json.loads((p.parent.parent/'dataset_statistics.json').read_text())
assert len(stats)==1 and len(next(iter(stats.values()))['action']['min'])==29
for port in range(int(sys.argv[2]),int(sys.argv[2])+int(sys.argv[3])):
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',port))
PY
for slot in "${!GPUS[@]}"; do
    ASSIGNED=()
    for j in "${!TASKS[@]}"; do
        if (( j % ${#GPUS[@]} == slot )); then ASSIGNED+=("${TASKS[j]}"); fi
    done
    if (( ${#ASSIGNED[@]} == 0 )); then continue; fi
    setsid bash scripts/robocasa/worker.sh "${GPUS[slot]}" "$((BASE_PORT+slot))" "$EPISODES" "${ASSIGNED[@]}" \
        > "$RUN_DIR/worker_${slot}.log" 2>&1 &
    PIDS+=("$!")
done
# Fail as soon as any worker fails; never summarize a failed shard as success.
remaining=${#PIDS[@]}
while (( remaining > 0 )); do
    wait -n
    remaining=$((remaining-1))
done
"$MODEL_PYTHON" scripts/robocasa/summarize.py "$RUN_DIR" \
    --expected-tasks "${#TASKS[@]}" --expected-episodes "$EPISODES"
