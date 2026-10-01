#!/usr/bin/env bash
set -Eeuo pipefail
umask 000
if [ ! -e /file_system ]; then ln -s / /file_system; fi
ROOT=/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT
cd "$ROOT"
source scripts/robocasa365/env.sh
export NO_ALBUMENTATIONS_UPDATE=1
export ROBO365_STATUS_MIRROR="$ROOT/runs/robocasa365_train_setup/live"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN365_TRAIN="${RUN365_TRAIN:-$ROOT/results/robocasa365_train_20260922}"
mkdir -p "$RUN365_TRAIN"
chmod 777 "$RUN365_TRAIN"
exec > >(tee -a "$RUN365_TRAIN/entrypoint.log") 2>&1
trap 'rc=$?; echo "TRAIN_JOB_EXIT code=$rc"; exit "$rc"' EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv
# Reuse the NAS-mounted environment and one task only; no developer GPU use.
test -x "$MODEL365_PYTHON"
test -f "$ROOT/checkpoints/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt"
if [ ! -f "$DATASET365/meta/info.json" ]; then
    "$SIM365_PYTHON" "$ROOT/../openpi/scripts/robocasa/download_robocasa365.py" \
        --tasks StirVegetables --category composite --split target --workers 1 \
        --dest "$ROOT/results/robocasa365_interface/datasets/v1.0"
fi
"$MODEL365_PYTHON" - <<'PY'
import os,json
from pathlib import Path
p=Path(os.environ['DATASET365']); info=json.loads((p/'meta/info.json').read_text())
episodes=[json.loads(x) for x in (p/'meta/episodes.jsonl').read_text().splitlines()]
for ep in episodes:
    i=ep['episode_index']
    assert list((p/'data').glob(f'*/episode_{i:06d}.parquet'))
    for cam in ['robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand']:
        assert list((p/'videos').glob(f'*/observation.images.{cam}/episode_{i:06d}.mp4'))
    assert (p/f'extras/episode_{i:06d}/states.npz').exists()
print('DATASET_READY',json.dumps({'path':str(p),'episodes':len(episodes),'fps':info['fps']}),flush=True)
PY
"$MODEL365_PYTHON" -u scripts/robocasa365/check_model.py \
    --dataset "$DATASET365" --output "$RUN365_TRAIN/preflight" --image-size 128 \
    --checkpoint "$ROOT/checkpoints/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt"
"$MODEL365_PYTHON" -u scripts/robocasa365/verify_adapter.py --output "$RUN365_TRAIN/preflight"
"$MODEL365_PYTHON" -u scripts/robocasa365/train_single_gpu.py \
    --dataset "$DATASET365" --output "$RUN365_TRAIN/train" \
    --checkpoint "$ROOT/checkpoints/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt" \
    --action-steps 300 --joint-steps 100 --accumulation 4 --stop-after 10
# Reload all state into a new process and continue the same schedule.
"$MODEL365_PYTHON" -u scripts/robocasa365/train_single_gpu.py \
    --dataset "$DATASET365" --output "$RUN365_TRAIN/train" \
    --checkpoint "$ROOT/checkpoints/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt" \
    --action-steps 300 --joint-steps 100 --accumulation 4 \
    --resume "$RUN365_TRAIN/train/action_latest.pt"
