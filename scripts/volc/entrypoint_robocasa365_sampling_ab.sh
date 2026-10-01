#!/usr/bin/env bash
set -Eeuo pipefail
umask 000
if [ ! -e /file_system ]; then ln -s / /file_system; fi
ROOT=/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT
RUN365="$ROOT/runs/robocasa365_composite_sampling_ab_20260928"
ARM365="${1:?Specify paired_uniform or initial_switch}"
case "$ARM365" in paired_uniform|initial_switch) ;; *) exit 2 ;; esac
ARM_RUN365="$RUN365/$ARM365"
OUT365="$ROOT/artifacts/training/robocasa365_composite_sampling_ab_20260928/$ARM365"
BACKUP365=/file_system/efs/checkpoint/intern/haozhe.jia/projects/DiT4DiT
export MODEL365_PYTHON="$BACKUP365/conda-env/bin/python"
export DIT365_CACHE="$ROOT/artifacts/runtime/cache"
export IMAGEIO_FFMPEG_EXE="$BACKUP365/conda-env/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
export TORCHINDUCTOR_CACHE_DIR="$DIT365_CACHE/torchinductor"
export PYTHONHASHSEED=0 NO_ALBUMENTATIONS_UPDATE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE MASTER_ADDR MASTER_PORT
for KEY365 in ${!TORCHELASTIC_@}; do unset "$KEY365"; done
cd "$RUN365/cloud_source"
source scripts/robocasa365/env.sh
mkdir -p "$OUT365" "$ARM_RUN365/train_live"
chmod 777 "$OUT365" "$ARM_RUN365/train_live"
export ROBO365_STATUS_MIRROR="$ARM_RUN365/train_live"
exec 9>"$ARM_RUN365/run.lock"
flock -n 9 || { echo 'Existing sampling worker holds the branch lock'; exit 2; }
exec > >(tee -a "$ARM_RUN365/cloud.log") 2>&1
trap 'code=$?; printf "failed code=%s line=%s\n" "$code" "$LINENO" > "$ARM_RUN365/stage.txt"; exit "$code"' ERR
"$MODEL365_PYTHON" "$RUN365/verify_candidate.py" --branch "$ARM365" --gpu-startup
GR1365="$ROOT/artifacts/checkpoints/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt"
WARM365="$ROOT/artifacts/training/robocasa365_composite16_atomicinit_4gpu_20260927/action_step_010000.pt"
WARM_RUN365="$ROOT/runs/robocasa365_composite16_atomicinit_4gpu_20260927"
AUDIT365="$ROOT/runs/robocasa365_composite_backup_20260927/live/checkpoint_010000.json"
MANIFEST365="$WARM_RUN365/prepared/manifest.json"
POOLS365="$ROOT/runs/robocasa365_composite_phase_sampling_20260928/pools/manifest.json"
DATA365=$("$MODEL365_PYTHON" -c 'import json,sys; print(next(row["path"] for row in json.load(open(sys.argv[1]))["tasks"] if row["task"]=="GetToastedBread"))' "$MANIFEST365")
TRAIN365=("$MODEL365_PYTHON" -u -m scripts.robocasa365.train_single_gpu
  --checkpoint "$GR1365" --manifest "$MANIFEST365" --output "$OUT365"
  --action-steps 2000 --joint-steps 0 --microbatch-size 4 --accumulation 16
  --warmup-steps 200 --validation-interval 200 --checkpoint-interval 200
  --snapshot-steps 10 12 1000 2000 --seed 42 --paired-ab
  --sampling-mode "$ARM365" --phase-pools "$POOLS365")
latest_step365() {
  "$MODEL365_PYTHON" -c 'import pathlib,sys,torch; p=pathlib.Path(sys.argv[1])/"action_latest.pt"; print(torch.load(p,map_location="cpu",mmap=True,weights_only=False)["global_step"] if p.exists() else 0)' "$OUT365"
}
stage365() { printf '%s\n' "$1" > "$ARM_RUN365/stage.txt"; }
STEP365=$(latest_step365)
if [ "$STEP365" -eq 0 ]; then
  stage365 cuda_first10_updates
  timeout --signal=TERM --kill-after=60s 1800s "${TRAIN365[@]}" --stop-after 10 \
    --warm-start "$WARM365" --warm-start-run "$WARM_RUN365" --warm-start-task-set composite_seen \
    --warm-start-step 10000 --warm-start-audit "$AUDIT365"
fi
STEP365=$(latest_step365)
if [ "$STEP365" -lt 12 ]; then
  stage365 cuda_independent_resume_to12
  timeout --signal=TERM --kill-after=60s 1800s "${TRAIN365[@]}" --resume "$OUT365/action_latest.pt" --stop-after 12
fi
if [ ! -f "$ARM_RUN365/cuda_checkpoint_12.json" ]; then
  "$MODEL365_PYTHON" -m scripts.robocasa365.audit_sampling_checkpoint --run "$RUN365" --mode "$ARM365" \
    --checkpoint "$OUT365/action_step_000012.pt" --step 12 --output "$ARM_RUN365/cuda_checkpoint_12.json"
fi
if [ ! -f "$ARM_RUN365/smoke_action_audit.json" ]; then
  test ! -e "$ARM_RUN365/smoke" || { echo 'Preserve and inspect the incomplete smoke attempt before recovery'; exit 2; }
  stage365 policy_simulator_smoke
  timeout --signal=TERM --kill-after=60s 1800s "$MODEL365_PYTHON" -u -m scripts.robocasa365.evaluate_policy \
    --checkpoint "$OUT365/action_step_000012.pt" \
    --dataset "$DATA365" \
    --task GetToastedBread --output "$ARM_RUN365/smoke" --sim-python "$SIM365_PYTHON" \
    --seeds 99999 --max-steps 8 --execute-horizon 8 --scene-protocol stable_counter_v1
  "$SIM365_PYTHON" -m scripts.robocasa365.audit_saved_actions --phase "$ARM_RUN365/smoke" \
    --normalization "$OUT365/normalization.json" --output "$ARM_RUN365/smoke_action_audit.json"
fi
"$MODEL365_PYTHON" "$RUN365/verify_candidate.py" --branch "$ARM365" --cuda-gates
STEP365=$(latest_step365)
if [ "$STEP365" -lt 2000 ]; then
  stage365 training_to2000
  "${TRAIN365[@]}" --resume "$OUT365/action_latest.pt"
fi
stage365 checkpoint2000_audit
"$MODEL365_PYTHON" -m scripts.robocasa365.audit_sampling_checkpoint --run "$RUN365" --mode "$ARM365" \
  --checkpoint "$OUT365/action_step_002000.pt" --step 2000 --output "$ARM_RUN365/checkpoint_2000.json"
stage365 development_three_tasks_60_trials
"$MODEL365_PYTHON" -u -m scripts.robocasa365.evaluate_multitask \
  --checkpoint "$OUT365/action_step_002000.pt" --manifest "$MANIFEST365" \
  --output "$ARM_RUN365/development_2000" --sim-python "$SIM365_PYTHON" --workers 1 \
  --development --allow-task-subset --tasks GetToastedBread DeliverStraw StirVegetables \
  --scene-protocol stable_counter_v1
if [ ! -f "$ARM_RUN365/development_2000_review.json" ]; then
  "$MODEL365_PYTHON" -m scripts.robocasa365.review_evaluation --evaluation "$ARM_RUN365/development_2000" \
    --manifest "$MANIFEST365" --step 2000 --purpose development \
    --tasks GetToastedBread DeliverStraw StirVegetables --output "$ARM_RUN365/development_2000_review.json"
fi
"$MODEL365_PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="complete" and r["completed_trials"]==r["planned_trials"]==60 and r["task_scope"]=="declared_task_subset" and r["saved_action_chain_passed"]' "$ARM_RUN365/development_2000_review.json"
stage365 complete
