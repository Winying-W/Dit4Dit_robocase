#!/usr/bin/env bash
# Historical original16-demo step2000 comparison; never submit without prequeue evidence.
set -Eeuo pipefail
umask 002
ROOT=/file_system/vepfs/intern/haozhe.jia/projects/DiT4DiT
RUN365="$ROOT/runs/robocasa365_legacy_ab_20260924"
OUT365="$ROOT/artifacts/training/robocasa365_legacy_ab_20260924"
test -f "$RUN365/staged/action_step_002000.pt" || {
  echo 'Original16-demo StirVegetables step2000 has not been staged. Refusing to start the historical A/B.'
  exit 2
}
BACKUP365=/file_system/efs/checkpoint/intern/haozhe.jia/projects/DiT4DiT
export MODEL365_PYTHON="$BACKUP365/conda-env/bin/python"
export DIT365_CACHE="$ROOT/artifacts/runtime/cache"
export IMAGEIO_FFMPEG_EXE="$BACKUP365/conda-env/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
export TORCHINDUCTOR_CACHE_DIR="$DIT365_CACHE/torchinductor"
export NO_ALBUMENTATIONS_UPDATE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE MASTER_ADDR MASTER_PORT
for KEY365 in ${!TORCHELASTIC_@}; do unset "$KEY365"; done
cd "$RUN365/source"
source scripts/robocasa365/env.sh
exec 9>"$RUN365/run.lock"
flock -n 9 || { echo 'Another worker holds the legacy A/B run lock'; exit 2; }
exec > >(tee -a "$RUN365/cloud.log") 2>&1
trap 'code=$?; printf "failed code=%s line=%s\n" "$code" "$LINENO" > "$RUN365/stage.txt"; exit "$code"' ERR
GR1365="$ROOT/artifacts/checkpoints/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt"
BASE365="$RUN365/staged/action_step_002000.pt"
DATASET365="$ROOT/data/robocasa365/v1.0/target/composite/StirVegetables/20250814"
"$MODEL365_PYTHON" - "$RUN365" <<'PY'
import hashlib,json,pathlib,sys,torch
run=pathlib.Path(sys.argv[1])
def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()
report=json.loads((run/'prequeue_acceptance.json').read_text())
assert report['status']=='complete' and report['original_checkpoint_staged']
assert {'staged/relocation.json','staged/normalization.json','staged/split.json','source_hashes.json','task.yaml'} <= set(report['evidence_sha256'])
for name,sha in report['evidence_sha256'].items():assert digest(run/name)==sha,name
for name,sha in json.loads((run/'source_hashes.json').read_text()).items():assert digest(run/'source'/name)==sha,name
relocation=json.loads((run/'staged/relocation.json').read_text())
assert relocation['status']=='complete' and relocation['original_global_step']==2000
assert relocation['train_episodes']==16 and relocation['validation_episodes']==4
assert digest(run/'staged/action_step_002000.pt')==relocation['staged_sha256']
assert torch.cuda.device_count()==1,'Historical A/B uses one allocated GPU, sequential arms'
PY
mkdir -p "$OUT365"
nvidia-smi --query-gpu=name,memory.total --format=csv
stage365() { printf '%s\n' "$1" > "$RUN365/stage.txt"; }
verify_evaluation365() {
  "$MODEL365_PYTHON" - "$1" "$2" "$3" <<'PY'
import json,pathlib,sys
folder=pathlib.Path(sys.argv[1]);audit=json.loads(pathlib.Path(sys.argv[2]).read_text())
smoke=sys.argv[3]=='smoke';seeds=[99999] if smoke else list(range(100,110))
evaluation=json.loads((folder/'evaluation.json').read_text())
checkpoint=json.loads((folder/'checkpoint_verification.json').read_text())
assert evaluation['status']=='complete' and evaluation['completed_trials']==len(seeds)
assert evaluation['task']=='StirVegetables' and evaluation['initialization']=='fresh_gym_target'
assert evaluation['official_horizon']==2400 and evaluation['horizon']==(8 if smoke else 2400)
assert [row['seed'] for row in evaluation['episodes']]==seeds
assert audit['passed'] and audit['completed_trials']==len(seeds)
assert checkpoint['global_step']==(12 if smoke else 1000)
PY
}
for ARM365 in frozen partial_joint; do
  mkdir -p "$OUT365/$ARM365" "$RUN365/ab/$ARM365/live"
  export ROBO365_STATUS_MIRROR="$RUN365/ab/$ARM365/live"
  if [ "$ARM365" = frozen ]; then
    ACTION365=1000; JOINT365=0; PHASE365=action
  else
    ACTION365=0; JOINT365=1000; PHASE365=partial_joint
  fi
  TRAIN365=("$MODEL365_PYTHON" -u scripts/robocasa365/train_single_gpu.py
    --dataset "$DATASET365" --checkpoint "$GR1365" --output "$OUT365/$ARM365"
    --action-steps "$ACTION365" --joint-steps "$JOINT365" --accumulation 4 --seed 42
    --snapshot-steps 10 12 1000)
  latest_step365() {
    "$MODEL365_PYTHON" -c 'import pathlib,sys,torch; p=pathlib.Path(sys.argv[1]); print(torch.load(p,map_location="cpu",weights_only=False,mmap=True)["global_step"] if p.exists() else 0)' "$OUT365/$ARM365/${PHASE365}_latest.pt"
  }
  STEP365=$(latest_step365)
  if [ "$STEP365" -eq 0 ]; then
    stage365 "${ARM365}_10_update_gate"
    timeout --signal=TERM --kill-after=60s 1200s "${TRAIN365[@]}" --warm-start "$BASE365" --stop-after 10
  fi
  STEP365=$(latest_step365)
  if [ "$STEP365" -lt 12 ]; then
    stage365 "${ARM365}_independent_resume_gate"
    timeout --signal=TERM --kill-after=60s 1200s "${TRAIN365[@]}" --resume "$OUT365/$ARM365/${PHASE365}_latest.pt" --stop-after 12
  fi
  # Resumption cannot bypass the actual model-to-simulator short gate.
  if [ ! -f "$RUN365/ab/$ARM365/smoke_action_audit.json" ]; then
    test ! -e "$RUN365/ab/$ARM365/smoke" || { echo 'Preserve the incomplete smoke attempt and explicitly prepare recovery'; exit 2; }
    stage365 "${ARM365}_policy_simulator_gate"
    timeout --signal=TERM --kill-after=60s 1200s "$MODEL365_PYTHON" -u scripts/robocasa365/evaluate_policy.py \
      --checkpoint "$OUT365/$ARM365/${PHASE365}_step_000012.pt" --dataset "$DATASET365" \
      --output "$RUN365/ab/$ARM365/smoke" --sim-python "$SIM365_PYTHON" --seeds 99999 --max-steps 8
    "$SIM365_PYTHON" scripts/robocasa365/audit_saved_actions.py \
      --phase "$RUN365/ab/$ARM365/smoke" --normalization "$OUT365/$ARM365/normalization.json" \
      --output "$RUN365/ab/$ARM365/smoke_action_audit.json"
  fi
  verify_evaluation365 "$RUN365/ab/$ARM365/smoke" "$RUN365/ab/$ARM365/smoke_action_audit.json" smoke
  STEP365=$(latest_step365)
  if [ "$STEP365" -lt 1000 ]; then
    stage365 "train_$ARM365"
    "${TRAIN365[@]}" --resume "$OUT365/$ARM365/${PHASE365}_latest.pt"
  fi
  stage365 "eval_$ARM365"
  if [ ! -f "$RUN365/ab/$ARM365/action_audit.json" ]; then
    test ! -e "$RUN365/ab/$ARM365/eval" || { echo 'Preserve the incomplete evaluation attempt and explicitly prepare recovery'; exit 2; }
    "$MODEL365_PYTHON" -u scripts/robocasa365/evaluate_policy.py \
      --checkpoint "$OUT365/$ARM365/${PHASE365}_step_001000.pt" --dataset "$DATASET365" \
      --sim-python "$SIM365_PYTHON" --output "$RUN365/ab/$ARM365/eval" \
      --seeds 100 101 102 103 104 105 106 107 108 109 --execute-horizon 8
    "$SIM365_PYTHON" scripts/robocasa365/audit_saved_actions.py \
      --phase "$RUN365/ab/$ARM365/eval" --normalization "$OUT365/$ARM365/normalization.json" \
      --output "$RUN365/ab/$ARM365/action_audit.json"
  fi
  verify_evaluation365 "$RUN365/ab/$ARM365/eval" "$RUN365/ab/$ARM365/action_audit.json" formal
done
stage365 compare_paired_arms
"$SIM365_PYTHON" scripts/robocasa365/select_ab.py --run "$RUN365" --train-root "$OUT365"
stage365 complete
