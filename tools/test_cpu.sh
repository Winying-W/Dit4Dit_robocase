#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES= NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=1
python -m pytest -q tests
for experiment in robocasa_official_recipe concat_random_head; do
  export PYTHONPATH="$PWD/experiments/$experiment/source"
  python -m pytest -q "experiments/$experiment/tests/test_optimizer.py" "experiments/$experiment/tests/test_recovery.py" \
    "experiments/$experiment/source/scripts/robocasa365/tests/test_distributed.py" \
    "experiments/$experiment/source/scripts/robocasa365/tests/test_multitask.py" \
    "experiments/$experiment/source/scripts/robocasa365/tests/test_prediction_metrics.py" \
    "experiments/$experiment/source/scripts/robocasa365/tests/test_replay_timing.py"
done
if [[ -n "${ROBOCASA_MANIFEST:-}" ]]; then
  python experiments/robocasa_official_recipe/tests/test_official_recipe.py
fi
if [[ -n "${GR1_CHECKPOINT:-}" ]]; then
  python experiments/concat_random_head/tests/test_head_init.py
fi
