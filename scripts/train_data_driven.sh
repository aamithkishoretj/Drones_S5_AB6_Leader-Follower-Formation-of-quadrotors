#!/usr/bin/env bash
# Generate measured SITL data and fit the dynamics used by the default demos.
# Optional: bash scripts/train_data_driven.sh --ardupilot_path /existing/source
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
for TASK_SEED in 1 2 3; do
  .venv/bin/python -u scripts/collect_ardupilot_data.py --seed "$TASK_SEED" --duration 60 "$@"
done
.venv/bin/python scripts/train_ardupilot_model.py \
  --train results/data_driven/training_seed_1.npz results/data_driven/training_seed_2.npz \
  --validation results/data_driven/training_seed_3.npz
.venv/bin/python scripts/validate_ardupilot_model.py
