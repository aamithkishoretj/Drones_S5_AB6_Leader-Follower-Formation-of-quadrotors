#!/usr/bin/env bash
# Usage: bash scripts/run_ardupilot_potato_chip.sh 60 [extra experiment arguments]
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_DURATION="${1:-60}"
if (( $# )); then shift; fi
if [[ ! -x "$TASK_ROOT/.venv/bin/python" ]]; then
  echo "Run bash scripts/setup_ardupilot.sh first." >&2
  exit 1
fi
cd "$TASK_ROOT"
exec .venv/bin/python scripts/run_experiment.py --simulator ardupilot \
  --experiment real_eig --controller data_driven --trajectory potato_chip --duration "$TASK_DURATION" --gui \
  --ardupilot_low_resource --ctrl_freq 24 --gazebo_render_engine ogre \
  --chip_r 2.55 --chip_w 0.10471975511965977 --chip_z0 2 --chip_z_amp 0.5 --chip_k 2 \
  --follow_distance 1.5 "$@"
