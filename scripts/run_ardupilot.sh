#!/usr/bin/env bash
# Usage: bash scripts/run_ardupilot.sh 60 [extra run_experiment arguments]
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
  --experiment real_eig --trajectory lemniscate --duration "$TASK_DURATION" --gui \
  --ardupilot_low_resource --ctrl_freq 24 --gazebo_render_engine ogre \
  --r_x 2.55 --r_y 1.95 --w_d 0.10471975511965977 --z0 2 \
  --follow_distance 1.5 "$@"
