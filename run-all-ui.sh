#!/usr/bin/env bash
# Open the shared simulator panel from any working directory.
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_SETUP="${GAZEBO_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
if [[ -f "$ROS_SETUP" ]]; then
    source "$ROS_SETUP"
fi
cd "$PROJECT_ROOT"
if [[ ! -x .venv/bin/python ]]; then
    echo "Missing Linux .venv. See ardupilot/README.md for setup." >&2
    exit 1
fi
exec .venv/bin/python scripts/run_ui.py
