#!/usr/bin/env bash
# One-time Ubuntu / WSL2 setup. Run interactively so sudo can request a password.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PARENT="$(dirname "$TASK_ROOT")"
TASK_PYTHON="${PYTHON:-python3}"
TASK_JOBS="${BUILD_JOBS:-2}"

source /etc/os-release
if [[ "${ID:-}" != ubuntu || ( "${VERSION_ID:-}" != 22.04 && "${VERSION_ID:-}" != 24.04 ) ]]; then
  echo "This installer targets Ubuntu 22.04/24.04 or WSL2 Ubuntu. See ardupilot/README.md." >&2
  exit 1
fi
sudo apt-get update
sudo apt-get install -y curl gnupg lsb-release git build-essential cmake pkg-config python3-venv \
  python3-dev ccache gawk libtool libxml2-dev libxslt1-dev
curl -fsSL https://packages.osrfoundation.org/gazebo.gpg -o /tmp/formation-gazebo.gpg
sudo install -m 644 /tmp/formation-gazebo.gpg /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" |
  sudo tee /etc/apt/sources.list.d/gazebo-stable.list >/dev/null
sudo apt-get update
sudo apt-get install -y gz-harmonic libgz-sim8-dev rapidjson-dev libopencv-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl

cd "$TASK_ROOT"
"$TASK_PYTHON" -m venv .venv
.venv/bin/python -m pip install -r requirements-ardupilot.txt
.venv/bin/python -m pip install future 'empy==3.3.4' pexpect dronecan pyserial psutil setuptools wheel
.venv/bin/python scripts/setup_simulators.py --simulator ardupilot

git -C "$TASK_PARENT/ardupilot" submodule update --init --recursive --depth 1
# Build native SITL with this project's virtual environment. No ARM firmware
# toolchain or ROS installation is required for the desktop demonstration.
(
  cd "$TASK_PARENT/ardupilot"
  export PATH="$TASK_ROOT/.venv/bin:$PATH"
  ./waf configure --board sitl
  ./waf copter -j "$TASK_JOBS"
)
GZ_VERSION=harmonic cmake -S "$TASK_PARENT/ardupilot_gazebo" -B "$TASK_PARENT/ardupilot_gazebo/build" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$TASK_PARENT/ardupilot_gazebo/build" -j "$TASK_JOBS"
.venv/bin/python scripts/check_ardupilot.py
echo "Setup complete. Run: bash scripts/run_ardupilot.sh 60"
