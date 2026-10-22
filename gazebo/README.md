# Gazebo backend

This project targets **Gazebo Sim Harmonic with ROS 2 Jazzy** for the first
Gazebo backend. Jazzy's supported/recommended Gazebo release is Harmonic.

## Required system software

Install Gazebo Harmonic and ROS 2 Jazzy on a supported Linux system, then make
sure the following commands are on your PATH:

```bash
gz --version
ros2 --version
```

Source ROS 2 before running the project:

```bash
source /opt/ros/jazzy/setup.bash
```

Install the ROS-Gazebo integration and actuator message package:

```bash
sudo apt install ros-jazzy-ros-gz ros-jazzy-actuator-msgs
```

The Python process also needs `rclpy`, `geometry_msgs`, and the normal ROS 2
Python runtime from the sourced Jazzy installation.

## Windows desktop UI

When the project UI runs with Windows Python, the Gazebo backend starts its
simulation worker inside WSL and relays states and motor commands to the UI.
This avoids trying to import Linux ROS packages into Windows Python. The
default distro is `Ubuntu-24.04`; enable WSLg for the native Gazebo window and
make sure the project drive is mounted in WSL (the standard `/mnt/<drive>`
mount is used). The UI worker sources `/opt/ros/jazzy/setup.bash` automatically.

To use a differently named distro or ROS installation, set these Windows
environment variables before launching the UI:

```powershell
$env:GAZEBO_WSL_DISTRO = "Your-Ubuntu-Distro"
$env:GAZEBO_ROS_SETUP = "/opt/ros/jazzy/setup.bash"
$env:GAZEBO_WSL_MOUNT_ROOT = "/mnt"
```

Install the system dependencies inside that distro:

```bash
sudo apt update
sudo apt install ros-jazzy-ros-gz ros-jazzy-actuator-msgs python3-numpy python3-yaml
```

The Gazebo viewer opens as a native WSLg window. CLI runs started inside WSL
should source ROS first as shown below; the Windows UI worker does this itself.

## How the backend works

The normal formation loop stays in Python:

```text
dq_control.KinematicController
        |
        v
simulators/leader_follower.py
        |
        v
simulators/gazebo.py
        |
        +--> geometry_msgs/Pose  <--- PosePublisher <--- Gazebo
        |
        +--> actuator_msgs/Actuators ---> ros_gz_bridge ---> MulticopterMotorModel
```

The project generates a temporary Gazebo world for each run so the analytic
trajectory's requested t=0 pose can be used without editing a permanent world
file. Two drone model copies are also generated at runtime so each model has a
unique motor-command namespace.

The Gazebo backend intentionally does **not** depend on gym-pybullet-drones.
Its low-level controller is `simulators/low_level.py`, which converts the
formation layer's position/velocity/attitude-rate target into four rotor RPMs.

## Run

From the project root:

```bash
source /opt/ros/jazzy/setup.bash

python scripts/run_experiment.py \
  --simulator gazebo \
  --experiment real_eig \
  --trajectory lemniscate \
  --duration 30 \
  --ctrl_freq 48 \
  --gui
```

Headless:

```bash
python scripts/run_experiment.py \
  --simulator gazebo \
  --experiment real_eig \
  --trajectory lemniscate \
  --duration 10
```

The results are written using the same `.npz` format as the original
PyBullet simulation, so existing plotting/metrics tools continue to work.

## Notes

The Gazebo quadrotor is a lightweight project-owned model, not a flight-stack
or vehicle-specific reproduction. The first goal is simulator integration and
a real physics loop. Controller tuning should be expected before claiming
numerical equivalence with the PyBullet/CF2X experiment.

Gazebo Classic is intentionally not targeted; the project uses modern Gazebo
Sim and `ros_gz`.
