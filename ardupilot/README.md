# ArduPilot SITL + Gazebo live leader–follower demonstration

Run two Iris quadrotors in **Gazebo Harmonic**, each controlled by its own
**ArduCopter SITL**. A data-driven dual-quaternion predictive controller now
commands velocity and yaw rate over MAVLink in Guided mode. Its flight-stack
response model is learned from generated simulator flights, and its learned
predictions select both drones' commands inside the actual flight loop.
See [training and validation details](../docs/data_driven.md).
ArduCopter handles
the attitude, thrust and motors. **ROS 2 is not required for this backend.**

## One-time installation

Use Ubuntu 22.04/24.04, or WSL2 Ubuntu with WSLg for the Gazebo window. Open
**this project folder** in VS Code. In its integrated Bash terminal run:

```bash
bash scripts/setup_ardupilot.sh
```

The script requests your sudo password for system packages, installs Gazebo
Harmonic and native build tools, creates `.venv`, installs Python packages,
and builds SITL and the official Gazebo plugin. Source repositories stay
beside this project, not inside it:

```text
parent/
  S5-Drones/                # this project; folder name may differ
  ardupilot/               # native executable build/sitl/bin/arducopter
  ardupilot_gazebo/         # plugin build + upstream Iris model resources
```

New source clones use `Copter-4.5.7` and plugin commit
`082a0fe231f6e63bc8d1598f1cba461d9e2ea7f5`. Existing source checkouts are
preserved; a different version may need integration adjustments. Allow time
and disk space for downloads and compilation. `BUILD_JOBS=4 bash
scripts/setup_ardupilot.sh` changes build parallelism (default 2).

If dependencies are already installed, use:

```bash
source .venv/bin/activate
python scripts/check_ardupilot.py
```

For an existing installation elsewhere, pass `--ardupilot_path /path/to/ardupilot`
and `--ardupilot_gazebo_path /path/to/ardupilot_gazebo` to both the checker and
the experiment command. These are the source/build roots, not the binary paths.

## Show the professor

```bash
bash scripts/run_ardupilot.sh 60
```

The launcher now defaults to a lighter desktop profile: Ogre rendering,
24 Hz control, 400 Hz physics, no shadows/anti-aliasing, a smaller window and
a closer camera. Its Iris lemniscate preset uses half-width 2.55 m,
half-height 1.95 m, altitude 2 m, a 60-second period and a 1.5 m following gap.
The larger path and gap reduce close approaches around tight turns; the
0.85 / 0.65 m paper path was designed for smaller vehicles. Initial EKF/GPS readiness may take about a minute on a slow
system; the terminal reports the stage and then prints flight progress.
Five-second headless and GUI runs have both completed on this machine,
including takeoff, tracking, landing, disarming, result saving and process cleanup.

This opens Gazebo, starts two SITLs, waits for normal arming checks, takes off,
flies the figure-eight for **60 seconds of simulation time**, then lands and
stops the processes. Takeoff and landing are additional time. Change `60` to
your required duration. Red lines show the leader's flown route; blue lines
show the follower's flown route. Select `leader` / `follower` in Gazebo's entity
tree and adjust the camera if needed. You can record the window for your talk.

VS Code also provides **Terminal → Run Task → ArduPilot: live leader-follower
demo**, with a duration prompt. Select `.venv/bin/python` as the Python interpreter.

Full command using the launcher's lighter settings:

```bash
source .venv/bin/activate
python scripts/run_experiment.py --simulator ardupilot \
  --experiment real_eig --trajectory lemniscate \
  --r_x 2.55 --r_y 1.95 --w_d 0.10471975511965977 --z0 2 \
  --follow_distance 1.5 --duration 60 --gui \
  --ardupilot_low_resource --ctrl_freq 24 --gazebo_render_engine ogre
```

For a first end-to-end smoke run without a window:

```bash
python scripts/run_experiment.py --simulator ardupilot \
  --experiment real_eig --trajectory lemniscate --duration 5
```

The ArduPilot launcher has a 60-second period, so 60 seconds shows one full
figure-eight, while 15 seconds shows only one quarter. The generic trajectory
class still has the original paper's 30-second period. Plots label the
recording duration and reference period to make partial runs clear.
The follower traces the leader's measured chronological route, 1.5 m behind
along that route. This is not a collision-avoidance controller at crossings.
The larger Iris vehicle and flight-stack response differ from the CF2X and
project-owned Gazebo vehicle; tracking accuracy must be assessed in live runs.
Tracking is stopped if measured separation falls below 0.65 m, followed by
the normal landing/cleanup attempt. `--ardupilot_min_separation` changes this
threshold. This check is not predictive collision avoidance and does not
guarantee separation during landing.

The original analytical-controller 60-second Iris preset was validated headlessly on 4 October 2026:
leader position RMS error 0.034 m, follower RMS error 0.073 m, and minimum
measured drone separation 1.04 m. Both drones landed and disarmed. The saved
run and plots are in `results/ardupilot_validation/` for this checkout.
These are measurements from that run, not guarantees for different paths,
gains, spacing or system conditions.

### Potato-chip demonstration

```bash
bash scripts/run_ardupilot_potato_chip.sh 60
```

VS Code provides **Terminal → Run Task → ArduPilot: potato-chip demo**.
This uses the same lighter Gazebo/ArduPilot settings and measured-route follower,
with a 2.55 m circular footprint radius, a 60-second revolution, a 1.5 m
following gap, and reference altitude varying from 1.5 to 2.5 m twice per
revolution (`z = 2 + 0.5 cos(2 theta)`). Change `60` to the required tracking
duration; takeoff and landing are extra. A shorter run shows part of the curve.

The original analytical-controller 60-second preset was validated headlessly on 4 October 2026: leader
position RMS error 0.027 m, follower RMS error 0.051 m, minimum measured
separation 1.23 m. Both drones landed and disarmed. The run is saved as
`results/ardupilot_validation/ardupilot_real_eig_potato_chip_20261004_162130.npz`.
These metrics describe this run, not a guarantee for other settings.

The plotting script saves an additional `_trajectories_3d.png` for potato-chip
runs, alongside the XY and axis/error plots. Use the 3D plot to show the saddle
shape: its XY projection is a circle. For example:

```bash
.venv/bin/python scripts/plot_results.py --run \
  results/ardupilot_validation/ardupilot_real_eig_potato_chip_20261004_162130.npz
```

Other reference paths are available using `--trajectory potato_chip`, `bspline`
or `interpolated`. `--no-trails` disables drawing. Press Ctrl+C in the terminal
to request landing and cleanup. If landing cannot finish within
`--ardupilot_land_timeout` (default 60 wall seconds), the simulator processes
are terminated and a warning is printed. This integration launches local
simulation connections only.

## Results and diagnostics

Tracking logs use the existing `results/*.npz` format and plotting script.
Runtime worlds, vehicle parameter/storage files, Gazebo output and both SITL
outputs stay under `results/ardupilot_runtime/<run-id>/`. Review these logs if
startup fails. The backend only terminates processes it launched.

- Missing `gz` or plugin library: finish system installation and plugin build.
- Missing `arducopter`: complete `./waf configure --board sitl` and `./waf copter`
  in the ArduPilot checkout, using the setup script's virtual environment.
- Arming timeout: read the printed pre-arm messages and SITL logs. Normal
  arming checks remain enabled; RC-loss failsafe is disabled for this demo,
  while Python sends a GCS heartbeat every second.
- Slow startup: increase `--ardupilot_startup_timeout 180`.
- Graphics errors: try `--gazebo_render_engine ogre`, or the headless command.
- Whole desktop lag: check the renderer in `~/.gz/rendering/ogre.log` (or
  `ogre2.log`). `llvmpipe` means CPU rendering. On this machine, the current
  boot included `nomodeset`, no GPU driver was active, and Ubuntu recommended
  `nvidia-driver-595`. Install the recommended driver through **Software &
  Updates → Additional Drivers** or `sudo ubuntu-drivers install`, and restart
  using the normal Ubuntu boot entry without adding `nomodeset`. Secure Boot
  is enabled, so complete MOK enrollment if the installer requests it. Save
  your work before restarting. Recheck graphics after boot; these steps require
  the user's sudo password and a reboot and have not been performed by Codex.
  See [Ubuntu's driver guide](https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/)
  and [the kernel's nomodeset documentation](https://www.kernel.org/doc/html/v6.6/admin-guide/kernel-parameters.html).
- Port conflict: stop another demo using TCP 5760/5770 or UDP 9002/9012.
- Telemetry/simulation stall: the run exits instead of advancing the reference
  trajectory indefinitely. Deliberately pausing Gazebo for more than 3 wall
  seconds also ends the run.

## Integration details and validation limits

Each run generates two copies of the upstream Iris model and changes its
ArduPilot JSON input port to 9002 / 9012. Lockstep is disabled for the shared
multi-vehicle world. Standard physics uses a 1 ms step; the lighter profile
uses 2.5 ms. Only the IMU system is loaded on the server; the Iris model has
no camera requiring a server render engine. Gazebo targets real-time factor 1, and the tracking timer
uses leader SITL timestamps; slow rendering may increase elapsed wall time.
Both vehicles must advance their telemetry clock before the next control step.

MAVLink local position and velocity are converted from NED to the project's
ENU world. Attitude converts both NED/ENU and body FRD/FLU conventions. Each
vehicle's local position origin is calibrated before takeoff. The backend
passes bounded velocity and Euler yaw rate through `SET_POSITION_TARGET_LOCAL_NED`;
the project's body angular velocity is converted to yaw rate using its current
attitude. ArduPilot controls roll/pitch, so this does not reproduce a full
three-axis angular command or the original low-level PID. Existing controller
equations, gain presets, route following and logging remain in use.

Automated tests verify frame conversion, actual MAVLink encoding and routing,
independent origins/ports, simulated-time duration, and cleanup without Gazebo.
These checks **do not certify flight stability or successful live takeoff**.
Run the headless smoke command and then the GUI demonstration on the installed
system before presenting.

References: [ArduPilot Gazebo integration](https://ardupilot.org/dev/docs/sitl-with-gazebo.html),
[Guided mode commands](https://ardupilot.org/dev/docs/copter-commands-in-guided-mode.html),
[official plugin](https://github.com/ArduPilot/ardupilot_gazebo),
[Gazebo Harmonic installation](https://gazebosim.org/docs/harmonic/install_ubuntu/).
