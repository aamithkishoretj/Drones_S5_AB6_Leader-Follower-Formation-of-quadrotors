# ArduPilot in the desktop panel

The SITL backend and world assets were imported from `origin/ardupilot-gazebo`.
The desktop panel now lists `ardupilot` and transports commands and trails to
the WSL worker. It uses the analytic dual-quaternion controller. The separate
branch's `ardupilot_dq_dynamics.npz` uses a different model implementation and
must not be renamed to a response model for this branch.

Required in Ubuntu-24.04:

- Built ArduCopter at `../ardupilot/build/sitl/bin/arducopter` and its default parameter files.
- Built official Gazebo plugin under `../ardupilot_gazebo/build/`, plus its Iris model resources.
- Gazebo Harmonic and `pymavlink` importable by the worker's Python 3.

These dependencies were missing at the expected paths during this update.
Selecting Start reports missing items; it does not install or build them.
The original branch contains installation instructions and a setup script,
but that script creates `.venv` for Linux. Do not run it over this checkout's
Windows `.venv`. Use a separate Linux checkout for that installation workflow.

The panel supplies a larger Iris trajectory preset and gap. Stop requests let
the flight stack land before shutting down. Startup can take up to 120 seconds;
the UI remains responsive while waiting. This integration has not been flight
validated here because the SITL/plugin dependencies are absent.
