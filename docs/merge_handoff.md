# UI branch handoff

This branch is based on `UI` and contains the current desktop-control fixes.

- Compact launcher with separate native viewers and live applied-value feedback.
- Live path-history retention for follower-gap changes and proportional route resizing.
- Gazebo trail requests forwarded through the Windows-to-WSL worker.
- Kinematic reference plot, using analytic control without a missing-model error.
- MuJoCo menus hidden, cleaner camera/shadow settings, bounded commanded yaw tracking.
- Retrained `models/mujoco_response.npz`, with recordings and reports in `data/yaw_tracking/mujoco`.

## ArduPilot merge notes

`simulators/ardupilot.py`, `ardupilot/world.sdf`, and `ardupilot/gui.config`
were taken from `origin/ardupilot-gazebo` and connected to the UI's WSL worker.
The backend's missing-dependency error text was adjusted for this checkout.
This machine does not have the required built SITL/plugin/MAVLink setup at the
expected paths, so this integration has not been flight validated here.

The UI currently selects analytic control for ArduPilot. The other branch's
`ardupilot_dq_dynamics.npz` is not compatible with this branch's `LearnedResponse`
model format. Merge the matching controller/model code together; do not rename
one model to bypass compatibility checks.

Likely merge overlaps: `simulators/base.py`, `simulators/__init__.py`,
`simulators/leader_follower.py`, `simulators/ardupilot.py`, and the UI/worker code.
Keep the updated ArduPilot implementation from its owner while preserving the
UI live-control and WSL protocol hooks. A worker reset can select `gazebo` or
`ardupilot`, and `draw_trails` is an explicit protocol operation.

Validation: the Python suite passed before handoff; two 60-second MuJoCo live
parameter checks preceded the yaw update, and the retraining pipeline evaluated
the yaw update on separate figure-eight and B-spline flights. Final GUI motion
and appearance still need a user check. Gazebo GUI-enabled runs completed, but
trail visibility was not visually confirmed.

Launch after merging from the project root:

```powershell
.\.venv\Scripts\python.exe -B scripts/run_ui.py
```
