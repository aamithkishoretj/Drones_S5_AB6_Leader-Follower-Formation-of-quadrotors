# Desktop flight controls

From the project folder in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -B scripts/run_ui.py
```

The panel is now a launcher and live control interface. Simulator windows stay
separate; Windows window reparenting and the unused embedded-viewer area were
removed. This avoids attaching to an unrelated existing simulator window.

- PyBullet and MuJoCo default to learned control with their saved models.
- Gazebo launches through WSL. Trail requests now travel to the WSL worker,
  where they use the active Gazebo transport partition.
- Kinematic is an ideal mathematical reference, not a motor/physics simulator.
  It uses analytic control and opens a separate 3D route plot; no model is needed.
- ArduPilot runs directly on Linux and through the WSL worker on Windows. Learned
  control uses the SITL-identified `models/ardupilot_dq_dynamics.npz` model; see
  `ardupilot/README.md` for installation requirements, `ARDUPILOT_PATH` and scope.

Speed, route radius, altitude and follower gap update during the same run.
Values ramp gradually to avoid target jumps. The panel reports the applied
values separately from the slider targets. Radius changes scale both dimensions
of the figure-eight. History retention supports increasing the follower gap
instead of discarding the needed past path. At startup only the initial segment
is available, until enough real leader history has accumulated.

The UI starts the trajectory at the selected altitude/radius, rather than
initializing at defaults and immediately jumping toward the slider settings.
Simulator/controller/route selection is locked while a flight runs. Stop first
to change these; sliders remain enabled.

MuJoCo still uses simplified world-force translation and attitude stabilization,
not a full rotor-driven flight model. It now tracks commanded yaw with bounded
torque and wrapped angle error; its response model was retrained. Its
body colors now match its trails (red leader, blue follower). Conservative UI
speed/gap ranges reduce abrupt or extreme configurations, but are not an
obstacle/collision avoidance system. Model validity at arbitrary new slider
combinations is not guaranteed.
