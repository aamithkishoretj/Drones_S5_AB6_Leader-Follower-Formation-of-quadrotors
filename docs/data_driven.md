# Simulator-trained dual-quaternion predictive control

The command-line runner now defaults to `--controller learned`. This replaces
the original high-level gain-based control law during flight. The analytic
controller remains available for collecting excitation flights and as a baseline.
The Python `SimConfig` API retains its analytic default for compatibility;
set `controller="learned"` when using that API directly.

## What changed

| Before | Now |
| --- | --- |
| Fixed analytic gain law selects high-level commands | A fitted simulator-response model predicts motion; a bounded optimizer selects commands |
| No training stage needed | Collect actual transitions, fit a backend-specific model, validate, then fly |
| Dual-quaternion pose and error | Preserved and used in the predictive control objective |
| Course topics mostly separate demonstrations | System identification, multivariate linear approximation, prediction and constrained optimization operate inside the main workflow |
| Recorded leader path and visible trails | Preserved: follower tracks past leader poses about 0.8 m along the route behind; red/blue trails show actual motion |

This is identification-based data-driven high-level control. It is not
reinforcement learning, a neural network, or a fully model-free flight stack.
Simulator physics, low-level stabilization and known rigid-body kinematics remain.

## Training and control

1. Run four seeded excitation flights on varied figure-eight and potato-chip
   references. Record both drones' actual current state, applied command and
   actual next state. The analytic controller supplies exploratory flights;
   its output is not the regression target.
2. Fit a standardized ridge-regression response model
   `x_next = A x + B u + c`. State contains body linear velocity, body angular
   velocity, and two components of gravity direction. Input contains body
   velocity and angular-rate commands. No physical parameters enter this fit.
3. Reserve a whole fifth flight for prediction validation. Compare against
   persistence (`x_next = x`), rather than randomly splitting adjacent samples.
4. In flight, calculate relative pose using `Q_current.conj() * Q_desired`.
   Its translation and shortest rotation vector supply the tracking error.
   Roll the learned dynamics forward 24 control steps (0.5 s), integrate predicted
   twist locally, and solve a bounded quadratic least-squares problem.
   The six command variables are held constant over this prediction horizon;
   replan after each observed step. SciPy solves this convex problem, not CVXPY.
5. Compare learned and analytic controllers on two additional evaluation
   trajectories. Save full logs, model hashes, training provenance and metrics.

The model is a local linear response approximation, not an exact nonlinear
dual-quaternion dynamics model. It has no formal closed-loop stability guarantee.
Command bounds limit extrapolation but are not a collision-avoidance guarantee.
There is no silent fallback to analytic control if a model is missing or invalid.
Gazebo additionally caps roll/pitch rate commands at 0.15 rad/s because its
low-level controller already computes tilt from translation commands. This
constraint was selected after tracking failures with the original wider bounds;
the reported Gazebo evaluation is therefore validation used during development,
not a completely untouched test. Yaw keeps its separate learned bound.

## Run in Windows PowerShell

```powershell
cd "D:\drones\Dual-Quaternion-Based-Control-for-a-Leader-Follower-Formation-of-Two-Quadrotors"
# Needed once, or after changing backend dynamics:
.venv\Scripts\python.exe -B scripts/train_data_driven.py --simulator pybullet
# Uses the saved trained model:
.venv\Scripts\python.exe -B scripts/run_experiment.py --simulator pybullet --controller learned --trajectory lemniscate --follow_distance 0.8 --duration 60 --gui
```

For MuJoCo, replace `pybullet` with `mujoco` in both commands. Training is
headless; the experiment's `--gui` displays the flight and trails.
Models were generated locally for these backends during this update, so
retraining is not required merely to repeat the same experiment.

## Gazebo in Ubuntu / WSL

```bash
cd /mnt/d/drones/Dual-Quaternion-Based-Control-for-a-Leader-Follower-Formation-of-Two-Quadrotors
source /opt/ros/jazzy/setup.bash
source .venv-gazebo/bin/activate
python -B scripts/train_data_driven.py --simulator gazebo
env -u QT_QUICK_BACKEND QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1 python -B scripts/run_experiment.py --simulator gazebo --controller learned --trajectory lemniscate --follow_distance 0.8 --duration 60 --gui --gazebo_render_engine ogre
```

ROS/Gazebo and each backend's dependencies must already be installed. Gazebo
training runs independently of its GUI; successful training does not verify
WSL graphics. Use the existing installation guides for missing dependencies.

## Artifacts and evaluation

- `models/<backend>_response.npz`: fitted matrices, action bounds and metadata.
- `data/identification/<backend>/train_*.npz`: four actual transition recordings.
- `validation_04.npz`: whole withheld identification flight.
- `eval_*.npz` and `report.json`: baseline/learned evaluation trajectories,
  position/attitude RMSE, altitude and optimizer saturation counts.
- `results/learned_validation/`: additional 60-second PyBullet run.

Default training uses 7,680 training transitions and 1,920 validation transitions
at 48 Hz control and 240 Hz physics. Backend and control timestep are checked
when loading. Keep physics frequency and backend settings the same as collection;
retrain after changes to physics, stabilization gains, assets or command semantics.
`--reuse-data` refits saved episodes with matching collection settings; it does
not recollect flights. Do not treat repeated tuning against these evaluations
as an untouched final test set.

PyBullet position RMSE in metres, on 20-second evaluation flights:

| Trajectory | Analytic leader | Learned leader | Analytic follower | Learned follower |
| --- | ---: | ---: | ---: | ---: |
| Figure-eight | 0.00995 | 0.00896 | 0.01430 | 0.01123 |
| B-spline | 0.00781 | 0.00813 | 0.01005 | 0.00972 |

The learned leader was slightly worse on the B-spline; improvement is not
universal. A separate 60-second PyBullet figure-eight completed with leader
RMSE 0.01060 m and follower RMSE 0.01404 m, with altitude remaining between
1.000 and 1.020 m. These errors are relative to each drone's reference, not
a proof that the follower exactly retraces every point without error.

After restricting roll/pitch commands, Gazebo's 20-second validation produced
leader/follower position RMSE of 0.01840/0.03556 m on the figure-eight and
0.01742/0.03515 m on the B-spline. Original wider command bounds caused severe
drift and were rejected. Follower commands still frequently reach the tighter
bounds; these results do not establish performance outside the evaluated regime.
ROS/Gazebo timing makes repeat runs vary.
A further 60-second Gazebo figure-eight completed at 0.02136 m leader and
0.03425 m follower position RMSE, with altitude between 0.928 and 1.027 m.
The 60-second summaries are saved in `results/learned_validation/summary.json`.
These verification flights were headless; GUI rendering was not reverified.

MuJoCo's present backend stabilizes attitude but does not follow requested yaw;
its position metrics must not be presented as full pose-tracking success.
Its learned commands frequently reach bounds. Angular prediction is also less
accurate than translation in PyBullet. Inspect per-state prediction errors and
saturation counts in each report. All results are simulator-only, with no
real-flight validation or demonstrated robustness to disturbances/noise.

The original controller remains available with `--controller analytic`.
`--experiment real_eig` selects its gain set; it does not train or change the
learned controller. Optional Kalman filtering, B-splines and SLERP remain
available, while unrelated Taylor/ODE demonstrations are supporting material.
