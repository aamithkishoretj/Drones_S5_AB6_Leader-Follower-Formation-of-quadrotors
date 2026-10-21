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

The earlier MuJoCo evaluation below used a backend that did not follow requested yaw;
its position metrics must not be presented as full pose-tracking success.
Its learned commands frequently reach bounds. Angular prediction is also less
accurate than translation in PyBullet. Inspect per-state prediction errors and
saturation counts in each report. All results are simulator-only, with no
real-flight validation or demonstrated robustness to disturbances/noise.

The original controller remains available with `--controller analytic`.
`--experiment real_eig` selects its gain set; it does not train or change the
learned controller. Optional Kalman filtering, B-splines and SLERP remain
available, while unrelated Taylor/ODE demonstrations are supporting material.

Update: MuJoCo now tracks commanded yaw. Its matching retrained model and
new evaluation recordings are in `data/yaw_tracking/mujoco/`. Earlier MuJoCo
results describe the old backend and should not be used for the new model.

## ArduCopter SITL: data-driven dual-quaternion formation control

ArduPilot does not use the `LearnedResponse` models above. Guided mode accepts
velocity and yaw rate only (ArduCopter controls roll/pitch), so it has a
separate four-input model, `models/ardupilot_dq_dynamics.npz`, loaded by
`dq_control/data_driven.py`. Selecting `learned` (UI or CLI) or the CLI
default `auto` with `--simulator ardupilot` uses this controller; the two
model formats are intentionally not interchangeable.

The default ArduPilot flight loop uses an identified dynamics model to choose
commands. It does not run the paper's kinematic law alongside a prediction
plot. Both leader and follower use `DataDrivenDQController`; the original
analytical controller is available explicitly as a teacher/comparison mode.

### What the model learns

We generated three separate 60-second airborne ArduPilot–Gazebo flights, with
two Iris vehicles separated by 10 m during identification. Smooth target motion
and independent multisine command perturbations excite horizontal/vertical
velocity, yaw and tilt. The existing analytical DQ controller bounds the
training flights; it is not the deployed learned formation controller.

The raw NPZ files contain each vehicle's own SITL timestamp, estimated state,
unit dual-quaternion pose, and the ENU velocity/yaw-rate command after backend
limiting/frame conversion. In this first batch, yaw-rate targets were computed
from the sampled attitude before the backend's final telemetry refresh; this
small timing approximation is recorded in each dataset's metadata. The updated
collector records the backend's exact transmitted targets. These are simulated ArduPilot estimates, not real
flight data or independent Gazebo ground truth. Dataset files are:

- `results/data_driven/training_seed_1.npz`: training.
- `results/data_driven/training_seed_2.npz`: training.
- `results/data_driven/training_seed_3.npz`: held-out validation/model selection.

The first five seconds of each flight are excluded from model fitting and
validation to omit settling after takeoff. Entire flights are kept in distinct
splits; data hashes are stored with the model. The raw flights are retained.

Transitions use a fixed 0.2-second interval. Velocity/rates are interpolated,
orientation uses SLERP, and applied inputs use time-weighted zero-order holding.
This accounts for unequal telemetry sampling intervals without assuming 24 Hz
measurements. No train/validation samples are shuffled across flight boundaries.

The learned state is six-dimensional:

`s = [heading_vx, heading_vy, vz, yaw_rate, body_up_x, body_up_y]`.

Heading-relative velocity and tilt are computed from the principal quaternion
of the measured dual-quaternion pose. Commands are four-dimensional:
`u = [heading_vx_command, heading_vy_command, vz_command, yaw_rate_command]`.
Translation and yaw-frame invariance allow the same model to operate elsewhere
in the Gazebo world. The model learns the closed-loop ArduPilot/Iris response to
Guided inputs; it is not a motor-level physical model of an uncontrolled drone.

We fit `s_next = A s + B u + c`, testing a quadratic-feature extension as well.
Scaled ridge regression and shrinkage of learned increments toward persistence
regularize sensor noise. The final selected model uses linear features; adding
nonlinear features did not improve validation enough to select them. This is
not claimed to remove nonlinearity globally or to be an exact Koopman model.

`models/ardupilot_dq_dynamics.npz` stores the fitted weights, 6x6 state-transition
matrix, 6x4 control matrix, interval and provenance. The matrix maps the learned
response state, not the entire dual-quaternion pose. When quadratic features are
selected in a future retraining, those matrices are only the linear portions.

### How dual quaternions determine the commands

At every control update the controller:

1. Builds current/desired unit dual-quaternion poses using the existing algebra.
2. Rolls out learned response dynamics for five 0.2-second steps (one second).
3. Reconstructs a valid dual-quaternion pose after each predicted step.
4. Computes `pose_error(predicted_pose, desired_future_pose)`.
5. Chooses bounded velocity/yaw-rate commands by minimizing this pose error,
   desired-twist mismatch and command-change cost.
6. Applies the first constant-command solution and replans after new telemetry.

This is a short-horizon, single-control-block predictive controller. It does not
optimize a separate command for every horizon step. Roll/pitch responses are
predicted from data, while only velocity/yaw rate are directly commanded through
Guided mode. DQ sign ambiguity and unit-pose constraints are preserved.
The follower still targets the leader's measured chronological path with a
1.5 m along-path gap; live trails and result plots remain available.

The `--experiment real_eig` option remains for CLI compatibility and teacher
selection. Its analytical gains are not used to choose commands when
`--controller data_driven` is active. Model changes measurably change commands;
unit tests forbid calling `KinematicController.compute` in the learned loop.
Missing or unvalidated models produce an error rather than silently reverting
to the analytical controller.

### Generate data and retrain

From the project root, after installing ArduPilot/Gazebo prerequisites:

```bash
bash scripts/train_data_driven.sh
```

On the development machine the existing Copter-4.5.7 build has been moved to
`../ardupilot-old`, while `../ardupilot` is incomplete. For that installation:

```bash
bash scripts/train_data_driven.sh --ardupilot_path ../ardupilot-old
```

The script collects three flights sequentially, fits on seeds 1/2, selects and
validates on seed 3, then creates one-second forecast plots. Runtime logs remain
ignored by Git. Running this script overwrites the named seed files/model.
The saved model is supplied, so retraining is not required for a demonstration.

### Run learned formation control

```bash
bash scripts/run_ardupilot.sh 60 --ardupilot_path ../ardupilot-old
bash scripts/run_ardupilot_potato_chip.sh 60 --ardupilot_path ../ardupilot-old
```

Omit the source-path argument on an installation using the usual sibling name.
The generic ArduPilot CLI also defaults to data-driven control. Other simulator
backends keep their existing controllers; this model has not been identified
for PyBullet, MuJoCo or the ROS Gazebo vehicle.

For B-splines, whose duration is traversal time rather than a repeated period:

```bash
.venv/bin/python scripts/run_experiment.py --simulator ardupilot \
  --experiment real_eig --controller data_driven --trajectory bspline \
  --duration 40 --follow_distance 1.5 --ctrl_freq 24 --gui \
  --ardupilot_low_resource --gazebo_render_engine ogre \
  --ardupilot_path ../ardupilot-old
```

The same learned controller is used for each path. Test new path shapes/speeds
before presenting them; short durations can demand motion outside training.
To compare the legacy law, explicitly pass `--controller analytical`.

### Prediction validation

`models/ardupilot_dq_validation.json` contains every tested model, selection
metrics and data hashes. Training used 1,096 transitions; validation used 548.
The selected model improves one-step RMSE over persistence on all six outputs.
Its state-transition spectral radius is 0.874 and input matrix rank is four.
These properties concern the fitted local response model, not a proof of
stability or controllability for the complete nonlinear formation.

Across 108 independent one-second windows in the validation flight:

| Metric | Learned predictor | Constant-velocity/rate prediction |
| --- | --- | --- |
| Position RMS | 0.0490 m | 0.1006 m |
| Yaw RMS | 0.0185 rad | 0.1171 rad |

These forecasts use recorded future commands. They are prediction tests, not
closed-loop tracking errors. The validation flight also selected hyperparameters,
so it is not an untouched test set. Subsequent formation flights use new paths
and assess the controller in the live simulator.
The vertical one-step improvement is small (0.0640 vs 0.0646 m/s); richer
identification flights would be needed to strengthen this part of the model.
Forecast reports and plots are in `results/data_driven/`.

### Live learned-controller tracking tests

All three tests below were run headlessly on 6 October 2026. Both vehicles
completed tracking, landed and disarmed in each run. These paths were not
identification trajectories, and the analytical controller did not choose
commands in these flights.

| Reference | Tracking time | Leader position RMS | Follower position RMS | Minimum measured separation |
| --- | --- | --- | --- | --- |
| Lemniscate | 60 s | 0.0793 m | 0.0867 m | 0.978 m |
| Potato-chip | 60 s | 0.0691 m | 0.0702 m | 1.278 m |
| Existing 3D B-spline | 40 s | 0.0465 m | 0.0655 m | 0.860 m |

Saved runs, tracking/error plots and `tracking_summary.json` are in
`results/data_driven/closed_loop/`. B-splines now also receive a 3D plot.
All errors are relative to the desired poses using estimated telemetry.
Measured separation is not a guaranteed collision-clearance bound.
The analytical baseline previously achieved smaller position errors on the
figure-eight and potato-chip paths. Data-driven operation is demonstrated here;
improved tracking over that baseline is not claimed. More excitation data and
horizon/cost tuning may improve performance. Learned-controller GUI performance
has not been separately benchmarked; these live tests used the headless profile.
