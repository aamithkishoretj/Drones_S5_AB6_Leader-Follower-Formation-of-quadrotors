# Data-driven dual-quaternion formation control

The default ArduPilot flight loop uses an identified dynamics model to choose
commands. It does not run the paper's kinematic law alongside a prediction
plot. Both leader and follower use `DataDrivenDQController`; the original
analytical controller is available explicitly as a teacher/comparison mode.
`Modified` is not changed: this work lives on `ardupilot-gazebo`.

## What the model learns

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

## How dual quaternions determine the commands

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

## Generate data and retrain

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

## Run learned formation control

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

## Prediction validation

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

## Live learned-controller tracking tests

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
