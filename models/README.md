# Learned ArduPilot response model

`ardupilot_dq_dynamics.npz` is used by both live-demo controllers. It contains:

- `weights`: regression coefficients for constant/state/input features.
- `state_transition`: the learned 6x6 response-state matrix A.
- `control_matrix`: the learned 6x4 Guided-input matrix B.
- `dt`: 0.2 simulation seconds.
- `quadratic`: false for the selected model.
- `metadata_json`: splits, dataset hashes, feature/input order and validation.

This is an identified model of ArduPilot estimated motion under Guided velocity
commands, not an imitation of teacher commands or a fitted motor model.
Dual-quaternion pose error inside the predictive controller selects the commands.
The pose remains on its unit dual-quaternion manifold during prediction.

See [the full method and reproduction guide](../docs/data_driven.md).
The supplied raw datasets and report permit rerunning the fit without launching
SITL; regenerating the datasets requires Gazebo and the built SITL installation.
Validation was used for model selection; it is not an untouched test set.
New-path live formation tests are saved separately under
`results/data_driven/closed_loop/`.
