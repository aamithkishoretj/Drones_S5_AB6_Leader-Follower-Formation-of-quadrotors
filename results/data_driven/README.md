# Simulator-generated identification data and live control results

Three two-vehicle ArduPilot–Gazebo identification flights supply training and
validation data. See [the complete method](../../docs/data_driven.md) for input
capture provenance, dataset splits, model fitting and limitations.

- `training_seed_1.npz`, `training_seed_2.npz`: model-training flights.
- `training_seed_3.npz`: validation and hyperparameter selection.
- `forecast_validation.json` / `.png`: independent one-second prediction windows
  within the validation flight, with recorded future commands.
- `closed_loop/`: actual learned-controller flights on new reference paths,
  their tracking plots and measured error/separation summary.

The initial batch calculates post-limiter yaw targets from the sampled attitude
before the final telemetry refresh; this approximation is noted inside each
NPZ's metadata. The collector now captures exact transmitted commands from the
backend. Velocity targets in the initial batch are independent of that refresh.
Raw arrays are retained and their full-file hashes are recorded with the model.
All data comes from simulation, not real drone flights. Runtime worlds, log
files and SITL parameter/storage directories are excluded from Git.
