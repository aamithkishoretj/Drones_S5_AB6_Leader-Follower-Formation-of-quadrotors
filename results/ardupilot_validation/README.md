# ArduPilot + Gazebo validation examples

These are measured headless runs with two Iris vehicles, ArduCopter
`Copter-4.5.7`, Gazebo Harmonic, and the official Gazebo plugin at commit
`082a0fe231f6e63bc8d1598f1cba461d9e2ea7f5`, completed on 4 October 2026.
Both runs completed takeoff, 60 seconds of tracking, landing and disarming.

| Run | Leader position RMS | Follower position RMS | Minimum measured separation |
| --- | --- | --- | --- |
| Lemniscate `20261004_161123` | 0.0340 m | 0.0727 m | 1.042 m |
| Potato-chip `20261004_162130` | 0.0274 m | 0.0514 m | 1.234 m |

RMS is `sqrt(mean(sum((actual - desired)**2, axis=1)))` in metres.
Separation is the Euclidean distance between measured vehicle positions.
Each NPZ contains 878 samples, from 0 to 59.943 seconds. Tracking uses SITL
timestamps; the requested control rate is 24 Hz, with observed intervals of
0.042–0.083 seconds while waiting for both vehicles' telemetry.
The follower reference replays the leader's measured route with a 1.5 m arc
distance gap. These are individual run measurements, not general guarantees.

Reproduce the flight with the documented demo launchers in
[the ArduPilot guide](../../ardupilot/README.md). To regenerate figures:

```bash
.venv/bin/python scripts/plot_results.py --run \
  results/ardupilot_validation/ardupilot_real_eig_lemniscate_20261004_161123.npz --no-show
.venv/bin/python scripts/plot_results.py --run \
  results/ardupilot_validation/ardupilot_real_eig_potato_chip_20261004_162130.npz --no-show
```

Temporary runtime logs, generated worlds and SITL storage are excluded from Git.
