# Follow the leader's route

The default formation is now `path`. The leader follows its chosen trajectory.
The follower targets the leader's measured past positions in chronological
order, a configurable arc length behind. It follows the same turns rather
than flying a translated curve or a heading-offset curve. Recorded attitude
is interpolated with SLERP; the same dual-quaternion controller tracks the
resulting pose. No dual-quaternion algebra or core control equations changed.

In the Ubuntu terminal, with ROS and the project environment active:

```bash
python scripts/run_experiment.py --simulator gazebo --experiment real_eig --trajectory lemniscate --follow_distance 0.8 --duration 30 --gui
```

The same arguments work with `--simulator mujoco` or `--simulator pybullet`
in their respective environments. Red lines show actual leader movement;
blue lines show actual follower movement. Trails are sampled every 2.5 cm,
bounded in length to avoid unlimited rendering cost, and cleared for a new
run. Gazebo lines update asynchronously approximately twice per second via
its GUI marker service. Kinematic runs have saved position logs but no GUI.
Use `--no-trails` to disable the overlay.

The initial follower position is behind the leader along its initial motion
direction. A straight entry segment connects it to the leader's starting
position; this segment is seeded, not a claim of recorded pre-start flight.
After the leader has travelled the gap distance, targets lie on its recorded
track. At a stop, the follower maintains the along-path gap rather than
continuing into the leader. Desired velocity comes from successive target
positions, including motion through corners. Finite sampling and physical
tracking error mean actual follower positions will not be mathematically
identical to the leader's track. Noisy position measurements also affect arc
length; use the optional Kalman filter for noisy experiments.

`--follow_distance` is distance along the route, not Euclidean separation.
At self-intersections it is not a collision-avoidance guarantee. At a
crossing, chronological path history determines which branch to follow.

Explicit legacy modes remain available:

- `--follower_offset_mode world`: the original fixed world offset.
- `--follower_offset_mode body`: rotating heading offset.
- `--follower_offset_mode auto`: previous trajectory-dependent selection.

For saved plots use `scripts/plot_results.py --run <saved-file.npz>`; these
plot the logged paths after a run. Live overlays show only motion already
flown, not future reference points.
