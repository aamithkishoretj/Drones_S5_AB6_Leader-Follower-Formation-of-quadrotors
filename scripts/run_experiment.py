#!/usr/bin/env python3


from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np

# Make the project root importable regardless of CWD.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dq_control import (
    get_gains, EXPERIMENTS,
    LeaderTrajectory, LemniscateParams,
    PotatoChipTrajectory, PotatoChipParams,
)
from simulators import SimConfig, LeaderFollowerSimulation, available_simulators
from utils import save_run
from dq_control import Quaternion


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--experiment", choices=list(EXPERIMENTS.keys()), required=True,
        help="Which gain set from Section V to run.",
    )
    p.add_argument(
        "--simulator", choices=available_simulators(), default="pybullet",
        help="Simulation backend. The formation controller is shared across backends.",
    )
    p.add_argument(
        "--trajectory", choices=["lemniscate", "potato_chip", "bspline", "interpolated"], default="lemniscate",
        help="Shape the leader flies. 'lemniscate' = paper's figure-eight (default). "
             "'potato_chip' = a saddle/Pringle-shaped 3D curve (circle in x,y with a "
             "cos(k*theta) ripple in z).",
    )
    p.add_argument("--duration", type=float, default=30.0, help="Simulation duration, s.")
    p.add_argument(
        "--physics_freq", "--pyb_freq", dest="pyb_freq", type=int, default=240,
        help="Physics steps / s. Used by PyBullet; accepted for compatibility with the original CLI.",
    )
    p.add_argument("--ctrl_freq", type=int, default=48, help="Controller / low-level control steps / s.")
    p.add_argument("--gui", action="store_true", help="Show the simulator GUI when supported.")
    p.add_argument("--gazebo_render_engine", choices=["ogre2", "ogre"], default="ogre2",
                   help="Gazebo GUI renderer; try ogre for WSL graphics issues")
    p.add_argument("--output", type=str, default=os.path.join(_ROOT, "results"))
    ap = p.add_argument_group("ArduPilot SITL + Gazebo")
    ap.add_argument("--ardupilot_path", help="ArduPilot source/build directory (default: sibling ardupilot)")
    ap.add_argument("--ardupilot_gazebo_path", help="Built official Gazebo plugin directory (default: sibling ardupilot_gazebo)")
    ap.add_argument("--ardupilot_startup_timeout", type=float, default=120., help="Wall seconds allowed for connection, EKF readiness and takeoff")
    ap.add_argument("--ardupilot_land_timeout", type=float, default=60., help="Wall seconds allowed for landing before stopping SITL")
    ap.add_argument("--ardupilot_max_speed", type=float, default=2., help="Maximum commanded velocity norm in m/s")
    ap.add_argument("--ardupilot_low_resource", action="store_true", help="400 Hz physics, no shadows, reduced rendering cost")
    ap.add_argument("--ardupilot_min_separation", type=float, default=.65, help="Stop tracking when the Iris vehicles get closer than this distance (m)")
    p.add_argument("--trajectory_file", help="JSON control points/knots or interpolation endpoints; see configs/course_*.json")
    p.add_argument("--optimize_path", action="store_true", help="Smooth B-spline control points with CVXPY")
    p.add_argument("--kalman", action="store_true", help="Estimate position/velocity from position measurements")
    p.add_argument("--position_noise_std", type=float, default=0.0, help="Injected position noise standard deviation in metres")
    p.add_argument("--kalman_measurement_std", type=float, default=0.03)
    p.add_argument("--kalman_acceleration_std", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--follow_distance", type=float, default=0.8, help="Distance behind the leader along its recorded path, metres")
    p.add_argument("--no-trails", action="store_true", help="Disable actual-flight path lines")
    p.add_argument("--x_offset", type=float, default=1.85, help="Follower offset magnitude, m (eq. 16).")
    p.add_argument(
        "--follower_offset_mode", choices=["path", "world", "body", "auto"], default="path",
        help="'path' (default) replays the leader's recorded track at --follow_distance. "
             "'world' = fixed [x_offset,0,0] in the world frame (paper eq. 16 exactly, "
             "good for the lemniscate). 'body' = offset rotated into the leader's current "
             "heading so the follower always trails directly behind it (needed for curved "
             "paths like potato_chip). 'auto' picks 'world' for --trajectory "
             "lemniscate and 'body' for --trajectory potato_chip.",
    )
    p.add_argument(
        "--follower_heading_source", choices=["velocity", "attitude"], default="velocity",
        help="(only used with --follower_offset_mode body) 'velocity' (default, robust) "
             "derives the trailing direction from the leader's smoothed measured motion, "
             "independent of yaw-tracking lag. 'attitude' uses the leader's measured yaw "
             "directly -- simpler but sensitive to yaw-tracking error.",
    )
    p.add_argument(
        "--follower_heading_smoothing", type=float, default=0.15,
        help="(0,1]: smoothing for the velocity-derived heading estimate. Lower = more "
             "noise rejection but slower to respond to real turns.",
    )
    p.add_argument(
        "--follower_vel_smoothing", type=float, default=1.0,
        help="(0,1]: 1.0 = raw finite-difference velocity (paper default), lower = smoother "
             "but laggier follower velocity estimate.",
    )
    p.add_argument("--start_x", type=float, default=None, help="Override leader's exact t=0 X position, m (e.g. 0.0 for the origin).")
    p.add_argument("--start_y", type=float, default=None, help="Override leader's exact t=0 Y position, m.")
    p.add_argument("--start_z", type=float, default=None, help="Override leader's exact t=0 Z (altitude) position, m.")

    # --- lemniscate shape params (used when --trajectory lemniscate) ---
    g_lem = p.add_argument_group("lemniscate trajectory options")
    g_lem.add_argument("--r_x", type=float, default=0.85, help="Lemniscate half-width, m.")
    g_lem.add_argument("--r_y", type=float, default=0.65, help="Lemniscate half-height, m.")
    g_lem.add_argument("--w_d", type=float, default=np.pi / 15, help="Angular speed, rad/s (bigger = faster loop).")
    g_lem.add_argument("--x0", type=float, default=1.51, help="Lemniscate center x, m.")
    g_lem.add_argument("--y0", type=float, default=-0.27, help="Lemniscate center y, m.")
    g_lem.add_argument("--z0", type=float, default=1.0, help="Flight altitude, m.")

    # --- potato chip shape params (used when --trajectory potato_chip) ---
    g_chip = p.add_argument_group("potato_chip trajectory options")
    g_chip.add_argument("--chip_r", type=float, default=0.85, help="Circular footprint radius, m.")
    g_chip.add_argument("--chip_w", type=float, default=np.pi / 15, help="Angular speed around the circle, rad/s.")
    g_chip.add_argument("--chip_z_amp", type=float, default=0.35, help="Peak height of the saddle ripple, m.")
    g_chip.add_argument("--chip_k", type=int, default=2, help="Saddle lobes per revolution (2 = classic Pringle shape).")
    g_chip.add_argument("--chip_phase", type=float, default=0.0, help="Phase offset of the z ripple, rad.")
    g_chip.add_argument("--chip_x0", type=float, default=1.51, help="Chip center x, m.")
    g_chip.add_argument("--chip_y0", type=float, default=-0.27, help="Chip center y, m.")
    g_chip.add_argument("--chip_z0", type=float, default=1.0, help="Chip center altitude, m.")

    return p.parse_args()


def build_leader_trajectory(args):
    if args.optimize_path and args.trajectory != "bspline":
        raise ValueError("--optimize_path requires --trajectory bspline")
    if args.trajectory_file and args.trajectory not in ("bspline", "interpolated"):
        raise ValueError("--trajectory_file requires bspline or interpolated")
    if args.trajectory in ("bspline", "interpolated"):
        from dq_control.course_trajectories import BSplineTrajectory, InterpolatedPoseTrajectory
        path = args.trajectory_file or os.path.join(_ROOT, "configs", f"course_{args.trajectory}.json")
        with open(path, encoding="utf-8") as stream:
            spec = json.load(stream)
        if args.trajectory == "bspline":
            points = np.asarray(spec["control_points"], float)
            if args.optimize_path:
                from dq_control.optimization import smooth_control_points
                points = smooth_control_points(points, spec.get("smoothing_weight", 1.), spec.get("bounds"))
            traj = BSplineTrajectory(points, args.duration, spec.get("altitude", 1.), spec.get("knots"))
        else:
            traj = InterpolatedPoseTrajectory(spec["start"], spec["end"],
                Quaternion.from_rpy(spec.get("start_rpy", [0, 0, 0])),
                Quaternion.from_rpy(spec.get("end_rpy", [0, 0, 0])), args.duration)
        if any(v is not None for v in (args.start_x, args.start_y, args.start_z)):
            raise ValueError("For course trajectories, set starting coordinates in --trajectory_file")
        return traj
    if args.trajectory == "lemniscate":
        params = LemniscateParams(
            r_x=args.r_x, r_y=args.r_y, w_d=args.w_d,
            x0=args.x0, y0=args.y0, z0=args.z0,
        )
        traj = LeaderTrajectory(params)

    elif args.trajectory == "potato_chip":
        params = PotatoChipParams(
            r=args.chip_r, w=args.chip_w, z_amp=args.chip_z_amp,
            k=args.chip_k, phase=args.chip_phase,
            x0=args.chip_x0, y0=args.chip_y0, z0=args.chip_z0,
        )
        traj = PotatoChipTrajectory(params)

    else:
        raise ValueError(f"Unknown trajectory '{args.trajectory}'")

    # If --start_x/--start_y/--start_z were given, shift the trajectory's
    # center offsets (x0, y0, z0) so that traj.position(0.0) lands exactly
    # on the requested start point, regardless of trajectory shape. This
    # works because every trajectory's own (x0, y0, z0) fields are plain
    # additive offsets on top of a zero-mean oscillation -- shifting them
    # by (desired - actual) moves the whole shape rigidly without changing
    # its size/speed/orientation.
    if args.start_x is not None or args.start_y is not None or args.start_z is not None:
        p0 = traj.position(0.0)
        desired = np.array([
            args.start_x if args.start_x is not None else p0[0],
            args.start_y if args.start_y is not None else p0[1],
            args.start_z if args.start_z is not None else p0[2],
        ])
        offset = desired - p0
        params.x0 += offset[0]
        params.y0 += offset[1]
        params.z0 += offset[2]
        assert np.allclose(traj.position(0.0), desired, atol=1e-9), (
            "internal error: trajectory did not shift to the requested start point"
        )

    return traj


def main():
    args = parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        raise ValueError("--duration must be finite and positive")

    follower_offset_mode = args.follower_offset_mode
    if follower_offset_mode == "auto":
        follower_offset_mode = "body" if args.trajectory == "potato_chip" else "world"

    gains = get_gains(args.experiment)
    cfg = SimConfig(
        simulator=args.simulator,
        gazebo_render_engine=args.gazebo_render_engine,
        duration_sec=args.duration,
        pyb_freq=args.pyb_freq,
        ctrl_freq=args.ctrl_freq,
        gui=args.gui,
        output_folder=args.output,
        x_offset=args.x_offset,
        follower_offset_mode=follower_offset_mode,
        follower_heading_source=args.follower_heading_source,
        follower_heading_smoothing=args.follower_heading_smoothing,
        follower_vel_smoothing=args.follower_vel_smoothing,
        kalman=args.kalman,
        position_noise_std=args.position_noise_std,
        kalman_measurement_std=args.kalman_measurement_std,
        kalman_acceleration_std=args.kalman_acceleration_std,
        seed=args.seed,
        follow_distance=args.follow_distance,
        trails=not args.no_trails,
        ardupilot_path=args.ardupilot_path,
        ardupilot_gazebo_path=args.ardupilot_gazebo_path,
        ardupilot_startup_timeout=args.ardupilot_startup_timeout,
        ardupilot_land_timeout=args.ardupilot_land_timeout,
        ardupilot_max_speed=args.ardupilot_max_speed,
        ardupilot_low_resource=args.ardupilot_low_resource,
        ardupilot_min_separation=args.ardupilot_min_separation,
    )
    leader_traj = build_leader_trajectory(args)

    print(
        f"[run_experiment] simulator={args.simulator}  experiment={args.experiment}  "
        f"trajectory={args.trajectory}  follower_offset_mode={follower_offset_mode}  "
        f"duration={args.duration}s  gui={args.gui}"
    )
    sim = LeaderFollowerSimulation(gains=gains, cfg=cfg, leader_traj=leader_traj)
    log = sim.run()
    log["run_options_json"] = np.asarray(json.dumps(vars(args), sort_keys=True))
    if args.trajectory == "bspline":
        log["trajectory_control_points"] = leader_traj.points.copy()
        log["trajectory_knots"] = leader_traj.knots.copy()
    elif args.trajectory == "interpolated":
        log["trajectory_endpoints"] = np.array([leader_traj.start, leader_traj.end])
        log["trajectory_attitudes_xyzw"] = np.array([leader_traj.q0.as_array(), leader_traj.q1.as_array()])
    run_name = f"{args.simulator}_{args.experiment}_{args.trajectory}"
    save_run(log, experiment_name=run_name, output_folder=args.output)


if __name__ == "__main__":
    main()
