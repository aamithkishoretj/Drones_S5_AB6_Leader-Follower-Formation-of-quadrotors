#!/usr/bin/env python3


from __future__ import annotations
import argparse
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
from envs import LeaderFollowerSim, SimConfig
from utils import save_run


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--experiment", choices=list(EXPERIMENTS.keys()), required=True,
        help="Which gain set from Section V to run.",
    )
    p.add_argument(
        "--trajectory", choices=["lemniscate", "potato_chip"], default="lemniscate",
        help="Shape the leader flies. 'lemniscate' = paper's figure-eight (default). "
             "'potato_chip' = a saddle/Pringle-shaped 3D curve (circle in x,y with a "
             "cos(k*theta) ripple in z).",
    )
    p.add_argument("--duration", type=float, default=30.0, help="Simulation duration, s.")
    p.add_argument("--pyb_freq", type=int, default=240, help="Physics steps / s.")
    p.add_argument("--ctrl_freq", type=int, default=48, help="Controller / PID steps / s.")
    p.add_argument("--gui", action="store_true", help="Show the PyBullet GUI.")
    p.add_argument("--output", type=str, default=os.path.join(_ROOT, "results"))
    p.add_argument("--x_offset", type=float, default=1.85, help="Follower offset magnitude, m (eq. 16).")
    p.add_argument(
        "--follower_offset_mode", choices=["world", "body", "auto"], default="auto",
        help="'world' = fixed [x_offset,0,0] in the world frame (paper eq. 16 exactly, "
             "good for the lemniscate). 'body' = offset rotated into the leader's current "
             "heading so the follower always trails directly behind it (needed for curved "
             "paths like potato_chip). 'auto' (default) picks 'world' for --trajectory "
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

    follower_offset_mode = args.follower_offset_mode
    if follower_offset_mode == "auto":
        follower_offset_mode = "body" if args.trajectory == "potato_chip" else "world"

    gains = get_gains(args.experiment)
    cfg = SimConfig(
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
    )
    leader_traj = build_leader_trajectory(args)

    print(
        f"[run_experiment] experiment={args.experiment}  trajectory={args.trajectory}  "
        f"follower_offset_mode={follower_offset_mode}  duration={args.duration}s  gui={args.gui}"
    )
    sim = LeaderFollowerSim(gains=gains, cfg=cfg, leader_traj=leader_traj)
    log = sim.run()
    run_name = f"{args.experiment}_{args.trajectory}"
    save_run(log, experiment_name=run_name, output_folder=args.output)


if __name__ == "__main__":
    main()