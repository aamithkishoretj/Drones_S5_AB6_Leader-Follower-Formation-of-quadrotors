#!/usr/bin/env python3
"""
Run one of the three leader-follower experiments from Section V of the paper
inside gym-pybullet-drones.

Usage:
    python scripts/run_experiment.py --experiment proportional
    python scripts/run_experiment.py --experiment complex_eig --gui
    python scripts/run_experiment.py --experiment real_eig --duration 20 --gui
"""

from __future__ import annotations
import argparse
import os
import sys

# Make the project root importable regardless of CWD.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dq_control import get_gains, EXPERIMENTS
from envs import LeaderFollowerSim, SimConfig
from utils import save_run


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--experiment", choices=list(EXPERIMENTS.keys()), required=True,
        help="Which gain set from Section V to run.",
    )
    p.add_argument("--duration", type=float, default=30.0, help="Simulation duration, s.")
    p.add_argument("--pyb_freq", type=int, default=240, help="Physics steps / s.")
    p.add_argument("--ctrl_freq", type=int, default=48, help="Controller / PID steps / s.")
    p.add_argument("--gui", action="store_true", help="Show the PyBullet GUI.")
    p.add_argument("--output", type=str, default=os.path.join(_ROOT, "results"))
    return p.parse_args()


def main():
    args = parse_args()

    gains = get_gains(args.experiment)
    cfg = SimConfig(
        duration_sec=args.duration,
        pyb_freq=args.pyb_freq,
        ctrl_freq=args.ctrl_freq,
        gui=args.gui,
        output_folder=args.output,
    )

    print(f"[run_experiment] experiment={args.experiment}  duration={args.duration}s  gui={args.gui}")
    sim = LeaderFollowerSim(gains=gains, cfg=cfg)
    log = sim.run()
    save_run(log, experiment_name=args.experiment, output_folder=args.output)


if __name__ == "__main__":
    main()
