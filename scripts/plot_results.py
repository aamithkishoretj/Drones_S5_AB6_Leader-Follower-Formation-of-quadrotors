#!/usr/bin/env python3
"""
Reproduce paper-style plots (Fig. 6-14) and MAE/MSE tables (Tables I-IV)
from a saved run.

Usage:
    python scripts/plot_results.py --run results/proportional_20260101_120000.npz
"""

from __future__ import annotations
import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib.pyplot as plt

from utils import load_run, position_error_metrics, attitude_error_metrics


def plot_trajectories(log: dict, save_path: str | None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(log["leader_pos_d"][:, 0], log["leader_pos_d"][:, 1], "m--", label="Leader Desired")
    ax.plot(log["follower_pos_d"][:, 0], log["follower_pos_d"][:, 1], "c--", label="Follower Desired")
    ax.plot(log["leader_pos"][:, 0], log["leader_pos"][:, 1], "r-", label="Leader Actual")
    ax.plot(log["follower_pos"][:, 0], log["follower_pos"][:, 1], "b-", label="Follower Actual")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Leader and Follower Trajectories")
    ax.legend()
    ax.axis("equal")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_axis_tracking(log: dict, prefix: str, save_path: str | None):
    """prefix: 'leader' or 'follower'. Reproduces Fig. 7/8/10/11/13/14 style plots."""
    t = log["t"]
    pos = log[f"{prefix}_pos"]
    pos_d = log[f"{prefix}_pos_d"]
    labels = ["X", "Y", "Z"]

    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    for i in range(3):
        axes[i, 0].plot(t, pos_d[:, i], "k--", label="desired")
        axes[i, 0].plot(t, pos[:, i], label="simulated")
        axes[i, 0].set_title(f"{prefix.capitalize()} {labels[i]} axis Positioning")
        axes[i, 0].set_xlabel("Time [s]")
        axes[i, 0].set_ylabel("Position [m]")
        axes[i, 0].legend()

        err = pos[:, i] - pos_d[:, i]
        axes[i, 1].plot(t, err, color="r" if i == 0 else ("b" if i == 1 else "g"))
        axes[i, 1].set_title(f"{prefix.capitalize()} {labels[i]} axis Error")
        axes[i, 1].set_xlabel("Time [s]")
        axes[i, 1].set_ylabel("Error [m]")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def print_metrics_tables(log: dict):
    print("\n=== TABLE I/II-style: position error metrics ===")
    for prefix in ("leader", "follower"):
        m = position_error_metrics(log[f"{prefix}_pos"], log[f"{prefix}_pos_d"])
        print(f"{prefix.capitalize():9s} MAE[m]={m['MAE[m]']:.4f}  MSE[m^2]={m['MSE[m^2]']:.4f}")

    print("\n=== TABLE III/IV-style: attitude error metrics ===")
    for prefix in ("leader", "follower"):
        m = attitude_error_metrics(log[f"{prefix}_rpy"], log[f"{prefix}_rpy_d"])
        print(f"-- {prefix} --")
        for axis, vals in m.items():
            print(f"  {axis:6s} MAE[rad]={vals['MAE[rad]']:.2e}  MSE[rad^2]={vals['MSE[rad^2]']:.2e}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="Path to a saved .npz run file.")
    p.add_argument("--no-show", action="store_true", help="Don't call plt.show().")
    args = p.parse_args()

    log = load_run(args.run)
    out_dir = os.path.dirname(os.path.abspath(args.run))
    base = os.path.splitext(os.path.basename(args.run))[0]

    plot_trajectories(log, save_path=os.path.join(out_dir, f"{base}_trajectories.png"))
    plot_axis_tracking(log, "leader", save_path=os.path.join(out_dir, f"{base}_leader.png"))
    plot_axis_tracking(log, "follower", save_path=os.path.join(out_dir, f"{base}_follower.png"))
    print_metrics_tables(log)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
