#!/usr/bin/env python3
"""Prepare optional simulator dependencies.

External Git repositories are cloned as siblings of this project:
    parent/
      Drones_S5_AB6_Leader-Follower-Formation-of-quadrotors/
      gym-pybullet-drones/
      ardupilot/

MuJoCo is distributed as a Python package. Gazebo Sim is installed as a
system package together with ROS 2/ros_gz, so neither belongs inside this
repository.
"""
from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simulators.dependencies import ensure_external_repo, print_dependency_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulator",
        choices=["pybullet", "mujoco", "gazebo", "ardupilot", "all"],
        required=True,
    )
    parser.add_argument("--update", action="store_true", help="git pull an existing sibling repository")
    args = parser.parse_args()

    targets = {
        "pybullet": ["gym-pybullet-drones"],
        "ardupilot": ["ardupilot"],
        "mujoco": [],
        "gazebo": [],
        "all": ["gym-pybullet-drones", "ardupilot"],
    }[args.simulator]

    for repo in targets:
        path = ensure_external_repo(repo, update=args.update)
        print(f"[setup] {repo}: {path}")

    if args.simulator in ("mujoco", "all"):
        print(
            "[setup] MuJoCo backend is reserved in the architecture. "
            "Install the Python package in your virtual environment when its backend is added."
        )

    if args.simulator in ("gazebo", "all"):
        print(
            "[setup] Gazebo uses a system installation (recommended: Gazebo Harmonic + ROS 2 Jazzy). "
            "See gazebo/README.md."
        )

    if not targets:
        print_dependency_status()


if __name__ == "__main__":
    main()
