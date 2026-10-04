#!/usr/bin/env python3
"""Check ArduPilot launch prerequisites without starting any simulator."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulators.ardupilot import ArduPilotBackend
from simulators.base import SimConfig

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ardupilot_path")
parser.add_argument("--ardupilot_gazebo_path")
args = parser.parse_args()
backend = ArduPilotBackend(SimConfig(ardupilot_path=args.ardupilot_path,
                                    ardupilot_gazebo_path=args.ardupilot_gazebo_path))
try:
    ap, plugin, binary = backend._check_dependencies()
except (RuntimeError, ValueError) as exc:
    print(exc, file=sys.stderr)
    sys.exit(1)
print(f"SITL: {binary}\nGazebo plugin: {plugin / 'build'}\nDependencies and ports ready.")
