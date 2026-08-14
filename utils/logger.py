"""Minimal run persistence: save the dict returned by LeaderFollowerSim.run()
to a timestamped .npz file, and reload it later for plotting/metrics."""

from __future__ import annotations
import os
import time
import numpy as np


def save_run(log: dict, experiment_name: str, output_folder: str = "results") -> str:
    os.makedirs(output_folder, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_folder, f"{experiment_name}_{stamp}.npz")
    np.savez(path, **log)
    print(f"[logger] saved run to {path}")
    return path


def load_run(path: str) -> dict:
    data = np.load(path)
    return {k: data[k] for k in data.files}
