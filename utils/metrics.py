"""
Error metrics reproducing Tables I-IV of the paper:
  Table I:   leader position  MAE[m], MSE[m^2]
  Table II:  follower position MAE[m], MSE[m^2]
  Table III: leader attitude  MAE[rad], MSE[rad^2] (per phi, theta, psi)
  Table IV:  follower attitude MAE[rad], MSE[rad^2] (per phi, theta, psi)
"""

from __future__ import annotations
import numpy as np


def mae(err: np.ndarray) -> float:
    """Mean absolute error over a (T,) or (T,N) error array (scalar result)."""
    return float(np.mean(np.abs(err)))


def mse(err: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean(err ** 2))


def position_error_metrics(pos: np.ndarray, pos_d: np.ndarray) -> dict:
    """pos, pos_d: (T, 3) arrays. Returns MAE/MSE of the Euclidean position error,
    matching how Tables I-II report a single MAE[m]/MSE[m^2] per vehicle."""
    err = np.linalg.norm(pos - pos_d, axis=1)  # (T,)
    return {"MAE[m]": mae(err), "MSE[m^2]": mse(err)}


def attitude_error_metrics(rpy: np.ndarray, rpy_d: np.ndarray) -> dict:
    """rpy, rpy_d: (T, 3) arrays [roll, pitch, yaw] in rad.
    Returns per-axis MAE/MSE matching Tables III-IV (phi, theta, psi rows)."""
    err = _wrap_angle(rpy - rpy_d)  # (T, 3)
    axes = ["phi", "theta", "psi"]
    out = {}
    for i, ax in enumerate(axes):
        out[ax] = {"MAE[rad]": mae(err[:, i]), "MSE[rad^2]": mse(err[:, i])}
    return out


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angle error to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi
