"""
Gain sets from Section V ("Experimental Results") of the paper, reproduced
verbatim for the leader (L) and follower (F).

Three experiments:
  1. proportional  -- Sec V-A: proportional-only gains, all integral gains = 0
  2. complex_eig    -- Sec V-B: integral gains chosen s.t. M_omega, M_v have
                        complex eigenvalues (oscillatory tracking error)
  3. real_eig        -- Sec V-C: integral gains chosen s.t. M_omega, M_v have
                        only real eigenvalues (best performance per the paper)

All matrices are 3x3 and negative definite, as required by Theorem 1.
"""

from __future__ import annotations
import numpy as np
from .controller import ControllerGains

I2 = np.eye(2)


def _block(diag_xy: float, diag_z: float) -> np.ndarray:
    """Builds diag(diag_xy, diag_xy, diag_z), matching the paper's
    [ -a*I_2x2, 0; 0, -b ] block structure used throughout Sec. V."""
    return np.diag([diag_xy, diag_xy, diag_z])


# ---------------------------------------------------------------------------
# Experiment 1: Proportional controller only (Sec V-A)
# ---------------------------------------------------------------------------
K_L_omega_p = _block(-3 / 2, -2.0)
K_L_v_p = _block(-9 / 5, -3 / 2)
K_F_omega_p = _block(-3 / 2, -9 / 5)
K_F_v_p = _block(-8 / 5, -9 / 5)

ZERO3 = np.zeros((3, 3))


def gains_proportional() -> dict:
    """Sec V-A: Kw_i = Kv_i = K_eta = K_xi = 0 for both vehicles."""
    leader = ControllerGains(
        Kw_p=K_L_omega_p, Kv_p=K_L_v_p,
        Kw_i=ZERO3, Kv_i=ZERO3, K_eta=ZERO3, K_xi=ZERO3,
    )
    follower = ControllerGains(
        Kw_p=K_F_omega_p, Kv_p=K_F_v_p,
        Kw_i=ZERO3, Kv_i=ZERO3, K_eta=ZERO3, K_xi=ZERO3,
    )
    return {"leader": leader, "follower": follower}


# ---------------------------------------------------------------------------
# Experiment 2: Integral gains -> complex eigenvalues of M_omega, M_v (Sec V-B)
# ---------------------------------------------------------------------------
def gains_complex_eig() -> dict:
    Kw_i = -4 / 5 * np.eye(3)
    Kv_i = -4 / 5 * np.eye(3)
    K_xi = -3 / 10 * np.eye(3)
    K_eta = -1 / 5 * np.eye(3)

    leader = ControllerGains(
        Kw_p=K_L_omega_p, Kv_p=K_L_v_p,
        Kw_i=Kw_i, Kv_i=Kv_i, K_eta=K_eta, K_xi=K_xi,
    )
    follower = ControllerGains(
        Kw_p=K_F_omega_p, Kv_p=K_F_v_p,
        Kw_i=Kw_i, Kv_i=Kv_i, K_eta=K_eta, K_xi=K_xi,
    )
    return {"leader": leader, "follower": follower}


# ---------------------------------------------------------------------------
# Experiment 3: Integral gains -> real eigenvalues of M_omega, M_v (Sec V-C)
# ---------------------------------------------------------------------------
def gains_real_eig() -> dict:
    Kw_i = -1 / 5 * np.eye(3)
    Kv_i = -2 / 5 * np.eye(3)
    K_xi = -4 / 100 * np.eye(3)
    K_eta = -7 / 100 * np.eye(3)

    leader = ControllerGains(
        Kw_p=K_L_omega_p, Kv_p=K_L_v_p,
        Kw_i=Kw_i, Kv_i=Kv_i, K_eta=K_eta, K_xi=K_xi,
    )
    follower = ControllerGains(
        Kw_p=K_F_omega_p, Kv_p=K_F_v_p,
        Kw_i=Kw_i, Kv_i=Kv_i, K_eta=K_eta, K_xi=K_xi,
    )
    return {"leader": leader, "follower": follower}


EXPERIMENTS = {
    "proportional": gains_proportional,
    "complex_eig": gains_complex_eig,
    "real_eig": gains_real_eig,
}


def get_gains(experiment_name: str) -> dict:
    if experiment_name not in EXPERIMENTS:
        raise KeyError(
            f"Unknown experiment '{experiment_name}'. "
            f"Choose one of {list(EXPERIMENTS)}"
        )
    return EXPERIMENTS[experiment_name]()
