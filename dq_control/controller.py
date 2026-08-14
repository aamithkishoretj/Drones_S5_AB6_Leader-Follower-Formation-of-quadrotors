"""
Kinematic dual-quaternion tracking controller -- Theorem 1 of:

    Marciano et al., "Dual Quaternion-Based Control for a Leader-Follower
    Formation of Two Quadrotors," ICUAS 2024.

Implements eq. (5)-(9):

    omega = dq* o omega_d o dq
            + (sgn(dq0) (Kw_p @ dq + eta0 * Kw_i @ eta), 0)                (5,6)

    v     = v_d + R(qd) @ (Kv_p @ dp_b + Kv_i @ xi)                        (7)

    eta_dot = 1/2 * eta o (-|dq0| * Kw_i @ dq + sgn(eta0) * Kh @ eta, 0)   (8)

    xi_dot  = -Kv_i @ dp_b + Kxi @ xi                                     (9)

where dq = P(delta Q) is the attitude error quaternion and dp_b is the
position error expressed in the desired body frame (Sec II-B, eq. 4).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .quaternion import Quaternion
from .dual_quaternion import DualQuaternion


def _sgn(x: float) -> float:
    """sign(.) with sgn(0) := 1, consistent with keeping the quaternion
    double-cover branch continuous (standard convention in dual-quaternion
    attitude control to avoid unwinding)."""
    return 1.0 if x >= 0.0 else -1.0


@dataclass
class ControllerGains:
    """Negative-definite 3x3 gain matrices, see Sec. II-C / V (per vehicle)."""
    Kw_p: np.ndarray   # attitude proportional gain
    Kv_p: np.ndarray   # position proportional gain
    Kw_i: np.ndarray   # attitude integral gain
    Kv_i: np.ndarray   # position integral gain
    K_eta: np.ndarray  # attitude integral forgetting-factor gain
    K_xi: np.ndarray   # position integral forgetting-factor gain

    def __post_init__(self):
        for name in ("Kw_p", "Kv_p", "Kw_i", "Kv_i", "K_eta", "K_xi"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float).reshape(3, 3))


@dataclass
class ControllerState:
    """Integral terms (xi, eta) carried between control steps, eq. (8)-(9)."""
    xi: np.ndarray = field(default_factory=lambda: np.zeros(3))
    eta: Quaternion = field(default_factory=Quaternion.identity)


def pose_error(Q: DualQuaternion, Qd: DualQuaternion):
    """delta Q = Qd* o Q, returns (dq: Quaternion, dp_b: np.ndarray).

    dq       = P(delta Q) = qd* o q                       (eq. 4)
    dp       = p - pd
    dp_b     = qd* o dp o qd  (position error in desired body frame)
    """
    qd = Qd.attitude()
    q = Q.attitude()
    dq = qd.conj() * q

    dp = Q.position() - Qd.position()
    dp_pure = Quaternion.pure(dp)
    dp_b = (qd.conj() * dp_pure * qd).vec
    return dq, dp_b


class KinematicController:
    """Stateful wrapper: holds (xi, eta) and evaluates eq. (5)-(9) each step."""

    def __init__(self, gains: ControllerGains, state: ControllerState | None = None):
        self.gains = gains
        self.state = state or ControllerState()

    def reset(self):
        self.state = ControllerState()

    def compute(
        self,
        Q: DualQuaternion,
        Qd: DualQuaternion,
        omega_d: np.ndarray,
        v_d: np.ndarray,
        dt: float,
    ):
        """Evaluate the control law and Euler-integrate the integral states.

        Parameters
        ----------
        Q, Qd     : current / desired pose dual quaternions
        omega_d   : desired body-frame angular velocity, R^3
        v_d       : desired inertial-frame linear velocity, R^3
        dt        : control timestep, s

        Returns
        -------
        omega_cmd : R^3, commanded body angular velocity
        v_cmd     : R^3, commanded inertial linear velocity
        """
        g = self.gains
        xi = self.state.xi
        eta = self.state.eta

        dq, dp_b = pose_error(Q, Qd)
        dq_vec, dq0 = dq.vec, dq.scalar
        eta_vec, eta0 = eta.vec, eta.scalar
        qd = Qd.attitude()

        # --- eq. (5)-(6): angular velocity command --------------------------
        omega_d_pure = Quaternion.pure(omega_d)
        omega_transport = dq.conj() * omega_d_pure * dq  # dq* o omega_d o dq (stays pure)
        omega_fb = _sgn(dq0) * (g.Kw_p @ dq_vec + eta0 * (g.Kw_i @ eta_vec))
        omega_cmd = omega_transport.vec + omega_fb

        # --- eq. (7): linear velocity command --------------------------------
        R_qd = qd.rotation_matrix()
        v_cmd = v_d + R_qd @ (g.Kv_p @ dp_b + g.Kv_i @ xi)

        # --- eq. (8): eta_dot = 1/2 eta o (-|dq0| Kw_i dq + sgn(eta0) K_eta eta, 0)
        eta_arg_vec = -abs(dq0) * (g.Kw_i @ dq_vec) + _sgn(eta0) * (g.K_eta @ eta_vec)
        eta_arg = Quaternion.pure(eta_arg_vec)
        eta_dot = (eta * eta_arg).scale(0.5)

        # --- eq. (9): xi_dot = -Kv_i dp_b + Kxi xi --------------------------
        xi_dot = -(g.Kv_i @ dp_b) + (g.K_xi @ xi)

        # Euler-integrate integral states
        new_eta = Quaternion.from_array(eta.as_array() + dt * eta_dot.as_array()).normalized()
        new_xi = xi + dt * xi_dot
        self.state = ControllerState(xi=new_xi, eta=new_eta)

        return omega_cmd, v_cmd
