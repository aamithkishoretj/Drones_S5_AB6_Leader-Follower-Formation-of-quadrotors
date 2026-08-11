"""
Dual quaternion algebra and pose representation, following Sections II-A/B of:

    Marciano et al., "Dual Quaternion-Based Control for a Leader-Follower
    Formation of Two Quadrotors," ICUAS 2024.

A dual quaternion is Q = P(Q) + eps * D(Q), P, D in H, eps^2 = 0.
Unit-norm dual quaternions encode a rigid pose: P(Q) is the attitude
quaternion and D(Q) = 1/2 * p_pure o P(Q), with p in R^3 the position.
"""

from __future__ import annotations
import numpy as np
from .quaternion import Quaternion


class DualQuaternion:
    __slots__ = ("P", "D")

    def __init__(self, principal: Quaternion, dual: Quaternion):
        self.P = principal
        self.D = dual

    # ---- constructors ---------------------------------------------------
    @classmethod
    def from_pose(cls, position, attitude: Quaternion) -> "DualQuaternion":
        """Build unit-norm Q from position p in R^3 and attitude quaternion q.

        D(Q) = 1/2 * p_pure o q   (paper, Sec II-A, right after eq. characterizing
        unit-norm dual quaternions).
        """
        p_pure = Quaternion.pure(position)
        dual = (p_pure * attitude).scale(0.5)
        return cls(attitude, dual)

    @classmethod
    def identity(cls) -> "DualQuaternion":
        return cls(Quaternion.identity(), Quaternion((0, 0, 0), 0.0))

    # ---- pose extraction --------------------------------------------------
    def attitude(self) -> Quaternion:
        return self.P

    def position(self) -> np.ndarray:
        """p = 2 * D(Q) o P(Q)*."""
        return (self.D * self.P.conj()).scale(2.0).vec

    # ---- algebra ------------------------------------------------------------
    def __mul__(self, other: "DualQuaternion") -> "DualQuaternion":
        """(P1 + eps D1)(P2 + eps D2) = P1 P2 + eps (P1 D2 + D1 P2)."""
        p = self.P * other.P
        d = (self.P * other.D) + (self.D * other.P)
        return DualQuaternion(p, d)

    def __add__(self, other: "DualQuaternion") -> "DualQuaternion":
        return DualQuaternion(self.P + other.P, self.D + other.D)

    def conj(self) -> "DualQuaternion":
        """Q* = P(Q)* + eps D(Q)*."""
        return DualQuaternion(self.P.conj(), self.D.conj())

    def norm_check(self) -> float:
        """||P(Q)|| should be 1 for a valid pose dual quaternion."""
        return self.P.norm()

    def __repr__(self) -> str:
        return f"DualQuaternion(P={self.P}, D={self.D})"


class Twist:
    """Dual quaternion twist Omega(omega, v), paper eq. right after (3).

    P(Omega) = omega (pure quaternion, body-frame angular velocity)
    D(Omega) = P(Q)* o v o P(Q)   (v expressed in the inertial frame,
               rotated into the body frame consistent with the paper's
               convention that vehicles receive commands in body frame).
    """

    def __init__(self, omega_body: np.ndarray, v_inertial: np.ndarray, attitude: Quaternion):
        self.omega = Quaternion.pure(omega_body)
        v_pure = Quaternion.pure(v_inertial)
        self.v_body = attitude.conj() * v_pure * attitude  # D(Omega)

    def as_dual_quaternion(self) -> DualQuaternion:
        return DualQuaternion(self.omega, self.v_body)


def pose_derivative(Q: DualQuaternion, omega_body: np.ndarray, v_inertial: np.ndarray) -> DualQuaternion:
    """Q_dot = 1/2 Q o Omega(omega, v), eq. (3).

    Returns a DualQuaternion representing dP/dt + eps dD/dt (NOT itself a
    unit-norm pose dual quaternion -- it is a derivative).
    """
    twist = Twist(omega_body, v_inertial, Q.attitude()).as_dual_quaternion()
    Qdot = Q * twist
    return DualQuaternion(Qdot.P.scale(0.5), Qdot.D.scale(0.5))


def integrate_pose(Q: DualQuaternion, omega_body: np.ndarray, v_inertial: np.ndarray, dt: float) -> DualQuaternion:
    """First-order (Euler) integration of the pose given a body-rate /
    inertial-velocity twist command, then re-normalizes the attitude part
    to guard against unit-norm drift.
    """
    Qdot = pose_derivative(Q, omega_body, v_inertial)
    new_P = (Q.P + Qdot.P.scale(dt)).normalized()
    new_pos = Q.position() + v_inertial * dt
    return DualQuaternion.from_pose(new_pos, new_P)
