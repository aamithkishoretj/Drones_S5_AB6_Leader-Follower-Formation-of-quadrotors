"""
Quaternion algebra following Section II-A of:

    Marciano et al., "Dual Quaternion-Based Control for a Leader-Follower
    Formation of Two Quadrotors," ICUAS 2024.

Convention: a quaternion is q = (q_vec, q0) with q_vec in R^3 the vector
(imaginary) part and q0 in R the scalar (real) part. Internally stored as a
length-4 numpy array [x, y, z, w] (vector-first, scalar-last).

The Hamilton product implemented here matches eq. (1) of the paper:

    p o q = [ S(p) + I*p0    p ] [q ]
            [   -p^T        p0] [q0]

which is algebraically identical to the standard Hamilton product
    (p o q).vec   = p0*q_vec + q0*p_vec + p_vec x q_vec
    (p o q).scalar = p0*q0 - p_vec . q_vec
"""

from __future__ import annotations
import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """S(.) : R^3 -> R^{3x3} skew-symmetric matrix such that S(v) w = v x w."""
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


class Quaternion:
    __slots__ = ("_v",)

    def __init__(self, vec=(0.0, 0.0, 0.0), scalar: float = 1.0):
        vec = np.asarray(vec, dtype=float).reshape(3)
        self._v = np.array([vec[0], vec[1], vec[2], float(scalar)])

    # ---- constructors -----------------------------------------------
    @classmethod
    def from_array(cls, arr) -> "Quaternion":
        """arr = [x, y, z, w] (vector-first, scalar-last)."""
        arr = np.asarray(arr, dtype=float).reshape(4)
        return cls(arr[:3], arr[3])

    @classmethod
    def identity(cls) -> "Quaternion":
        return cls((0.0, 0.0, 0.0), 1.0)

    @classmethod
    def pure(cls, vec) -> "Quaternion":
        """Embed a vector p in R^3 as the pure quaternion (p, 0). See Sec II-A."""
        return cls(vec, 0.0)

    @classmethod
    def from_rotvec(cls, axis, angle: float) -> "Quaternion":
        """Unit quaternion for a rotation of `angle` rad about unit `axis`."""
        axis = np.asarray(axis, dtype=float)
        n = np.linalg.norm(axis)
        if n < 1e-12:
            return cls.identity()
        axis = axis / n
        return cls(axis * np.sin(angle / 2.0), np.cos(angle / 2.0))

    # ---- basic accessors ----------------------------------------------
    @property
    def vec(self) -> np.ndarray:
        return self._v[:3].copy()

    @property
    def scalar(self) -> float:
        return float(self._v[3])

    # backwards-compatible aliases matching the paper's notation (q, q0)
    @property
    def q(self) -> np.ndarray:
        return self.vec

    @property
    def q0(self) -> float:
        return self.scalar

    def as_array(self) -> np.ndarray:
        return self._v.copy()

    # ---- algebra --------------------------------------------------------
    def __mul__(self, other: "Quaternion") -> "Quaternion":
        """Hamilton product p o q, eq. (1)."""
        p, p0 = self.vec, self.scalar
        q, q0 = other.vec, other.scalar
        vec = p0 * q + q0 * p + np.cross(p, q)
        scalar = p0 * q0 - p @ q
        return Quaternion(vec, scalar)

    # alias to mirror the paper's `o` (circle) operator explicitly
    def hamilton(self, other: "Quaternion") -> "Quaternion":
        return self * other

    def __add__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion.from_array(self._v + other._v)

    def __sub__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion.from_array(self._v - other._v)

    def __neg__(self) -> "Quaternion":
        return Quaternion.from_array(-self._v)

    def scale(self, s: float) -> "Quaternion":
        return Quaternion.from_array(self._v * s)

    def conj(self) -> "Quaternion":
        """q* = (-q, q0)."""
        return Quaternion(-self.vec, self.scalar)

    def norm(self) -> float:
        return float(np.linalg.norm(self._v))

    def normalized(self) -> "Quaternion":
        n = self.norm()
        if n < 1e-12:
            return Quaternion.identity()
        return Quaternion.from_array(self._v / n)

    def inv(self) -> "Quaternion":
        n2 = self._v @ self._v
        return Quaternion.from_array(self.conj()._v / n2)

    def rotation_matrix(self) -> np.ndarray:
        """R(q) = (q0^2 - q^T q) I + 2 q q^T + 2 q0 S(q)."""
        q, q0 = self.vec, self.scalar
        I = np.eye(3)
        return (q0 ** 2 - q @ q) * I + 2.0 * np.outer(q, q) + 2.0 * q0 * skew(q)

    def rotate(self, v: np.ndarray) -> np.ndarray:
        """Rotate vector v (R^3) from body to inertial frame: p^i = q o p^b o q*."""
        pv = Quaternion.pure(v)
        return (self * pv * self.conj()).vec

    def to_rpy(self) -> np.ndarray:
        """Roll-pitch-yaw (XYZ intrinsic, matches PyBullet convention) in rad."""
        x, y, z, w = self._v
        # roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        # pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        # yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw])

    @classmethod
    def from_rpy(cls, rpy) -> "Quaternion":
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return cls((x, y, z), w)

    def __repr__(self) -> str:
        return f"Quaternion(vec={self.vec}, scalar={self.scalar:.4f})"
