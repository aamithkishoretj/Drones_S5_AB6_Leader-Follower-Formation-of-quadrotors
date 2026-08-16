"""
Reference trajectories used in Section V of the paper.

Leader: simplified Lemniscate curve
    x_Ld(t) = r_x sin(w_d t) + x0
    y_Ld(t) = r_y sin(2 w_d t) + y0
    z_Ld(t) = z0
with yaw always tangent to the trajectory:
    psi_Ld = atan2(y_Ld_dot, x_Ld_dot)                                  (17)

Follower: replicates the *current measured* leader position with a fixed
offset along x to avoid collisions (eq. 16):
    p_Fd[k] = p_L[k] + [x_offset, 0, 0]
with q_Fd = q_L (follower attitude tracks the leader's current attitude).

Default numeric parameters match the paper (Sec. V, end of page 6):
    r_x = 0.85 m, r_y = 0.65 m, w_d = pi/15 rad/s,
    [x0, y0, z0] = [1.51, -0.27, 1.0] m, x_offset = 1.85 m
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .quaternion import Quaternion
from .dual_quaternion import DualQuaternion


@dataclass
class LemniscateParams:
    r_x: float = 0.85
    r_y: float = 0.65
    w_d: float = np.pi / 15
    x0: float = 1.51
    y0: float = -0.27
    z0: float = 1.0


class LeaderTrajectory:
    """Analytic leader trajectory: position, velocity, attitude (yaw-tangent),
    body angular velocity -- everything the controller needs as (Qd, omega_d, v_d).
    """

    def __init__(self, params: LemniscateParams = LemniscateParams()):
        self.p = params

    def position(self, t: float) -> np.ndarray:
        p = self.p
        return np.array([
            p.r_x * np.sin(p.w_d * t) + p.x0,
            p.r_y * np.sin(2 * p.w_d * t) + p.y0,
            p.z0,
        ])

    def velocity(self, t: float) -> np.ndarray:
        p = self.p
        return np.array([
            p.r_x * p.w_d * np.cos(p.w_d * t),
            2 * p.r_y * p.w_d * np.cos(2 * p.w_d * t),
            0.0,
        ])

    def acceleration(self, t: float) -> np.ndarray:
        p = self.p
        return np.array([
            -p.r_x * p.w_d ** 2 * np.sin(p.w_d * t),
            -4 * p.r_y * p.w_d ** 2 * np.sin(2 * p.w_d * t),
            0.0,
        ])

    def yaw(self, t: float) -> float:
        """psi_Ld = atan2(y_dot, x_dot), eq. (17)."""
        v = self.velocity(t)
        return np.arctan2(v[1], v[0])

    def attitude(self, t: float, dt: float = 1e-3) -> Quaternion:
        """q_L_psi = [0, 0, sin(psi/2), cos(psi/2)]^T (yaw-only attitude, eq. 17)."""
        psi = self.yaw(t)
        return Quaternion((0.0, 0.0, np.sin(psi / 2.0)), np.cos(psi / 2.0))

    def angular_velocity(self, t: float, dt: float = 1e-3) -> np.ndarray:
        """Numerically differentiate yaw to get body-frame omega = [0,0,psi_dot]."""
        psi_p = self.yaw(t + dt)
        psi_m = self.yaw(t - dt)
        psi_dot = (psi_p - psi_m) / (2 * dt)
        return np.array([0.0, 0.0, psi_dot])

    def desired_pose(self, t: float) -> DualQuaternion:
        return DualQuaternion.from_pose(self.position(t), self.attitude(t))

    def desired_twist(self, t: float):
        """Returns (omega_d, v_d) for use directly with KinematicController."""
        return self.angular_velocity(t), self.velocity(t)


class FollowerTrajectory:
    """Follower desired trajectory, defined from the *measured* leader pose
    plus a fixed offset (eq. 16). Since it depends on measurements, velocity
    is obtained by numerically differentiating consecutive leader positions
    (as done in the paper) rather than analytically.
    """

    def __init__(self, x_offset: float = 1.85):
        self.x_offset = x_offset
        self._prev_leader_pos = None
        self._prev_leader_vel = None
        self._prev_t = None

    def reset(self):
        self._prev_leader_pos = None
        self._prev_leader_vel = None
        self._prev_t = None

    def update(self, t: float, leader_pos: np.ndarray, leader_attitude: Quaternion, dt: float):
        """Call once per control step with the *measured* leader state.

        Returns (Qd_follower, omega_d, v_d) suitable for KinematicController.
        """
        offset = np.array([self.x_offset, 0.0, 0.0])
        pos_d = leader_pos + offset

        if self._prev_leader_pos is None:
            vel_d = np.zeros(3)
        else:
            vel_d = (leader_pos - self._prev_leader_pos) / dt

        if self._prev_leader_vel is None:
            acc_d = np.zeros(3)
        else:
            acc_d = (vel_d - self._prev_leader_vel) / dt  # noqa: F841 (kept for future feed-fwd use)

        self._prev_leader_pos = leader_pos.copy()
        self._prev_leader_vel = vel_d.copy()
        self._prev_t = t

        # Follower attitude tracks the leader's current attitude (q_Fd = q_L).
        Qd = DualQuaternion.from_pose(pos_d, leader_attitude)

        # Follower has no independent yaw command -> desired body rate taken as 0
        # (it simply mirrors whatever rotation the leader dual-quaternion encodes
        # through Qd itself; the controller's proportional/integral terms handle
        # any residual rotation tracking).
        omega_d = np.zeros(3)
        return Qd, omega_d, vel_d
