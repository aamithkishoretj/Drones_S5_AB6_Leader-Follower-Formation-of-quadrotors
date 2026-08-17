"""
Reference trajectories used in Section V of the paper, plus custom shapes.

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

Also included: PotatoChipTrajectory, a "Pringle"/hyperbolic-paraboloid-style
saddle curve -- the leader traces a circle in (x, y) while z oscillates at
twice the angular rate and out of phase with the radius direction, i.e.
z(theta) = z0 + z_amp * cos(k * theta) with k=2. This is the standard
parametric trick for a saddle surface restricted to a circular boundary
(z ~ x^2 - y^2 on the circle x=r cos(theta), y=r sin(theta)), which is
exactly the shape of a potato chip / Pringle.

Every trajectory class in this file implements the same six-method contract
used throughout the codebase, so any of them can be passed as `leader_traj=`
to `envs.LeaderFollowerSim` interchangeably:

    position(t) -> np.array([x, y, z])
    velocity(t) -> np.array([vx, vy, vz])
    acceleration(t) -> np.array([ax, ay, az])
    yaw(t) -> float (rad)
    attitude(t) -> Quaternion
    angular_velocity(t) -> np.array([wx, wy, wz])
    desired_pose(t) -> DualQuaternion
    desired_twist(t) -> (omega_d, v_d)
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
        """Exact tangent-heading yaw rate, avoiding atan2 +/-pi wrap spikes.

        For psi = atan2(vy, vx),
            psi_dot = (vx*ay - vy*ax) / (vx^2 + vy^2).
        """
        v = self.velocity(t)
        a = self.acceleration(t)
        denom = float(v[0] ** 2 + v[1] ** 2)
        if denom < 1e-12:
            return np.zeros(3)
        psi_dot = (v[0] * a[1] - v[1] * a[0]) / denom
        return np.array([0.0, 0.0, psi_dot])

    def desired_pose(self, t: float) -> DualQuaternion:
        return DualQuaternion.from_pose(self.position(t), self.attitude(t))

    def desired_twist(self, t: float):
        """Returns (omega_d, v_d) for use directly with KinematicController."""
        return self.angular_velocity(t), self.velocity(t)


@dataclass
class PotatoChipParams:
    r: float = 0.85           # radius of the circular footprint in (x, y), m
    w: float = np.pi / 15     # angular speed around the circle, rad/s
    z_amp: float = 0.35       # peak height of the saddle ripple, m
    k: int = 2                # saddle lobes per revolution (2 = classic Pringle/chip shape)
    phase: float = 0.0        # phase offset of the z ripple relative to theta, rad
    x0: float = 1.51
    y0: float = -0.27
    z0: float = 1.0


class PotatoChipTrajectory:
    """Leader traces a circle in (x, y) while z rides a cos(k*theta) ripple,
    tracing out a saddle ("Pringle" / potato-chip) shaped 3D curve.

        theta(t) = w t
        x(t) = x0 + r cos(theta)
        y(t) = y0 + r sin(theta)
        z(t) = z0 + z_amp cos(k theta + phase)

    Same six-method contract as LeaderTrajectory, so it's a drop-in
    replacement anywhere a leader_traj is accepted.
    """

    def __init__(self, params: PotatoChipParams = PotatoChipParams()):
        self.p = params

    def _theta(self, t: float) -> float:
        return self.p.w * t

    def position(self, t: float) -> np.ndarray:
        p = self.p
        th = self._theta(t)
        return np.array([
            p.x0 + p.r * np.cos(th),
            p.y0 + p.r * np.sin(th),
            p.z0 + p.z_amp * np.cos(p.k * th + p.phase),
        ])

    def velocity(self, t: float) -> np.ndarray:
        p = self.p
        th = self._theta(t)
        return np.array([
            -p.r * p.w * np.sin(th),
            p.r * p.w * np.cos(th),
            -p.z_amp * p.k * p.w * np.sin(p.k * th + p.phase),
        ])

    def acceleration(self, t: float) -> np.ndarray:
        p = self.p
        th = self._theta(t)
        return np.array([
            -p.r * p.w ** 2 * np.cos(th),
            -p.r * p.w ** 2 * np.sin(th),
            -p.z_amp * (p.k * p.w) ** 2 * np.cos(p.k * th + p.phase),
        ])

    def yaw(self, t: float) -> float:
        """Tangent heading in the horizontal (x, y) plane, same convention as
        LeaderTrajectory.yaw -- the z ripple doesn't drive yaw, only xy motion does."""
        v = self.velocity(t)
        return np.arctan2(v[1], v[0])

    def attitude(self, t: float, dt: float = 1e-3) -> Quaternion:
        psi = self.yaw(t)
        return Quaternion((0.0, 0.0, np.sin(psi / 2.0)), np.cos(psi / 2.0))

    def angular_velocity(self, t: float, dt: float = 1e-3) -> np.ndarray:
        """Exact tangent-heading yaw rate, avoiding atan2 +/-pi wrap spikes."""
        v = self.velocity(t)
        a = self.acceleration(t)
        denom = float(v[0] ** 2 + v[1] ** 2)
        if denom < 1e-12:
            return np.zeros(3)
        psi_dot = (v[0] * a[1] - v[1] * a[0]) / denom
        return np.array([0.0, 0.0, psi_dot])

    def desired_pose(self, t: float) -> DualQuaternion:
        return DualQuaternion.from_pose(self.position(t), self.attitude(t))

    def desired_twist(self, t: float):
        return self.angular_velocity(t), self.velocity(t)


class FollowerTrajectory:
    """Follower desired trajectory, defined from the *measured* leader pose
    plus an offset (generalizes eq. 16). Since it depends on measurements,
    velocity is obtained by numerically differentiating the resulting desired
    follower position (as done in the paper, extended to also capture the
    motion induced by a rotating offset -- see offset_mode='body' below).
    An optional exponential smoothing filter reduces the noise differentiation
    introduces.

    offset_mode:
      'world' (paper default, eq. 16): pos_Fd = p_L + [x_offset, 0, 0], a
          FIXED offset in the world frame. This matches the paper exactly and
          works well for trajectories that predominantly move along +x (like
          the lemniscate), but for a curving/circular path (e.g. potato_chip)
          it does NOT keep the follower "behind" the leader -- as the leader's
          heading rotates, a world-frame offset ends up beside or ahead of it.
      'body': pos_Fd = p_L + R(heading) @ [-x_offset, 0, 0], i.e. the offset
          points opposite to the leader's direction of travel, keeping the
          follower trailing directly behind it regardless of trajectory shape.
          Where "heading" comes from is controlled by heading_source below.

    heading_source (only used when offset_mode='body'):
      'velocity' (default): heading = atan2(vy, vx) from a smoothed finite-
          difference estimate of the leader's own MEASURED position over time.
          This is robust to the leader's yaw-*tracking* error: it reflects
          where the leader is actually, geometrically going, independent of
          how well its onboard attitude controller is tracking a commanded
          yaw. This matters because yaw tracking can lag substantially
          (observed: several seconds of phase lag in practice), and rotating
          a large offset vector (~1-2 m) by an erroneous yaw angle turns a
          modest angular error into a large positional error for the
          follower's reference -- which then destabilizes its own tracking.
      'attitude': heading = yaw extracted from the leader's measured attitude
          quaternion (roll/pitch discarded so transient tilting for
          translational control doesn't contaminate the trailing direction).
          Simpler, but sensitive to the leader's own yaw-tracking error as
          described above -- prefer 'velocity' unless you have a specific
          reason not to.
    """

    def __init__(
        self,
        x_offset: float = 1.85,
        vel_smoothing: float = 1.0,
        offset_mode: str = "world",
        heading_source: str = "velocity",
        heading_smoothing: float = 0.15,
    ):
        """vel_smoothing in (0, 1]: 1.0 = raw finite difference (paper default),
        lower values trade responsiveness for a smoother (less noisy) velocity
        estimate for the follower's own translational feedforward.
        heading_smoothing in (0, 1]: smoothing applied specifically to the
        velocity-derived heading estimate used in 'body' mode -- lower values
        reject more high-frequency noise at the cost of responsiveness. Kept
        separate from vel_smoothing because the offset radius amplifies
        heading noise into position error more than translational noise does."""
        if offset_mode not in ("world", "body"):
            raise ValueError(f"offset_mode must be 'world' or 'body', got {offset_mode!r}")
        if heading_source not in ("velocity", "attitude"):
            raise ValueError(f"heading_source must be 'velocity' or 'attitude', got {heading_source!r}")
        self.x_offset = x_offset
        self.vel_smoothing = vel_smoothing
        self.offset_mode = offset_mode
        self.heading_source = heading_source
        self.heading_smoothing = heading_smoothing
        self._prev_pos_d = None
        self._prev_vel_d = None
        self._prev_t = None
        self._prev_leader_pos_for_heading = None
        self._leader_vel_filt = None
        self._heading = 0.0

    def reset(self, initial_heading: float | None = None):
        self._prev_pos_d = None
        self._prev_vel_d = None
        self._prev_t = None
        self._prev_leader_pos_for_heading = None
        self._leader_vel_filt = None
        # For body-frame offsets, initialize the trailing direction from the
        # trajectory geometry when available. This is important at t=0 for
        # circular/saddle trajectories because the simulator can report zero
        # velocity immediately after reset even though the analytic trajectory
        # already has a well-defined tangent direction.
        self._heading = float(initial_heading) if initial_heading is not None else 0.0

    @staticmethod
    def _yaw_from_attitude(q: Quaternion) -> float:
        return q.to_rpy()[2]

    def _update_heading_from_velocity(self, leader_pos: np.ndarray, dt: float):
        """Smoothed atan2(vy, vx) of the leader's own measured (x, y) motion."""
        if self._prev_leader_pos_for_heading is None:
            raw_vel_xy = np.zeros(2)
        else:
            raw_vel_xy = (leader_pos[:2] - self._prev_leader_pos_for_heading[:2]) / dt

        if self._leader_vel_filt is None:
            self._leader_vel_filt = raw_vel_xy
        else:
            a = self.heading_smoothing
            self._leader_vel_filt = a * raw_vel_xy + (1 - a) * self._leader_vel_filt

        self._prev_leader_pos_for_heading = leader_pos.copy()

        speed = np.linalg.norm(self._leader_vel_filt)
        if speed > 1e-3:  # hold last known heading while nearly stationary (avoid atan2 noise at rest)
            self._heading = float(np.arctan2(self._leader_vel_filt[1], self._leader_vel_filt[0]))

    def desired_position(self, leader_pos: np.ndarray, leader_attitude: Quaternion, heading: float = None) -> np.ndarray:
        """pos_Fd for the current instant, given the *measured* leader pose.

        `heading` (radians) overrides the offset direction directly when given
        -- used internally by update() to pass the velocity-derived heading.
        If omitted, falls back to the yaw extracted from leader_attitude
        (still yaw-only, i.e. roll/pitch-safe, but see heading_source docs
        above for why the velocity-derived heading is preferred in practice).
        """
        if self.offset_mode == "body":
            h = heading if heading is not None else self._yaw_from_attitude(leader_attitude)
            yaw_only = Quaternion((0.0, 0.0, np.sin(h / 2.0)), np.cos(h / 2.0))
            offset = yaw_only.rotate(np.array([-self.x_offset, 0.0, 0.0]))
        else:
            offset = np.array([self.x_offset, 0.0, 0.0])
        return leader_pos + offset

    def update(
        self,
        t: float,
        leader_pos: np.ndarray,
        leader_attitude: Quaternion,
        dt: float,
        leader_velocity: np.ndarray | None = None,
        leader_angular_velocity: np.ndarray | None = None,
        reference_heading: float | None = None,
        reference_heading_rate: float | None = None,
    ):
        """Call once per control step with the measured leader state.

        When simulator velocities are available they are used directly rather
        than differentiating sampled positions. This gives a better first-step
        heading estimate and avoids unnecessary numerical noise.

        Returns (Qd_follower, omega_d, v_d) suitable for KinematicController.
        """
        offset_vel = np.zeros(3)

        if self.offset_mode == "body":
            if reference_heading is not None:
                # For a known analytic leader trajectory (e.g. potato_chip),
                # use its exact tangent heading rather than noisy simulator
                # velocity. A 1.85 m formation offset amplifies tiny heading
                # errors into large false follower velocities.
                self._heading = float(reference_heading)
                if reference_heading_rate is not None:
                    hdot = float(reference_heading_rate)
                    offset_vel = np.array([
                        self.x_offset * np.sin(self._heading) * hdot,
                        -self.x_offset * np.cos(self._heading) * hdot,
                        0.0,
                    ])
            elif self.heading_source == "velocity":
                if leader_velocity is not None:
                    lv = np.asarray(leader_velocity, dtype=float).reshape(3)
                    xy_speed = float(np.linalg.norm(lv[:2]))
                    if xy_speed > 1e-3:
                        raw_xy = lv[:2]
                        if self._leader_vel_filt is None:
                            self._leader_vel_filt = raw_xy.copy()
                        else:
                            a = self.heading_smoothing
                            self._leader_vel_filt = a * raw_xy + (1.0 - a) * self._leader_vel_filt
                        self._heading = float(np.arctan2(self._leader_vel_filt[1], self._leader_vel_filt[0]))
                    self._prev_leader_pos_for_heading = leader_pos.copy()
                else:
                    self._update_heading_from_velocity(leader_pos, dt)
            else:
                self._heading = self._yaw_from_attitude(leader_attitude)

            pos_d = self.desired_position(leader_pos, leader_attitude, heading=self._heading)
        else:
            pos_d = self.desired_position(leader_pos, leader_attitude)

        # Build desired translational velocity. For a known analytic body-frame
        # offset, use the exact derivative of the rotating offset; this avoids
        # finite-difference spikes at startup. Otherwise retain the existing
        # measurement-based differentiation used by the paper-style follower.
        if self.offset_mode == "body" and reference_heading is not None and leader_velocity is not None:
            vel_d = np.asarray(leader_velocity, dtype=float).reshape(3) + offset_vel
        elif self._prev_pos_d is None:
            vel_d = np.zeros(3)
        else:
            raw_vel = (pos_d - self._prev_pos_d) / dt
            if self._prev_vel_d is None:
                vel_d = raw_vel
            else:
                a = self.vel_smoothing
                vel_d = a * raw_vel + (1 - a) * self._prev_vel_d

        self._prev_pos_d = pos_d.copy()
        self._prev_vel_d = vel_d.copy()
        self._prev_t = t

        # Follower attitude tracks the leader's current attitude (q_Fd = q_L).
        Qd = DualQuaternion.from_pose(pos_d, leader_attitude)

        # Since the desired follower attitude is q_L, its feed-forward angular
        # velocity should also follow the leader. Setting omega_d=0 would force
        # the follower to chase a moving attitude using feedback only.
        omega_d = (
            np.asarray(leader_angular_velocity, dtype=float).reshape(3)
            if leader_angular_velocity is not None
            else np.zeros(3)
        )
        return Qd, omega_d, vel_d