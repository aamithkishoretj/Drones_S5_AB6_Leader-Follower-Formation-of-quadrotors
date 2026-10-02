"""Optional reference generators; the dual-quaternion controller is unchanged."""
import numpy as np
from scipy.interpolate import BSpline

from .quaternion import Quaternion
from .dual_quaternion import DualQuaternion
from .trajectories import LeaderTrajectory


def lerp(a, b, u):
    """Euclidean interpolation of points (not quaternion rotations)."""
    return (1 - u) * np.asarray(a) + u * np.asarray(b)


def slerp(a, b, u):
    """Shortest-arc interpolation of unit orientations; q and -q are equivalent."""
    if not 0 <= u <= 1:
        raise ValueError("Interpolation fraction must be in [0, 1]")
    if min(a.norm(), b.norm()) < 1e-12:
        raise ValueError("Cannot interpolate a zero quaternion")
    qa, qb = a.normalized().as_array(), b.normalized().as_array()
    dot = float(qa @ qb)
    if dot < 0:
        qb, dot = -qb, -dot
    if dot > 0.9995:
        return Quaternion.from_array(lerp(qa, qb, u)).normalized()
    angle = np.arccos(np.clip(dot, -1, 1))
    return Quaternion.from_array(
        (np.sin((1-u)*angle)*qa + np.sin(u*angle)*qb) / np.sin(angle)
    ).normalized()


class BSplineTrajectory(LeaderTrajectory):
    """Clamped cubic control-point curve with analytic derivatives.

    A quintic time map stops smoothly at both ends. Two-dimensional points
    are lifted to a constant altitude. Interior control points are generally
    NOT interpolated. Knot values parameterize geometry, not seconds.
    """

    def __init__(self, points, duration=30.0, altitude=1.0, knots=None):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[0] < 4 or points.shape[1] not in (2, 3):
            raise ValueError("Provide at least four 2D or 3D control points")
        if not np.all(np.isfinite(points)) or not np.isfinite(duration) or duration <= 0:
            raise ValueError("Points must be finite and duration positive")
        if points.shape[1] == 2:
            points = np.column_stack([points, np.full(len(points), altitude)])
        self.points = points.copy()
        self.duration = float(duration)
        n = len(points)
        if knots is None:
            knots = np.r_[np.zeros(4), np.linspace(0, 1, n-2)[1:-1], np.ones(4)]
        self.knots = np.asarray(knots, dtype=float)
        if (self.knots.shape != (n+4,) or not np.all(np.isfinite(self.knots))
                or np.any(np.diff(self.knots) < 0)
                or not np.all(self.knots[:4] == 0) or not np.all(self.knots[-4:] == 1)
                or np.any(self.knots[4:-4] <= 0) or np.any(self.knots[4:-4] >= 1)
                or np.any(np.diff(self.knots[4:-4]) <= 0)):
            raise ValueError("Use cubic clamped [0,1] knots with simple interior knots")
        self.spline = BSpline(self.knots, self.points, 3)
        self.d1, self.d2 = self.spline.derivative(1), self.spline.derivative(2)

    def _time(self, t):
        s = np.clip(t / self.duration, 0, 1)
        u = 10*s**3 - 15*s**4 + 6*s**5
        du = 30*s**2*(1-s)**2 / self.duration
        ddu = 60*s*(1-s)*(1-2*s) / self.duration**2
        return u, du, ddu

    def position(self, t):
        return self.spline(self._time(t)[0])

    def velocity(self, t):
        u, du, _ = self._time(t)
        return self.d1(u) * du

    def acceleration(self, t):
        u, du, ddu = self._time(t)
        return self.d2(u)*du**2 + self.d1(u)*ddu

    def yaw(self, t):
        tangent = self.d1(self._time(t)[0])
        return float(np.arctan2(tangent[1], tangent[0]))

    def angular_velocity(self, t, dt=1e-3):
        u, du, _ = self._time(t)
        v, a = self.d1(u), self.d2(u)
        denom = v[:2] @ v[:2]
        rate = (v[0]*a[1] - v[1]*a[0])*du/denom if denom > 1e-12 else 0.0
        return np.array([0., 0., rate])


class InterpolatedPoseTrajectory:
    """LERP positions and SLERP attitudes using a smooth quintic time map."""

    def __init__(self, start, end, start_attitude, end_attitude, duration=30.):
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be positive")
        self.start, self.end = np.asarray(start, float), np.asarray(end, float)
        if self.start.shape != (3,) or self.end.shape != (3,) or not np.all(np.isfinite([self.start, self.end])):
            raise ValueError("start and end must be finite 3D points")
        if min(start_attitude.norm(), end_attitude.norm()) < 1e-12:
            raise ValueError("Attitudes must be nonzero")
        self.q0, self.q1 = start_attitude.normalized(), end_attitude.normalized()
        self.duration = duration
        relative = self.q0.conj() * self.q1
        if relative.scalar < 0:
            relative = -relative
        n = np.linalg.norm(relative.vec)
        self.rotvec = relative.vec * (2*np.arctan2(n, relative.scalar)/n) if n > 1e-12 else np.zeros(3)

    _time = BSplineTrajectory._time

    def position(self, t):
        return lerp(self.start, self.end, self._time(t)[0])

    def velocity(self, t):
        return (self.end-self.start)*self._time(t)[1]

    def acceleration(self, t):
        return (self.end-self.start)*self._time(t)[2]

    def attitude(self, t):
        # Exponential form of SLERP gives exactly consistent angular velocity.
        u = self._time(t)[0]
        return self.q0 * Quaternion.from_rotvec(self.rotvec, np.linalg.norm(self.rotvec)*u)

    def yaw(self, t):
        return float(self.attitude(t).to_rpy()[2])

    def angular_velocity(self, t):
        return self.rotvec*self._time(t)[1]

    def desired_pose(self, t):
        return DualQuaternion.from_pose(self.position(t), self.attitude(t))

    def desired_twist(self, t):
        return self.angular_velocity(t), self.velocity(t)
