"""Follow the leader's measured track, a fixed arc length behind."""
from collections import deque
import numpy as np
from .dual_quaternion import DualQuaternion
from .course_trajectories import slerp


class PathFollowerTrajectory:
    def __init__(self, distance, start, leader_start, attitude):
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError("follow_distance must be positive")
        self.distance = distance
        self.history = deque([(0., np.array(start, float), attitude),
                              (distance, np.array(leader_start, float), attitude)])
        self.length = distance
        self.previous_position = np.array(start, float)
        self.previous_attitude = attitude

    def update(self, t, leader_pos, leader_attitude, dt, **kwargs):
        position = np.asarray(leader_pos, float)
        increment = np.linalg.norm(position-self.history[-1][1])
        if increment > 1e-6:
            self.length += increment
            self.history.append((self.length, position.copy(), leader_attitude))
        target_length = max(0., self.length-self.distance)
        while len(self.history) > 2 and self.history[1][0] <= target_length:
            self.history.popleft()
        a, b = self.history[0], self.history[1]
        u = np.clip((target_length-a[0])/(b[0]-a[0]), 0., 1.)
        desired = (1-u)*a[1] + u*b[1]
        attitude = slerp(a[2], b[2], float(u))
        velocity = (desired-self.previous_position)/dt
        relative = self.previous_attitude.conj()*attitude
        if relative.scalar < 0:
            relative = -relative
        norm = np.linalg.norm(relative.vec)
        omega = relative.vec*(2*np.arctan2(norm, relative.scalar)/(norm*dt)) if norm > 1e-12 else np.zeros(3)
        self.previous_position = desired.copy()
        self.previous_attitude = attitude
        return DualQuaternion.from_pose(desired, attitude), omega, velocity
