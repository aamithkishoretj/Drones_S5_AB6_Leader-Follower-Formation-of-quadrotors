"""
A lightweight quadrotor low-level controller for the Gazebo backend.

The existing PyBullet backend intentionally keeps using gym-pybullet-drones'
DSLPIDControl because that is how the original project was validated. Gazebo
uses this self-contained controller so it does not inherit a PyBullet
dependency just to fly a vehicle inside Gazebo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


def wrap_angle(x: float) -> float:
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class GazeboControllerConfig:
    mass_kg: float = 0.8
    gravity: float = 9.81
    motor_constant: float = 8.54858e-6  # N/(rad/s)^2
    max_motor_rpm: float = 10000.0

    pos_kp: np.ndarray = field(default_factory=lambda: np.array([1.8, 1.8, 4.0], dtype=float))
    vel_kd: np.ndarray = field(default_factory=lambda: np.array([1.2, 1.2, 2.0], dtype=float))

    att_kp: np.ndarray = field(default_factory=lambda: np.array([4.5, 4.5, 2.5], dtype=float))
    att_kd: np.ndarray = field(default_factory=lambda: np.array([0.18, 0.18, 0.25], dtype=float))

    max_tilt_rad: float = 0.55
    max_horizontal_accel: float = 5.0
    max_vertical_accel: float = 7.0


class GazeboQuadrotorController:
    """Cascaded position/attitude controller -> four motor RPM values."""

    def __init__(self, config: GazeboControllerConfig | None = None):
        self.cfg = config or GazeboControllerConfig()
        self.cfg.pos_kp = np.asarray(self.cfg.pos_kp, dtype=float)
        self.cfg.vel_kd = np.asarray(self.cfg.vel_kd, dtype=float)
        self.cfg.att_kp = np.asarray(self.cfg.att_kp, dtype=float)
        self.cfg.att_kd = np.asarray(self.cfg.att_kd, dtype=float)

    @property
    def max_motor_rad_s(self) -> float:
        return self.cfg.max_motor_rpm * 2.0 * np.pi / 60.0

    def compute(self, state, command) -> np.ndarray:
        current_rpy = state.attitude.to_rpy()
        pos_error = command.target_pos - state.position
        vel_error = command.target_vel - state.velocity

        accel = self.cfg.pos_kp * pos_error + self.cfg.vel_kd * vel_error
        accel[:2] = np.clip(accel[:2], -self.cfg.max_horizontal_accel, self.cfg.max_horizontal_accel)
        accel[2] = np.clip(accel[2], -self.cfg.max_vertical_accel, self.cfg.max_vertical_accel)

        # Gravity compensation.
        accel[2] += self.cfg.gravity

        yaw = command.target_rpy[2]
        ax, ay = accel[0], accel[1]

        # Small-angle mapping from desired world acceleration to desired roll/pitch.
        roll_des = (ax * np.sin(yaw) - ay * np.cos(yaw)) / self.cfg.gravity
        pitch_des = (ax * np.cos(yaw) + ay * np.sin(yaw)) / self.cfg.gravity
        roll_des = float(np.clip(roll_des, -self.cfg.max_tilt_rad, self.cfg.max_tilt_rad))
        pitch_des = float(np.clip(pitch_des, -self.cfg.max_tilt_rad, self.cfg.max_tilt_rad))

        target_rpy = np.array([roll_des, pitch_des, command.target_rpy[2]])
        attitude_error = np.array([
            wrap_angle(target_rpy[0] - current_rpy[0]),
            wrap_angle(target_rpy[1] - current_rpy[1]),
            wrap_angle(target_rpy[2] - current_rpy[2]),
        ])

        rate_error = command.target_rpy_rates - state.angular_velocity
        att_cmd = self.cfg.att_kp * attitude_error + self.cfg.att_kd * rate_error

        # Collective thrust.
        thrust = self.cfg.mass_kg * accel[2]
        thrust = float(np.clip(thrust, 0.0, 4.0 * self.cfg.motor_constant * self.max_motor_rad_s**2))
        base_w2 = thrust / (4.0 * self.cfg.motor_constant)

        # Normalize attitude corrections as fractions of collective.
        # These are intentionally bounded to keep the simple controller stable.
        roll_mix = float(np.clip(att_cmd[0], -1.0, 1.0)) * 0.22
        pitch_mix = float(np.clip(att_cmd[1], -1.0, 1.0)) * 0.22
        yaw_mix = float(np.clip(att_cmd[2], -1.0, 1.0)) * 0.10

        # X-configuration mixer.
        w2 = base_w2 * np.array([
            1.0 + roll_mix + pitch_mix - yaw_mix,
            1.0 - roll_mix + pitch_mix + yaw_mix,
            1.0 - roll_mix - pitch_mix - yaw_mix,
            1.0 + roll_mix - pitch_mix + yaw_mix,
        ])
        w2 = np.clip(w2, 0.0, self.max_motor_rad_s**2)
        return np.sqrt(w2) * 60.0 / (2.0 * np.pi)
