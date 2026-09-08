"""
Common simulator contracts.

The formation controller should not know whether the vehicle is simulated by
PyBullet, Gazebo, MuJoCo, or a flight-stack SITL. Backends expose the same
small state/command interface and own all simulator-specific details.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from dq_control import Quaternion


@dataclass
class DroneState:
    """Normalized simulator-independent vehicle state."""

    position: np.ndarray
    attitude: Quaternion
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    motor_rpm: np.ndarray = field(default_factory=lambda: np.zeros(4))

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=float).reshape(3)
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=float).reshape(3)
        self.motor_rpm = np.asarray(self.motor_rpm, dtype=float).reshape(4)

    def as_pose(self):
        """Return the common dual-quaternion pose representation."""
        from dq_control import DualQuaternion
        return DualQuaternion.from_pose(self.position, self.attitude)


@dataclass
class VehicleCommand:
    """Common high-level command consumed by simulator backends.

    target_pos / target_rpy are the one-control-step integrated pose target
    produced by the dual-quaternion controller. target_vel and target_rpy_rates
    are the corresponding feed-forward signals.
    """

    target_pos: np.ndarray
    target_rpy: np.ndarray
    target_vel: np.ndarray
    target_rpy_rates: np.ndarray

    def __post_init__(self) -> None:
        self.target_pos = np.asarray(self.target_pos, dtype=float).reshape(3)
        self.target_rpy = np.asarray(self.target_rpy, dtype=float).reshape(3)
        self.target_vel = np.asarray(self.target_vel, dtype=float).reshape(3)
        self.target_rpy_rates = np.asarray(self.target_rpy_rates, dtype=float).reshape(3)


@dataclass
class SimConfig:
    """Shared simulator settings plus formation-controller settings."""

    duration_sec: float = 30.0
    pyb_freq: int = 240          # preserved name for backwards compatibility
    ctrl_freq: int = 48
    gui: bool = False
    drone_model: object = None

    x_offset: float = 1.85
    follower_offset_mode: str = "world"
    follower_heading_source: str = "velocity"
    follower_heading_smoothing: float = 0.15
    follower_vel_smoothing: float = 1.0

    output_folder: str = "results"
    simulator: str = "pybullet"

    def control_timestep(self) -> float:
        if self.ctrl_freq <= 0:
            raise ValueError("ctrl_freq must be > 0")
        return 1.0 / float(self.ctrl_freq)


class SimulationBackend(ABC):
    """Minimal interface implemented by every simulator backend."""

    name: str

    @abstractmethod
    def reset(self, initial_xyzs: np.ndarray, initial_rpys: np.ndarray) -> None:
        """Start/reset the simulator with the requested initial vehicle poses."""

    @abstractmethod
    def get_states(self) -> list[DroneState]:
        """Return the latest state for every simulated vehicle."""

    @abstractmethod
    def apply_commands(self, commands: list[VehicleCommand]) -> None:
        """Send one command to each vehicle."""

    @abstractmethod
    def step(self) -> list[DroneState]:
        """Advance the simulator by one controller timestep and return states."""

    @abstractmethod
    def close(self) -> None:
        """Release simulator resources."""
