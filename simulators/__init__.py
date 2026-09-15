"""Multi-simulator entry point."""

from .base import DroneState, SimConfig, SimulationBackend, VehicleCommand
from .leader_follower import LeaderFollowerSimulation
from .registry import available_simulators, create_simulator, register_backend

# Real simulator backends
from .gym_pybullet import GymPyBulletDronesBackend
from .gazebo import GazeboBackend
from .mujoco import MuJoCoBackend


class SimulatorNotImplemented(RuntimeError):
    """Raised when a requested simulator backend is not implemented."""
    pass


def _not_implemented(name: str):
    """Create a placeholder backend for future simulators."""

    def factory(_cfg):
        raise SimulatorNotImplemented(
            f"The '{name}' backend is reserved in the multi-simulator "
            "architecture but is not implemented yet."
        )

    return factory


# ---------------------------------------------------------------------------
# Simulator registry
# ---------------------------------------------------------------------------

# PyBullet
register_backend("pybullet", GymPyBulletDronesBackend)
register_backend("gym-pybullet-drones", GymPyBulletDronesBackend)

# Gazebo
register_backend("gazebo", GazeboBackend)

# MuJoCo
register_backend("mujoco", MuJoCoBackend)

# Future backends
register_backend("ardupilot", _not_implemented("ardupilot"))


__all__ = [
    "DroneState",
    "SimConfig",
    "SimulationBackend",
    "VehicleCommand",
    "LeaderFollowerSimulation",
    "available_simulators",
    "create_simulator",
    "GymPyBulletDronesBackend",
    "GazeboBackend",
    "MuJoCoBackend",
    "SimulatorNotImplemented",
]