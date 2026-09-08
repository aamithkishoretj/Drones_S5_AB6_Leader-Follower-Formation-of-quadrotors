"""Multi-simulator entry point."""

from .base import DroneState, SimConfig, SimulationBackend, VehicleCommand
from .leader_follower import LeaderFollowerSimulation
from .registry import available_simulators, create_simulator, register_backend

# Real backends
from .gym_pybullet import GymPyBulletDronesBackend
from .gazebo import GazeboBackend

# Explicit placeholders make the architecture visible now; they can be
# replaced independently without changing dq_control/ or the experiment CLI.
class SimulatorNotImplemented(RuntimeError):
    pass


def _not_implemented(name: str):
    def factory(_cfg):
        raise SimulatorNotImplemented(
            f"The '{name}' backend is reserved in the multi-simulator architecture "
            "but is not implemented yet. Add its backend under simulators/ without "
            "changing the formation controller."
        )
    return factory


register_backend("pybullet", GymPyBulletDronesBackend)
register_backend("gym-pybullet-drones", GymPyBulletDronesBackend)
register_backend("gazebo", GazeboBackend)
register_backend("mujoco", _not_implemented("mujoco"))
register_backend("ardupilot", _not_implemented("ardupilot"))

__all__ = [
    "DroneState", "SimConfig", "SimulationBackend", "VehicleCommand",
    "LeaderFollowerSimulation", "available_simulators", "create_simulator",
    "GymPyBulletDronesBackend", "GazeboBackend", "SimulatorNotImplemented",
]
