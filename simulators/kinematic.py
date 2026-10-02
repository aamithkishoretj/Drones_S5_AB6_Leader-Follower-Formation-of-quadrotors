"""Ideal twist-actuated reference backend, not a quadrotor physics engine."""
from dq_control import Quaternion, integrate_pose
from .base import SimulationBackend, DroneState


class KinematicBackend(SimulationBackend):
    name = "kinematic"

    def __init__(self, cfg):
        self.dt = cfg.control_timestep()
        self.states = []
        self.commands = None

    def reset(self, initial_xyzs, initial_rpys):
        self.states = [DroneState(p.copy(), Quaternion.from_rpy(r)) for p, r in zip(initial_xyzs, initial_rpys)]
        self.commands = None

    def get_states(self):
        return self.states

    def apply_commands(self, commands):
        if len(commands) != len(self.states):
            raise ValueError("One command per vehicle required")
        self.commands = commands

    def step(self):
        if self.commands is None:
            raise RuntimeError("Apply commands before stepping")
        updated = []
        for state, cmd in zip(self.states, self.commands):
            pose = integrate_pose(state.as_pose(), cmd.target_rpy_rates, cmd.target_vel, self.dt)
            updated.append(DroneState(pose.position(), pose.attitude(), cmd.target_vel.copy(), cmd.target_rpy_rates.copy()))
        self.states = updated
        return self.states

    def close(self):
        self.commands = None
