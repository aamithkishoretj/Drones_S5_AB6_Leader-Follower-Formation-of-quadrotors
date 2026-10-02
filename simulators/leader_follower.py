"""Simulator-independent leader-follower experiment runner."""

from __future__ import annotations

import os
import numpy as np

from dq_control import DualQuaternion, FollowerTrajectory, KinematicController

from .base import SimConfig, VehicleCommand, DroneState
from dq_control.estimation import PositionKalmanFilter
from .registry import create_simulator


class LeaderFollowerSimulation:
    """Runs the same formation-control loop against any registered backend."""

    def __init__(
        self,
        gains: dict,
        cfg: SimConfig,
        leader_traj,
        backend=None,
    ):
        self.cfg = cfg
        if not np.isfinite(cfg.position_noise_std) or cfg.position_noise_std < 0:
            raise ValueError("position_noise_std must be finite and nonnegative")
        self.leader_traj = leader_traj
        self.follower_traj = FollowerTrajectory(
            x_offset=cfg.x_offset,
            vel_smoothing=cfg.follower_vel_smoothing,
            offset_mode="world" if cfg.follower_offset_mode == "path" else cfg.follower_offset_mode,
            heading_source=cfg.follower_heading_source,
            heading_smoothing=cfg.follower_heading_smoothing,
        )
        if cfg.follower_offset_mode == "body" and cfg.follower_heading_source == "velocity":
            self.follower_traj.reset(initial_heading=self.leader_traj.yaw(0.0))

        self.leader_ctrl = KinematicController(gains["leader"])
        self.follower_ctrl = KinematicController(gains["follower"])
        self.backend = backend or create_simulator(cfg.simulator, cfg)
        self.ctrl_timestep = cfg.control_timestep()

        init_leader = self.leader_traj.position(0.0)
        init_leader_att = self.leader_traj.attitude(0.0)
        init_follower = self.follower_traj.desired_position(init_leader, init_leader_att)
        if cfg.follower_offset_mode == "path":
            from dq_control.path_following import PathFollowerTrajectory
            direction = self.leader_traj.velocity(0.)
            if np.linalg.norm(direction) < 1e-8:
                direction = self.leader_traj.position(self.ctrl_timestep)-init_leader
            if np.linalg.norm(direction) < 1e-12:
                direction = init_leader_att.rotate(np.array([1., 0., 0.]))
            direction = direction / np.linalg.norm(direction)
            init_follower = init_leader - cfg.follow_distance*direction
            self.follower_traj = PathFollowerTrajectory(cfg.follow_distance, init_follower, init_leader, init_leader_att)
        self.initial_xyzs = np.array([init_leader, init_follower])
        self.initial_rpys = np.zeros((2, 3))

    def _target_from_twist(self, Q_current: DualQuaternion, omega_cmd, v_cmd):
        from dq_control import integrate_pose
        Q_target = integrate_pose(Q_current, omega_cmd, v_cmd, self.ctrl_timestep)
        return Q_target.position(), Q_target.attitude().to_rpy()

    def run(self):
        cfg = self.cfg
        num_steps = int(cfg.duration_sec * cfg.ctrl_freq)
        rng = np.random.default_rng(cfg.seed)
        filters = [PositionKalmanFilter(self.ctrl_timestep, cfg.kalman_measurement_std,
                                       cfg.kalman_acceleration_std) for _ in range(2)] if cfg.kalman else None
        self.backend.reset(self.initial_xyzs, self.initial_rpys)

        log = {k: [] for k in (
            "t",
            "leader_pos", "leader_pos_d", "leader_rpy", "leader_rpy_d",
            "follower_pos", "follower_pos_d", "follower_rpy", "follower_rpy_d",
            "leader_pos_measured", "follower_pos_measured",
            "leader_pos_estimated", "follower_pos_estimated",
        )}

        try:
            for step in range(num_steps):
                t = step * self.ctrl_timestep
                states = self.backend.get_states()
                truth = states
                if cfg.trails:
                    self.backend.draw_trails([state.position for state in truth])
                measurements = [s.position + rng.normal(0, cfg.position_noise_std, 3) for s in states]
                if cfg.kalman or cfg.position_noise_std:
                    states = []
                    for i, state in enumerate(truth):
                        pos, vel = filters[i].update(measurements[i]) if filters else (measurements[i], state.velocity)
                        states.append(DroneState(pos, state.attitude, vel, state.angular_velocity, state.motor_rpm))
                leader_state, follower_state = states[0], states[1]

                Q_leader = leader_state.as_pose()
                Q_follower = follower_state.as_pose()

                Qd_leader = self.leader_traj.desired_pose(t)
                omega_d_L, v_d_L = self.leader_traj.desired_twist(t)
                omega_cmd_L, v_cmd_L = self.leader_ctrl.compute(
                    Q_leader, Qd_leader, omega_d_L, v_d_L, self.ctrl_timestep
                )
                target_pos_L, target_rpy_L = self._target_from_twist(
                    Q_leader, omega_cmd_L, v_cmd_L
                )

                reference_heading = None
                reference_heading_rate = None
                if cfg.follower_offset_mode == "body":
                    reference_heading = float(self.leader_traj.yaw(t))
                    reference_heading_rate = float(self.leader_traj.angular_velocity(t)[2])

                Qd_follower, omega_d_F, v_d_F = self.follower_traj.update(
                    t,
                    leader_state.position,
                    leader_state.attitude,
                    self.ctrl_timestep,
                    leader_velocity=leader_state.velocity,
                    leader_angular_velocity=leader_state.angular_velocity,
                    reference_heading=reference_heading,
                    reference_heading_rate=reference_heading_rate,
                )
                omega_cmd_F, v_cmd_F = self.follower_ctrl.compute(
                    Q_follower, Qd_follower, omega_d_F, v_d_F, self.ctrl_timestep
                )
                target_pos_F, target_rpy_F = self._target_from_twist(
                    Q_follower, omega_cmd_F, v_cmd_F
                )

                self.backend.apply_commands([
                    VehicleCommand(target_pos_L, target_rpy_L, v_cmd_L, omega_cmd_L),
                    VehicleCommand(target_pos_F, target_rpy_F, v_cmd_F, omega_cmd_F),
                ])
                self.backend.step()

                log["t"].append(t)
                log["leader_pos"].append(truth[0].position.copy())
                log["leader_pos_d"].append(Qd_leader.position())
                log["leader_rpy"].append(leader_state.attitude.to_rpy())
                log["leader_rpy_d"].append(Qd_leader.attitude().to_rpy())
                log["follower_pos"].append(truth[1].position.copy())
                log["follower_pos_d"].append(Qd_follower.position())
                log["follower_rpy"].append(follower_state.attitude.to_rpy())
                log["follower_rpy_d"].append(Qd_follower.attitude().to_rpy())
                for i, name in enumerate(("leader", "follower")):
                    log[name + "_pos_measured"].append(measurements[i].copy())
                    log[name + "_pos_estimated"].append(states[i].position.copy())
        finally:
            self.backend.close()

        return {k: np.asarray(v) for k, v in log.items()}
