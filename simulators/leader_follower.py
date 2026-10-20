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
        if cfg.controller == "learned":
            from dq_control.learned_control import LearnedResponse, LearnedDQController
            model = LearnedResponse.load(cfg.learned_model or f"models/{cfg.simulator}_response.npz",
                                         cfg.simulator, self.ctrl_timestep)
            self.leader_ctrl = LearnedDQController(model, self.ctrl_timestep)
            self.follower_ctrl = LearnedDQController(model, self.ctrl_timestep)
        elif cfg.controller != "analytic":
            raise ValueError("controller must be analytic or learned")
        if cfg.excitation < 0 or not np.isfinite(cfg.excitation):
            raise ValueError("excitation must be finite and nonnegative")
        if cfg.controller == "learned" and cfg.excitation:
            raise ValueError("Training excitation is only supported with the analytic collection controller")

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

    def run(self, stop_event=None, step_callback=None, live_parameters=None):
        cfg = self.cfg
        num_steps = int(cfg.duration_sec * cfg.ctrl_freq)
        rng = np.random.default_rng(cfg.seed)
        excitation = np.zeros((2, 6))
        filters = [PositionKalmanFilter(self.ctrl_timestep, cfg.kalman_measurement_std,
                                       cfg.kalman_acceleration_std) for _ in range(2)] if cfg.kalman else None
        self.backend.reset(self.initial_xyzs, self.initial_rpys)

        log = {k: [] for k in (
            "t",
            "leader_pos", "leader_pos_d", "leader_rpy", "leader_rpy_d",
            "follower_pos", "follower_pos_d", "follower_rpy", "follower_rpy_d",
            "leader_pos_measured", "follower_pos_measured",
            "leader_pos_estimated", "follower_pos_estimated",
            "response_x", "response_u", "response_y",
        )}

        try:
            params = getattr(self.leader_traj, "p", None)
            speed_attr = "w_d" if params is not None and hasattr(params, "w_d") else "w"
            base_speed = float(getattr(params, speed_attr, 1.0)) if params is not None else 1.0
            if not np.isfinite(base_speed) or abs(base_speed) < 1e-8:
                base_speed = 1.0
            phase_time = 0.0
            active = {
                "leader_speed": abs(base_speed),
                "route_radius": float(getattr(params, "r_x", getattr(params, "r", 0.85))),
                "altitude": float(getattr(params, "z0", 1.0)),
                "follow_distance": float(getattr(self.follower_traj, "distance", getattr(self.follower_traj, "x_offset", cfg.follow_distance))),
            }
            previous_reference_position = None
            previous_reference_yaw = None
            for step in range(num_steps):
                if stop_event is not None and stop_event.is_set():
                    break
                if live_parameters is not None:
                    # UI-owned values are read once per control tick so edits
                    # take effect without restarting the simulator.
                    with live_parameters["lock"]:
                        values = dict(live_parameters["values"])
                    # Slew live values to avoid discontinuous references that can
                    # destabilize physics backends. Speed changes alter phase rate,
                    # not the trajectory's absolute phase.
                    targets = {
                        "leader_speed": (0.03, 0.8, 0.35),
                        "route_radius": (0.2, 2.0, 0.5),
                        "altitude": (0.3, 3.0, 0.4),
                        "follow_distance": (0.25, 2.5, 0.5),
                    }
                    for key, (low, high, slew) in targets.items():
                        target = values.get(key)
                        if target is None:
                            continue
                        try:
                            target = float(target)
                        except (TypeError, ValueError):
                            continue
                        if np.isfinite(target):
                            target = float(np.clip(target, low, high))
                            delta = target - active[key]
                            active[key] += float(np.clip(delta, -slew*self.ctrl_timestep, slew*self.ctrl_timestep))
                    if params is not None:
                        if hasattr(params, "r_x"):
                            params.r_x = active["route_radius"]
                        elif hasattr(params, "r"):
                            params.r = active["route_radius"]
                        if hasattr(params, "z0"):
                            params.z0 = active["altitude"]
                    if hasattr(self.follower_traj, "distance"):
                        self.follower_traj.distance = active["follow_distance"]
                    elif hasattr(self.follower_traj, "x_offset"):
                        self.follower_traj.x_offset = active["follow_distance"]
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

                Qd_leader = self.leader_traj.desired_pose(phase_time)
                reference_position = Qd_leader.position()
                reference_yaw = float(Qd_leader.attitude().to_rpy()[2])
                if previous_reference_position is None:
                    omega_d_L, v_d_L = self.leader_traj.desired_twist(phase_time)
                    rate_scale = active["leader_speed"] / abs(base_speed)
                    omega_d_L, v_d_L = omega_d_L * rate_scale, v_d_L * rate_scale
                else:
                    v_d_L = (reference_position - previous_reference_position) / self.ctrl_timestep
                    yaw_delta = (reference_yaw - previous_reference_yaw + np.pi) % (2*np.pi) - np.pi
                    omega_d_L = np.array([0.0, 0.0, yaw_delta / self.ctrl_timestep])
                previous_reference_position = reference_position
                previous_reference_yaw = reference_yaw
                omega_cmd_L, v_cmd_L = self.leader_ctrl.compute(
                    Q_leader, Qd_leader, omega_d_L, v_d_L, self.ctrl_timestep,
                    **({"measured_state": leader_state} if cfg.controller == "learned" else {})
                )

                reference_heading = None
                reference_heading_rate = None
                if cfg.follower_offset_mode == "body":
                    reference_heading = float(self.leader_traj.yaw(phase_time))
                    reference_heading_rate = float(self.leader_traj.angular_velocity(phase_time)[2]) * active["leader_speed"] / abs(base_speed)
                # Advance the virtual trajectory clock after evaluating the
                # current reference. Advancing it above Qd evaluation made the
                # controller chase a one-tick-ahead pose, then integrate that
                # command another tick forward.
                phase_time += self.ctrl_timestep * (active["leader_speed"] / abs(base_speed))

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
                    Q_follower, Qd_follower, omega_d_F, v_d_F, self.ctrl_timestep,
                    **({"measured_state": follower_state} if cfg.controller == "learned" else {})
                )

                from dq_control.learned_control import response_state, response_input
                raw_commands = [(omega_cmd_L, v_cmd_L), (omega_cmd_F, v_cmd_F)]
                if cfg.excitation and step % 12 == 0:
                    excitation = rng.uniform(-1, 1, (2, 6))*cfg.excitation
                commands = []
                training_inputs = []
                for i, (omega, velocity) in enumerate(raw_commands):
                    if cfg.excitation:
                        velocity = velocity + truth[i].attitude.rotation_matrix() @ excitation[i, :3]
                        omega = omega + excitation[i, 3:]*.5
                        velocity = np.clip(velocity, -1.5, 1.5)
                        omega = np.clip(omega, -1.5, 1.5)
                    pos, rpy = self._target_from_twist(states[i].as_pose(), omega, velocity)
                    commands.append(VehicleCommand(pos, rpy, velocity, omega))
                    if step_callback is None:
                        training_inputs.append(response_input(truth[i], omega, velocity))
                training_states = [response_state(s) for s in truth] if step_callback is None else None
                self.backend.apply_commands(commands)
                next_states = self.backend.step()
                if step_callback is not None:
                    step_callback(t, next_states)
                if step_callback is None:
                    log["response_x"].append(training_states)
                    log["response_u"].append(training_inputs)
                    log["response_y"].append([response_state(s) for s in next_states])
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
