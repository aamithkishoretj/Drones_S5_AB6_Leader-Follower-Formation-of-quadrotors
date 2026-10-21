"""MuJoCo backend for the shared dual-quaternion formation controller.

The dq_control package and LeaderFollowerSimulation are intentionally
unchanged. MuJoCo receives the same VehicleCommand objects as the other
simulation backends.

This backend uses:
    - world-frame translational force for trajectory tracking
    - bounded acceleration
    - strong vertical stabilization
    - stable angular damping
    - bounded commanded yaw tracking

The purpose is to provide a robust MuJoCo simulation of the same
leader-follower lemniscate experiment without changing the controller core.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .base import DroneState, SimConfig, SimulationBackend, VehicleCommand
from .trails import FlightTrails, COLORS


class MuJoCoBackend(SimulationBackend):
    name = "mujoco"

    def __init__(self, cfg: SimConfig):
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError(
                "MuJoCo is required. Install with: py -m pip install mujoco"
            ) from exc

        self.mujoco = mujoco
        self.cfg = cfg
        self.trails = FlightTrails(max_points=500)

        self.model = None
        self.data = None
        self.viewer = None

        self.body_ids: list[int] = []
        self.joint_ids: list[int] = []

        self._last_wall = None

        # ---------------------------------------------------------------
        # Physical parameters
        # ---------------------------------------------------------------

        self.mass = 0.8
        self.gravity = 9.81

        # Position controller.
        #
        # These values are intentionally conservative. The high-level
        # dual-quaternion controller provides the desired trajectory.
        self.pos_kp = np.array([3.5, 3.5, 7.0])
        self.vel_kd = np.array([3.0, 3.0, 4.5])

        # Maximum commanded acceleration.
        self.max_acc_xy = 4.0
        self.max_acc_z = 5.0

        # Maximum force.
        self.max_force_xy = 3.2
        self.max_force_z = 12.0

        # ---------------------------------------------------------------
        # Attitude stabilization
        # ---------------------------------------------------------------
        #
        # Stabilize roll/pitch; track bounded yaw rate with wrapped yaw error.
        self.att_kp = np.array([0.35, 0.35, 0.08])
        self.att_kd = np.array([0.30, 0.30, 0.03])

        self.max_torque = np.array([0.08, 0.08, 0.04])

        # Desired stable attitude.
        self.hold_roll = 0.0
        self.hold_pitch = 0.0


    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    @property
    def model_path(self) -> Path:
        return (
            Path(__file__).resolve().parent
            / "mujoco_models"
            / "leader_follower.xml"
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        initial_xyzs: np.ndarray,
        initial_rpys: np.ndarray,
    ) -> None:

        self.close()

        self.model = self.mujoco.MjModel.from_xml_path(
            str(self.model_path)
        )

        # Use the project's simulation frequency.
        self.model.opt.timestep = 1.0 / float(self.cfg.pyb_freq)

        self.data = self.mujoco.MjData(self.model)

        # Locate the two drone bodies/free joints.
        self.body_ids = [
            int(self.model.body(f"drone{i}").id)
            for i in range(2)
        ]

        self.joint_ids = [
            int(self.model.joint(f"drone{i}_free").id)
            for i in range(2)
        ]

        from dq_control import Quaternion

        for i in range(2):

            jid = self.joint_ids[i]

            qpos_addr = int(self.model.jnt_qposadr[jid])
            qvel_addr = int(self.model.jnt_dofadr[jid])

            xyz = np.asarray(
                initial_xyzs[i],
                dtype=float,
            )

            rpy = np.asarray(
                initial_rpys[i],
                dtype=float,
            )

            quat = Quaternion.from_rpy(rpy).as_array()

            # MuJoCo quaternion convention = w x y z.
            self.data.qpos[qpos_addr:qpos_addr + 3] = xyz

            self.data.qpos[qpos_addr + 3:qpos_addr + 7] = [
                quat[3],
                quat[0],
                quat[1],
                quat[2],
            ]

            self.data.qvel[qvel_addr:qvel_addr + 6] = 0.0


        self.data.xfrc_applied[:] = 0.0

        self.mujoco.mj_forward(
            self.model,
            self.data,
        )

        # ---------------------------------------------------------------
        # GUI
        # ---------------------------------------------------------------

        if self.cfg.gui:

            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False, show_right_ui=False,
            )

            self.viewer.user_scn.flags[self.mujoco.mjtRndFlag.mjRND_SHADOW] = False
            self.viewer.user_scn.flags[self.mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -60
            self.viewer.cam.distance = 6.5

            self.viewer.cam.lookat[:] = np.mean(initial_xyzs, axis=0)

            self.viewer.sync()

        self._last_wall = time.monotonic()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _state(self, i: int) -> DroneState:

        bid = self.body_ids[i]
        jid = self.joint_ids[i]

        qvel_addr = int(
            self.model.jnt_dofadr[jid]
        )

        # MuJoCo xquat is wxyz.
        q = np.asarray(
            self.data.xquat[bid],
            dtype=float,
        )

        from dq_control import Quaternion

        quat = Quaternion.from_array(
            [
                q[1],
                q[2],
                q[3],
                q[0],
            ]
        ).normalized()

        return DroneState(
            position=np.asarray(
                self.data.xpos[bid]
            ).copy(),

            attitude=quat,

            velocity=np.asarray(
                self.data.qvel[qvel_addr:qvel_addr + 3]
            ).copy(),

            angular_velocity=np.asarray(
                self.data.qvel[qvel_addr + 3:qvel_addr + 6]
            ).copy(),

            motor_rpm=np.zeros(4),
        )

    # ------------------------------------------------------------------

    def get_states(self) -> list[DroneState]:

        if self.model is None or self.data is None:
            raise RuntimeError(
                "MuJoCo backend has not been reset"
            )

        return [
            self._state(0),
            self._state(1),
        ]

    # ------------------------------------------------------------------
    # Command application
    # ------------------------------------------------------------------

    def apply_commands(
        self,
        commands: list[VehicleCommand],
    ) -> None:

        states = self.get_states()

        # Clear previous external forces.
        self.data.xfrc_applied[:] = 0.0

        for i, (state, cmd) in enumerate(
            zip(states, commands)
        ):

            # ===========================================================
            # 1. POSITION TRACKING
            # ===========================================================

            target_pos = np.asarray(
                cmd.target_pos,
                dtype=float,
            )

            target_vel = np.asarray(
                cmd.target_vel,
                dtype=float,
            )

            position_error = (
                target_pos - state.position
            )

            velocity_error = (
                target_vel - state.velocity
            )

            acceleration = (
                self.pos_kp * position_error
                + self.vel_kd * velocity_error
            )

            # Limit horizontal acceleration.
            horizontal_norm = np.linalg.norm(
                acceleration[:2]
            )

            if horizontal_norm > self.max_acc_xy:
                acceleration[:2] *= (
                    self.max_acc_xy
                    / horizontal_norm
                )

            # Limit vertical acceleration.
            acceleration[2] = np.clip(
                acceleration[2],
                -self.max_acc_z,
                self.max_acc_z,
            )

            # ===========================================================
            # 2. GRAVITY COMPENSATION
            # ===========================================================

            force = self.mass * acceleration

            force[2] += (
                self.mass * self.gravity
            )

            # Final force limits.
            force[0] = np.clip(
                force[0],
                -self.max_force_xy,
                self.max_force_xy,
            )

            force[1] = np.clip(
                force[1],
                -self.max_force_xy,
                self.max_force_xy,
            )

            force[2] = np.clip(
                force[2],
                0.0,
                self.max_force_z,
            )

            # ===========================================================
            # 3. STABLE ATTITUDE
            # ===========================================================
            #
            # Translation remains force controlled; only yaw follows the
            # high-level pose/rate command, keeping roll/pitch stabilized.
            current_rpy = np.asarray(
                state.attitude.to_rpy(),
                dtype=float,
            )

            desired_rpy = np.array(
                [
                    self.hold_roll,
                    self.hold_pitch,
                    cmd.target_rpy[2],
                ]
            )

            angle_error = (
                desired_rpy - current_rpy
            )

            # Wrap angular errors.
            angle_error = (
                angle_error + np.pi
            ) % (2.0 * np.pi) - np.pi

            # MuJoCo free-joint angular velocity is expressed in the
            # body's local frame. Convert it to world frame before using
            # it as damping for xfrc_applied.
            qmat = np.asarray(
                self.data.xmat[
                    self.body_ids[i]
                ],
                dtype=float,
            ).reshape(3, 3)

            body_angular_velocity = np.asarray(
                state.angular_velocity,
                dtype=float,
            )

            world_angular_velocity = (
                qmat @ body_angular_velocity
            )

            # Only the yaw component is commanded. xfrc_applied torque is
            # world-frame, so transform the body-rate feedforward first.
            desired_world_rate = np.zeros(3)
            desired_world_rate[2] = np.clip((qmat @ cmd.target_rpy_rates)[2], -1.5, 1.5)
            torque = self.att_kp * angle_error + self.att_kd * (desired_world_rate-world_angular_velocity)

            torque = np.clip(
                torque,
                -self.max_torque,
                self.max_torque,
            )

            # ===========================================================
            # 4. APPLY WORLD-FRAME WRENCH
            # ===========================================================

            bid = self.body_ids[i]

            self.data.xfrc_applied[
                bid,
                0:3
            ] = force

            self.data.xfrc_applied[
                bid,
                3:6
            ] = torque

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self) -> list[DroneState]:

        if self.model is None or self.data is None:
            raise RuntimeError(
                "MuJoCo backend has not been reset"
            )

        target_time = (
            self.data.time
            + self.cfg.control_timestep()
        )

        while (
            self.data.time
            + 0.5 * self.model.opt.timestep
            < target_time
        ):

            self.mujoco.mj_step(
                self.model,
                self.data,
            )

        # Update viewer.
        if self.viewer is not None:

            if not self.viewer.is_running():
                raise KeyboardInterrupt(
                    "MuJoCo viewer closed"
                )

            self.viewer.sync()

        # Keep real-time pacing.
        if (
            self.cfg.gui
            and self._last_wall is not None
        ):

            elapsed = (
                time.monotonic()
                - self._last_wall
            )

            remaining = (
                self.cfg.control_timestep()
                - elapsed
            )

            if remaining > 0:
                time.sleep(remaining)

            self._last_wall = time.monotonic()

        return self.get_states()

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self) -> None:

        if self.viewer is not None:

            try:
                self.viewer.close()
            except Exception:
                pass

        self.viewer = None
        self.model = None
        self.data = None
        self.trails = FlightTrails(max_points=500)

    def draw_trails(self, positions):
        if self.viewer is None or not self.cfg.trails:
            return
        self.trails.update(positions)
        with self.viewer.lock():
            scene = self.viewer.user_scn
            scene.ngeom = 0
            for i, points in enumerate(self.trails.points):
                points = list(points)
                for a, b in zip(points, points[1:]):
                    if scene.ngeom >= scene.maxgeom:
                        break
                    geom = scene.geoms[scene.ngeom]
                    self.mujoco.mjv_initGeom(geom, self.mujoco.mjtGeom.mjGEOM_CAPSULE,
                        np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(COLORS[i], dtype=np.float32))
                    self.mujoco.mjv_connector(geom, self.mujoco.mjtGeom.mjGEOM_CAPSULE, .008, a, b)
                    scene.ngeom += 1
