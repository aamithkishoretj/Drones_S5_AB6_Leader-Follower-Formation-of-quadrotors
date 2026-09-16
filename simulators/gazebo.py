"""
Gazebo Sim Harmonic backend.

Architecture:
  Python formation loop
      -> KinematicController (this repository)
      -> Gazebo backend
      -> ROS 2 actuator_msgs/Actuators
      -> ros_gz_bridge
      -> gz::sim::systems::MulticopterMotorModel
      -> quadrotor physics

State is fed back through Gazebo PosePublisher -> ros_gz_bridge ->
geometry_msgs/PoseStamped. Linear and angular velocity are reconstructed from the
pose stream at the controller rate.

This backend deliberately does not require the gym-pybullet-drones repository.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
import numpy as np

from dq_control import Quaternion
from .base import DroneState, SimConfig, SimulationBackend, VehicleCommand
from .low_level import GazeboQuadrotorController


@dataclass
class GazeboRuntime:
    sim_process: subprocess.Popen | None = None
    bridge_process: subprocess.Popen | None = None
    temp_root: Path | None = None


class _GazeboRosNode:
    """Small rclpy adapter kept isolated from the rest of the Python code."""

    def __init__(self, model_names: list[str]):
        try:
            import rclpy
            from rclpy.node import Node
            from geometry_msgs.msg import PoseStamped
            from actuator_msgs.msg import Actuators
        except ImportError as exc:
            raise RuntimeError(
                "Gazebo backend needs a ROS 2 Python environment with "
                "rclpy, geometry_msgs, and actuator_msgs. "
                "For the supported Jazzy + Harmonic setup, source "
                "/opt/ros/jazzy/setup.bash and install ros-jazzy-ros-gz "
                "and ros-jazzy-actuator-msgs."
            ) from exc

        self.rclpy = rclpy

        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = Node("leader_follower_multi_sim")
        self.model_names = model_names
        self.poses: dict[str, np.ndarray] = {}
        self.pose_stamp: dict[str, float] = {}
        self.pubs = {}

        for name in model_names:
            topic = f"/model/{name}/pose"
            self.node.create_subscription(
                PoseStamped, topic, self._make_pose_callback(name), 10
            )
            self.pubs[name] = self.node.create_publisher(
                Actuators, f"/{name}/gazebo/command/motor_speed", 10
            )

    def _make_pose_callback(self, name: str):
        def callback(msg):
            self.poses[name] = np.array(
                [
                    msg.position.x,
                    msg.position.y,
                    msg.position.z,
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                    msg.orientation.w,
                ],
                dtype=float,
            )
            self.pose_stamp[name] = (
                float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            )
        return callback

    def spin_once(self, timeout_sec: float = 0.0) -> None:
        self.rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def publish_motor_rpm(self, name: str, rpm: np.ndarray) -> None:
        # Import the message class lazily to keep import failures localized.
        from actuator_msgs.msg import Actuators
        msg = Actuators()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        # Gazebo's MulticopterMotorModel expects angular velocity in rad/s;
        # actuator_msgs exposes angular_velocities in rad/s as well.
        msg.velocity = np.asarray(rpm, dtype=float) * 2.0 * np.pi / 60.0
        self.pubs[name].publish(msg)

    def destroy(self) -> None:
        self.node.destroy_node()
        self.rclpy.shutdown()


def _quat_angular_velocity(previous: Quaternion, current: Quaternion, dt: float) -> np.ndarray:
    if previous is None or dt <= 0.0:
        return np.zeros(3)
    dq = previous.conj() * current
    angle = 2.0 * math.atan2(np.linalg.norm(dq.vec), abs(dq.scalar))
    if angle < 1e-9:
        return np.zeros(3)
    axis = dq.vec / (np.linalg.norm(dq.vec) + 1e-12)
    # The pose publisher gives a world pose; this finite-difference estimate is
    # expressed in the current body frame only approximately for small dt.
    return axis * (angle / dt)


class GazeboBackend(SimulationBackend):
    name = "gazebo"

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.runtime = GazeboRuntime()
        self.node: _GazeboRosNode | None = None
        self.states: list[DroneState] = []
        self.previous_poses: list[np.ndarray | None] = [None, None]
        self.previous_times: list[float | None] = [None, None]
        self.controllers = [GazeboQuadrotorController(), GazeboQuadrotorController()]
        self.model_names = ["drone_0", "drone_1"]

    def _require_command(self, command: str) -> str:
        found = shutil.which(command)
        if found is None:
            raise RuntimeError(
                f"'{command}' was not found on PATH. "
                "Make sure Gazebo Sim and/or ROS 2 are installed and sourced."
            )
        return found

    def _write_runtime_world(self, initial_xyzs: np.ndarray, initial_rpys: np.ndarray) -> tuple[Path, Path]:
        template_dir = Path(__file__).resolve().parents[1] / "gazebo"
        source_model = template_dir / "models" / "formation_quadrotor" / "model.sdf.template"
        source_config = template_dir / "models" / "formation_quadrotor" / "model.config"
        world_template = template_dir / "worlds" / "leader_follower.sdf.template"

        if not source_model.exists() or not world_template.exists():
            raise RuntimeError("Gazebo model/world templates are missing from the project.")

        temp_root = Path(tempfile.mkdtemp(prefix="leader_follower_gazebo_"))
        model_root = temp_root / "models"
        model_root.mkdir(parents=True)

        for i, name in enumerate(self.model_names):
            model_dir = model_root / name
            model_dir.mkdir()
            model_sdf = source_model.read_text()
            model_sdf = model_sdf.replace("__ROBOT_NAMESPACE__", name)
            (model_dir / "model.sdf").write_text(model_sdf)
            (model_dir / "model.config").write_text(source_config.read_text())

        p0 = np.asarray(initial_xyzs[0], dtype=float)
        p1 = np.asarray(initial_xyzs[1], dtype=float)
        r0 = np.asarray(initial_rpys[0], dtype=float)
        r1 = np.asarray(initial_rpys[1], dtype=float)

        world = world_template.read_text()
        replacements = {
            "__LEADER_POSE__": f"{p0[0]} {p0[1]} {p0[2]} {r0[0]} {r0[1]} {r0[2]}",
            "__FOLLOWER_POSE__": f"{p1[0]} {p1[1]} {p1[2]} {r1[0]} {r1[1]} {r1[2]}",
        }
        for token, value in replacements.items():
            world = world.replace(token, value)

        world_path = temp_root / "leader_follower.sdf"
        world_path.write_text(world)
        self.runtime.temp_root = temp_root
        return world_path, model_root

    def reset(self, initial_xyzs: np.ndarray, initial_rpys: np.ndarray) -> None:
        gz = self._require_command("gz")
        ros2 = self._require_command("ros2")

        world_path, model_root = self._write_runtime_world(initial_xyzs, initial_rpys)

        env = os.environ.copy()
        existing_resources = env.get("GZ_SIM_RESOURCE_PATH", "")
        resource_paths = [str(model_root)]
        if existing_resources:
            resource_paths.append(existing_resources)
        env["GZ_SIM_RESOURCE_PATH"] = os.pathsep.join(resource_paths)

        if self.cfg.gui:
            sim_cmd = [gz, "sim", "-r", str(world_path)]
        else:
            sim_cmd = [gz, "sim", "-r", "-s", str(world_path)]

        self.runtime.sim_process = subprocess.Popen(
            sim_cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        bridge_config = Path(__file__).resolve().parents[1] / "gazebo" / "config" / "bridge.yaml"
        if not bridge_config.exists():
            raise RuntimeError(f"Bridge configuration missing: {bridge_config}")

        self.runtime.bridge_process = subprocess.Popen(
            [
                ros2, "run", "ros_gz_bridge", "parameter_bridge",
                "--ros-args", "-p", f"config_file:={bridge_config}"
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        self.node = _GazeboRosNode(self.model_names)
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            self.node.spin_once(0.1)
            if all(name in self.node.poses for name in self.model_names):
                break
            if self.runtime.sim_process.poll() is not None:
                raise RuntimeError(
                    f"Gazebo exited immediately with code {self.runtime.sim_process.returncode}. "
                    "Run the same gz command manually to see the model/parser error."
                )
        else:
            raise TimeoutError(
                "Timed out waiting for Gazebo pose topics. "
                "Verify ros_gz_bridge is installed and that Gazebo is running."
            )

        self.states = self._read_states()

    def _read_states(self) -> list[DroneState]:
        if self.node is None:
            raise RuntimeError("Gazebo ROS node is not initialized")

        out: list[DroneState] = []
        for i, name in enumerate(self.model_names):
            data = self.node.poses.get(name)
            if data is None:
                raise RuntimeError(f"No pose received yet for {name}")

            pos = data[:3]
            quat = Quaternion.from_array(data[3:7]).normalized()
            velocity = np.zeros(3)
            angular_velocity = np.zeros(3)

            sim_now = self.node.pose_stamp.get(name, 0.0)
            previous_sim_time = self.previous_times[i]
            if self.previous_poses[i] is not None and previous_sim_time is not None:
                dt = max(1e-6, sim_now - previous_sim_time)
                velocity = (pos - self.previous_poses[i][:3]) / dt
                previous_q = Quaternion.from_array(self.previous_poses[i][3:7]).normalized()
                angular_velocity = _quat_angular_velocity(previous_q, quat, dt)

            previous_rpm = self.states[i].motor_rpm if self.states else np.zeros(4)
            out.append(
                DroneState(
                    position=pos,
                    attitude=quat,
                    velocity=velocity,
                    angular_velocity=angular_velocity,
                    motor_rpm=previous_rpm,
                )
            )
            self.previous_poses[i] = data.copy()
            self.previous_times[i] = sim_now

        return out

    def get_states(self) -> list[DroneState]:
        if self.node is not None:
            self.node.spin_once(0.0)
            if all(name in self.node.poses for name in self.model_names):
                self.states = self._read_states()
        if not self.states:
            raise RuntimeError("Gazebo has not supplied vehicle state yet")
        return self.states

    def apply_commands(self, commands: list[VehicleCommand]) -> None:
        if self.node is None:
            raise RuntimeError("Gazebo backend has not been reset")
        states = self.get_states()
        new_states: list[DroneState] = []
        for i, command in enumerate(commands):
            rpm = self.controllers[i].compute(states[i], command)
            self.node.publish_motor_rpm(self.model_names[i], rpm)
            new_states.append(
                DroneState(
                    position=states[i].position,
                    attitude=states[i].attitude,
                    velocity=states[i].velocity,
                    angular_velocity=states[i].angular_velocity,
                    motor_rpm=rpm,
                )
            )
        self.states = new_states

    def step(self) -> list[DroneState]:
        if self.node is None:
            raise RuntimeError("Gazebo backend has not been reset")

        # Match the requested controller rate without assuming Gazebo's wall
        # clock is exactly equal to the Python loop rate.
        period = self.cfg.control_timestep()
        start = time.monotonic()
        deadline = start + period
        while time.monotonic() < deadline:
            self.node.spin_once(min(0.01, max(0.0, deadline - time.monotonic())))

        return self.get_states()

    def close(self) -> None:
        if self.node is not None:
            try:
                self.node.destroy()
            except Exception:
                pass
            self.node = None

        for proc in (self.runtime.bridge_process, self.runtime.sim_process):
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except Exception:
                    proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except Exception:
                        proc.kill()

        self.runtime.bridge_process = None
        self.runtime.sim_process = None

        if self.runtime.temp_root is not None:
            shutil.rmtree(self.runtime.temp_root, ignore_errors=True)
            self.runtime.temp_root = None
