"""Two ArduCopter SITLs in an official ArduPilot Gazebo world; no ROS needed.

The shared dual-quaternion controller supplies world velocity and body angular
velocity. Guided mode consumes velocity and yaw rate; ArduCopter owns roll,
pitch, thrust and motor control. This is not a motor-level controller comparison.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import math
import os
import shutil
import signal
import socket
import subprocess
import time
import uuid
import warnings
import xml.etree.ElementTree as ET

import numpy as np

from dq_control import Quaternion
from .base import DroneState, SimulationBackend
from .dependencies import sibling_repo
from .trails import FlightTrails

# Both are involutions. Attitude also needs the FRD -> FLU body conversion.
ENU_NED = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
FRD_FLU = np.diag([1., -1., -1.])
VELOCITY_YAW_RATE_MASK = 0x05C7  # ignore position, acceleration and yaw angle
GROUND_Z = .195


def attitude_enu(roll, pitch, yaw):
    # R_ENU_FLU = R_ENU_NED R_NED_FRD R_FRD_FLU. Quaternion sandwich
    # composition avoids Euler sign guesses for a tilted vehicle.
    world = Quaternion.from_rpy([math.pi, 0., math.pi / 2.])
    body = Quaternion.from_rpy([math.pi, 0., 0.])
    return (world * Quaternion.from_rpy([roll, pitch, yaw]) * body).normalized()


def guided_velocity(command, state, max_speed):
    velocity = np.asarray(command.target_vel, float)
    rates = np.asarray(command.target_rpy_rates, float)
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(rates)):
        raise ValueError("ArduPilot received non-finite controller output")
    speed = np.linalg.norm(velocity)
    if speed > max_speed:
        velocity = velocity * (max_speed / speed)
    roll, pitch, _ = state.attitude.to_rpy()
    # Controller omega is a body rate, not Euler yaw-dot. NED yaw has
    # the opposite sign from ENU yaw. Limit extreme transient rates.
    yaw_dot = (math.sin(roll) * rates[1] + math.cos(roll) * rates[2]) / max(.1, math.cos(pitch))
    return ENU_NED @ velocity, float(np.clip(-yaw_dot, -1., 1.))


def write_world(directory, plugin_repo, initial_xyzs, initial_rpys, low_resource=False):
    """Reuse upstream Iris physics, with a separate JSON port per drone."""
    model_root = directory / "models"
    model_root.mkdir()
    source = plugin_repo / "models/iris_with_ardupilot/model.sdf"
    for i in range(2):
        model_dir = model_root / f"formation_iris_{i}"
        model_dir.mkdir()
        sdf = ET.parse(source)
        model = sdf.getroot().find("model")
        model.set("name", f"formation_iris_{i}")
        plugin = model.find("plugin[@name='ArduPilotPlugin']")
        if plugin is None:
            raise RuntimeError(f"ArduPilotPlugin missing in {source}")
        plugin.find("fdm_port_in").text = str(9002 + 10 * i)
        # Two independent lockstep SITLs can deadlock a shared physics world.
        # Run the world at real-time factor 1 and monitor telemetry progress.
        plugin.find("lock_step").text = "0"
        sdf.write(model_dir / "model.sdf", encoding="utf-8", xml_declaration=True)
        (model_dir / "model.config").write_text(
            f'<model><name>formation_iris_{i}</name><version>1</version>'
            '<sdf version="1.9">model.sdf</sdf></model>')

    template = Path(__file__).resolve().parents[1] / "ardupilot/world.sdf"
    world = ET.parse(template)
    if low_resource:
        world.find(".//physics/max_step_size").text = "0.0025"
        scene = world.find(".//scene")
        ET.SubElement(scene, "shadows").text = "false"
        for element in world.findall(".//cast_shadows"):
            element.text = "false"
    includes = world.getroot().find("world").findall("include")
    for i, include in enumerate(includes):
        include.find("uri").text = f"model://formation_iris_{i}"
        # Spawn on the floor, then take off. Iris FLU forward is +X in Gazebo.
        p, r = initial_xyzs[i], initial_rpys[i]
        include.find("pose").text = f"{p[0]} {p[1]} {GROUND_Z} 0 0 {r[2]}"
    path = directory / "formation.sdf"
    world.write(path, encoding="utf-8", xml_declaration=True)
    return path, model_root


@dataclass
class Vehicle:
    connection: object
    origin: np.ndarray
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    position: np.ndarray | None = None
    attitude: Quaternion | None = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    boot_time: float = 0.
    position_received: float = 0.
    attitude_received: float = 0.
    heartbeat_received: float = 0.
    armed: bool = False
    mode: int = -1
    acks: dict = field(default_factory=dict)


class ArduPilotBackend(SimulationBackend):
    name = "ardupilot"

    def __init__(self, cfg):
        self.cfg = cfg
        self.vehicles = []
        self.processes = []
        self.files = []
        self.run_dir = None
        self.env = None
        self.mavutil = None
        self.start_time = None
        self.trails = FlightTrails()
        self.trail_worker = None
        self.trail_future = None
        self.last_trail = 0.
        self.last_heartbeat = 0.
        self.flying = False
        self.last_progress = 0.
        self.wait_label = None
        self.last_applied_commands = None

    @property
    def elapsed_time(self):
        if self.start_time is None:
            return None
        return max(0., self.vehicles[0].boot_time - self.start_time)

    def _check_dependencies(self):
        for value in (self.cfg.ardupilot_startup_timeout, self.cfg.ardupilot_land_timeout,
                      self.cfg.ardupilot_max_speed, self.cfg.ardupilot_min_separation):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("ArduPilot timeouts, speed limit and minimum separation must be finite and positive")
        self.cfg.control_timestep()
        ap = Path(self.cfg.ardupilot_path or sibling_repo("ardupilot")).resolve()
        plugin = Path(self.cfg.ardupilot_gazebo_path or sibling_repo("ardupilot_gazebo")).resolve()
        binary = ap / "build/sitl/bin/arducopter"
        missing = []
        if not shutil.which("gz"):
            missing.append("Gazebo Harmonic ('gz' on PATH)")
        if not binary.is_file() or not os.access(binary, os.X_OK):
            missing.append(f"built SITL: {binary}")
        for name in ("copter.parm", "gazebo-iris.parm"):
            if not (ap / "Tools/autotest/default_params" / name).is_file():
                missing.append(f"SITL defaults: {name} in {ap}")
        if not list((plugin / "build").rglob("*ArduPilotPlugin*.so")):
            missing.append(f"built Gazebo plugin: {plugin / 'build'}")
        if not (plugin / "models/iris_with_ardupilot/model.sdf").is_file():
            missing.append(f"Iris model in {plugin}")
        try:
            from pymavlink import mavutil
            self.mavutil = mavutil
        except ImportError:
            missing.append("pymavlink (pip install -r requirements-ardupilot.txt)")
        if missing:
            raise RuntimeError("ArduPilot setup incomplete:\n  - " + "\n  - ".join(missing)
                               + "\nSee ardupilot/README.md or run bash scripts/setup_ardupilot.sh")
        for port in (5760, 5770, 9002, 9012):
            kind = socket.SOCK_STREAM if port < 9000 else socket.SOCK_DGRAM
            with socket.socket(socket.AF_INET, kind) as probe:
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError as exc:
                    raise RuntimeError(f"Port {port} is already in use; stop the other SITL run") from exc
        return ap, plugin, binary

    def _launch(self, name, args, cwd):
        log = (self.run_dir / f"{name}.log").open("w")
        self.files.append(log)
        process = subprocess.Popen(args, cwd=cwd, env=self.env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        self.processes.append(process)

    def reset(self, initial_xyzs, initial_rpys):
        self.close()
        ap, plugin, binary = self._check_dependencies()
        xyz = np.asarray(initial_xyzs, float)
        rpy = np.asarray(initial_rpys, float)
        if xyz.shape != (2, 3) or rpy.shape != (2, 3) or not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(rpy)):
            raise ValueError("ArduPilot requires two finite initial positions and attitudes")
        if np.any(xyz[:, 2] <= GROUND_Z + .2):
            raise ValueError("Initial altitude must exceed 0.395 m for Iris takeoff")
        self.run_dir = Path(self.cfg.output_folder).resolve() / "ardupilot_runtime" / uuid.uuid4().hex[:12]
        self.run_dir.mkdir(parents=True)
        self.env = os.environ.copy()
        self.env["GZ_PARTITION"] = "formation_" + self.run_dir.name
        for key, paths in {
            "GZ_SIM_SYSTEM_PLUGIN_PATH": [plugin / "build"],
            "GZ_SIM_RESOURCE_PATH": [self.run_dir / "models", plugin / "models", plugin / "worlds"],
        }.items():
            self.env[key] = os.pathsep.join(map(str, paths)) + os.pathsep + self.env.get(key, "")
        try:
            world, _ = write_world(self.run_dir, plugin, xyz, rpy, self.cfg.ardupilot_low_resource)
            print(f"[ardupilot] Runtime and diagnostic logs: {self.run_dir}")
            gz_args = ["gz", "sim", "-r"]
            gz_args += ["--render-engine-gui", self.cfg.gazebo_render_engine] if self.cfg.gui else ["-s"]
            if self.cfg.gui:
                gui = Path(__file__).resolve().parents[1] / "ardupilot/gui.config"
                text = gui.read_text().replace("__ENGINE__", self.cfg.gazebo_render_engine)
                # Aim at the formation's initial position, rather than the
                # world's origin six metres from the camera.
                center = np.mean(xyz, axis=0)
                text = text.replace("__CAMERA_POSE__", f"{center[0]-3} {center[1]-3} 3.5 0 0.55 0.7854")
                config = self.run_dir / "gui.config"
                config.write_text(text)
                gz_args += ["--gui-config", str(config)]
                if self.cfg.ardupilot_low_resource:
                    self.env.setdefault("LP_NUM_THREADS", "2")
            self._launch("gazebo", gz_args + [str(world)], self.run_dir)
            defaults = ap / "Tools/autotest/default_params"
            for i in range(2):
                working = self.run_dir / f"vehicle_{i}"
                working.mkdir()
                params = working / "formation.parm"
                # Disable RC-loss only: Python is the GCS, with periodic heartbeat.
                params.write_text("FS_THR_ENABLE 0\n")
                param_files = [defaults / "copter.parm", defaults / "gazebo-iris.parm", params]
                self._launch(f"sitl_{i}", [str(binary), "-I", str(i), "--model", "JSON",
                             "--speedup", "1", "--sysid", str(i + 1), "--defaults",
                             ",".join(map(str, param_files)), "--home", "-35.363262,149.165237,584,90"], working)
            deadline = time.monotonic() + self.cfg.ardupilot_startup_timeout
            for i in range(2):
                while True:
                    self._check_processes()
                    try:
                        connection = self.mavutil.mavlink_connection(
                            f"tcp:127.0.0.1:{5760 + 10*i}", source_system=255, retries=0)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("SITL TCP connection timeout; inspect sitl logs")
                        time.sleep(.1)
                origin = np.array([xyz[i, 0], xyz[i, 1], GROUND_Z])
                self.vehicles.append(Vehicle(connection, origin))
            self._wait(lambda: all(v.heartbeat_received for v in self.vehicles), deadline, "SITL heartbeat")
            for v in self.vehicles:
                for msg_id in (30, 32):  # ATTITUDE and LOCAL_POSITION_NED
                    self._send_long(v, 511, msg_id, 1e6 / self.cfg.ctrl_freq)
            self._wait(lambda: all(v.position is not None and v.attitude is not None for v in self.vehicles),
                       deadline, "EKF local position and attitude")
            # Each EKF's local origin may differ. Calibrate independently while landed.
            for v in self.vehicles:
                v.offset = v.position - v.origin
            for v in self.vehicles:
                v.connection.mav.set_mode_send(v.connection.target_system, 1, 4)  # GUIDED
            self._wait(lambda: all(v.mode == 4 for v in self.vehicles), deadline, "Guided mode")
            # Normal arming checks remain enabled. Retry while EKF/GPS converges.
            next_arm = 0.
            while not all(v.armed for v in self.vehicles):
                if time.monotonic() >= next_arm:
                    for v in self.vehicles:
                        if not v.armed:
                            self._send_long(v, 400, 1)
                    next_arm = time.monotonic() + 2.
                self._wait_once(deadline, "arming (check pre-arm messages in terminal)")
            self.flying = True
            for i, v in enumerate(self.vehicles):
                self._ack_command(v, 22, deadline, 0, 0, 0, 0, 0, 0, xyz[i, 2] - GROUND_Z)
            self._wait(lambda: all(abs((v.position - v.offset)[2] - xyz[i, 2]) < .15
                                   and abs(v.velocity[2]) < .2 for i, v in enumerate(self.vehicles)),
                       deadline, "takeoff")
            self.start_time = self.vehicles[0].boot_time
            self.trails = FlightTrails()
            print(f"[ardupilot] Both drones airborne; tracking for {self.cfg.duration_sec:g} simulation seconds")
        except BaseException:
            self.close()
            raise

    def _check_processes(self):
        for process in self.processes:
            if process.poll() is not None:
                raise RuntimeError(f"Gazebo/SITL exited with code {process.returncode}; inspect {self.run_dir}")

    def _send_long(self, v, command, *params):
        v.connection.mav.command_long_send(v.connection.target_system, v.connection.target_component,
                                          command, 0, *(list(params) + [0.] * (7 - len(params))))

    def _ack_command(self, v, command, deadline, *params):
        v.acks.pop(command, None)
        retry_at = 0.
        while command not in v.acks:
            if time.monotonic() >= retry_at:
                self._send_long(v, command, *params)
                retry_at = time.monotonic() + 2.
            self._wait_once(deadline, f"command {command} acknowledgement")
        if v.acks.pop(command) not in (0, 5):  # accepted or in progress
            raise RuntimeError(f"SITL rejected command {command}; inspect pre-arm/status messages")

    def _pump(self):
        now = time.monotonic()
        for i, v in enumerate(self.vehicles):
            for _ in range(1000):
                msg = v.connection.recv_match(blocking=False)
                if msg is None:
                    break
                if msg.get_srcSystem() != i + 1:
                    continue
                kind = msg.get_type()
                if kind == "HEARTBEAT":
                    v.heartbeat_received = now
                    v.armed = bool(msg.base_mode & 128)
                    v.mode = msg.custom_mode
                elif kind == "LOCAL_POSITION_NED":
                    v.position = v.origin + ENU_NED @ np.array([msg.x, msg.y, msg.z])
                    v.velocity = ENU_NED @ np.array([msg.vx, msg.vy, msg.vz])
                    v.boot_time = msg.time_boot_ms / 1000.
                    v.position_received = now
                elif kind == "ATTITUDE":
                    v.attitude = attitude_enu(msg.roll, msg.pitch, msg.yaw)
                    v.omega = FRD_FLU @ np.array([msg.rollspeed, msg.pitchspeed, msg.yawspeed])
                    v.attitude_received = now
                elif kind == "COMMAND_ACK":
                    v.acks[msg.command] = msg.result
                elif kind == "STATUSTEXT":
                    print(f"[ardupilot {i + 1}] {msg.text}")
        if now - self.last_heartbeat >= 1.:
            for v in self.vehicles:
                v.connection.mav.heartbeat_send(6, 8, 0, 0, 4)  # GCS, invalid autopilot, active
            self.last_heartbeat = now

    def _wait_once(self, deadline, label):
        now = time.monotonic()
        if self.start_time is None and (label != self.wait_label or now - self.last_progress >= 5.):
            print(f"[ardupilot] Waiting for {label}...", flush=True)
            self.wait_label, self.last_progress = label, now
        self._check_processes()
        self._pump()
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {label}; logs: {self.run_dir}")
        time.sleep(.005)

    def _wait(self, predicate, deadline, label):
        while not predicate():
            self._wait_once(deadline, label)

    def get_states(self):
        self._check_processes()
        self._pump()
        now = time.monotonic()
        if len(self.vehicles) != 2:
            raise RuntimeError("ArduPilot backend has not been reset")
        for v in self.vehicles:
            if min(v.position_received, v.attitude_received, v.heartbeat_received) < now - 3.:
                raise RuntimeError("SITL telemetry stopped; ending demonstration")
        return [DroneState(v.position - v.offset, v.attitude, v.velocity, v.omega) for v in self.vehicles]

    def apply_commands(self, commands):
        if len(commands) != 2 or len(self.vehicles) != 2:
            raise ValueError("Expected commands for exactly two initialized SITL vehicles")
        states = self.get_states()
        separation = float(np.linalg.norm(states[0].position - states[1].position))
        if separation < self.cfg.ardupilot_min_separation:
            raise RuntimeError(f"Iris separation {separation:.2f} m is below "
                               f"{self.cfg.ardupilot_min_separation:.2f} m; stopping tracking before contact. "
                               "Use a larger trajectory and a wider following gap (see the ArduPilot demo launchers).")
        if any(not v.armed or v.mode != 4 for v in self.vehicles):
            raise RuntimeError("A drone left armed Guided mode; ending demonstration")
        applied = []
        for v, state, command in zip(self.vehicles, states, commands):
            velocity, yaw_rate = guided_velocity(command, state, self.cfg.ardupilot_max_speed)
            applied.append(np.r_[ENU_NED @ velocity, -yaw_rate])
            v.connection.mav.set_position_target_local_ned_send(
                int(v.boot_time * 1000), v.connection.target_system, v.connection.target_component,
                1, VELOCITY_YAW_RATE_MASK, 0, 0, 0, *velocity, 0, 0, 0, 0, yaw_rate)
        self.last_applied_commands = np.array(applied)

    def step(self):
        before = [v.boot_time for v in self.vehicles]
        # Follow simulator timestamps, so pause/slow graphics cannot run the
        # reference trajectory ahead of the drones. A stall has a wall timeout.
        deadline = time.monotonic() + 3.
        self._wait(lambda: all(v.boot_time - old >= self.cfg.control_timestep() - .001
                               for v, old in zip(self.vehicles, before)), deadline, "simulation progress")
        if time.monotonic() - self.last_progress >= 5.:
            print(f"[ardupilot] Flight {self.elapsed_time:.1f}/{self.cfg.duration_sec:g} s", flush=True)
            self.last_progress = time.monotonic()
        return self.get_states()

    def draw_trails(self, positions):
        if not self.cfg.gui or not self.cfg.trails:
            return
        self.trails.update(positions)
        if time.monotonic() - self.last_trail < .5 or (self.trail_future and not self.trail_future.done()):
            return
        if self.trail_worker is None:
            self.trail_worker = ThreadPoolExecutor(max_workers=1)
        from .gazebo import GazeboBackend
        requests = [self.trails.marker(i) for i in range(2) if len(self.trails.points[i]) > 1]
        self.trail_future = self.trail_worker.submit(GazeboBackend._send_trails, requests, self.env)
        self.last_trail = time.monotonic()

    def close(self):
        try:
            if self.vehicles and (self.flying or any(v.armed for v in self.vehicles)):
                print("[ardupilot] Landing both drones before stopping the simulation")
                for v in self.vehicles:
                    if v.armed:
                        v.connection.mav.set_mode_send(v.connection.target_system, 1, 9)  # LAND
                deadline = time.monotonic() + self.cfg.ardupilot_land_timeout
                self._wait(lambda: all(not v.armed for v in self.vehicles), deadline, "landing and disarming")
        except (Exception, KeyboardInterrupt) as exc:
            warnings.warn(f"Landing did not finish: {exc}. Stopping simulation processes.")
        finally:
            if self.trail_worker:
                self.trail_worker.shutdown(wait=True, cancel_futures=True)
            for v in self.vehicles:
                try:
                    v.connection.close()
                except Exception as exc:
                    warnings.warn(f"MAVLink connection cleanup failed: {exc}")
            for process in reversed(self.processes):
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    except ProcessLookupError:
                        pass
            for handle in self.files:
                handle.close()
            self.vehicles, self.processes, self.files = [], [], []
            self.start_time, self.flying = None, False
            self.trail_worker, self.trail_future = None, None
