"""Protocol/coordinate/lifecycle checks without installed flight-stack binaries."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dq_control import Quaternion, get_gains, LeaderTrajectory
from simulators.ardupilot import (ArduPilotBackend, Vehicle, ENU_NED, FRD_FLU,
                                 attitude_enu, guided_velocity, write_world)
from simulators.base import SimConfig, VehicleCommand
from simulators.leader_follower import LeaderFollowerSimulation

mavutil = pytest.importorskip("pymavlink.mavutil")


@pytest.mark.parametrize("rpy", [[0, 0, 0], [.4, -.3, 1.2], [-.2, .1, -2.]])
def test_attitude_converts_world_and_body_axes(rpy):
    actual = attitude_enu(*rpy).rotation_matrix()
    expected = ENU_NED @ Quaternion.from_rpy(rpy).rotation_matrix() @ FRD_FLU
    assert np.allclose(actual, expected)


def test_guided_velocity_frame_rate_and_speed_limit():
    state = SimpleNamespace(attitude=Quaternion.from_rpy([.4, .3, 1.]))
    command = VehicleCommand(np.zeros(3), np.zeros(3), [3, 4, 0], [0, .2, .3])
    velocity, rate = guided_velocity(command, state, 2.)
    assert np.allclose(velocity, [1.6, 1.2, 0])
    assert np.isclose(rate, -(.2*np.sin(.4) + .3*np.cos(.4))/np.cos(.3))
    command.target_vel[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        guided_velocity(command, state, 2.)


@pytest.fixture
def plugin(tmp_path):
    directory = tmp_path / "plugin"
    model = directory / "models/iris_with_ardupilot"
    model.mkdir(parents=True)
    (model / "model.sdf").write_text('''<sdf version="1.9"><model name="iris_with_ardupilot">
      <include><uri>model://iris_with_standoffs</uri></include>
      <plugin name="ArduPilotPlugin" filename="ArduPilotPlugin">
        <fdm_port_in>9002</fdm_port_in><lock_step>1</lock_step>
        <control channel="0"><multiplier>838</multiplier></control>
      </plugin></model></sdf>''')
    return directory


def test_generated_world_separates_ports_and_preserves_upstream_model(tmp_path, plugin):
    run = tmp_path / "run"
    run.mkdir()
    world, models = write_world(run, plugin, np.array([[1, 2, 1], [3, 4, 1]]),
                                np.array([[0, 0, .2], [0, 0, -.2]]))
    for i in range(2):
        sdf = ET.parse(models / f"formation_iris_{i}/model.sdf")
        assert sdf.findtext(".//fdm_port_in") == str(9002 + i*10)
        assert sdf.findtext(".//lock_step") == "0"
        assert sdf.findtext(".//control/multiplier") == "838"
        assert sdf.findtext(".//include/uri") == "model://iris_with_standoffs"
    includes = ET.parse(world).findall(".//world/include")
    assert len(includes) == 2
    assert includes[0].findtext("pose") == "1 2 0.195 0 0 0.2"
    assert includes[1].findtext("uri") == "model://formation_iris_1"
    assert ET.parse(plugin / "models/iris_with_ardupilot/model.sdf").findtext(".//lock_step") == "1"


def test_low_resource_world_keeps_imu_without_server_renderer(tmp_path, plugin):
    run = tmp_path / "run"
    run.mkdir()
    path, _ = write_world(run, plugin, np.ones((2, 3)), np.zeros((2, 3)), low_resource=True)
    world = ET.parse(path)
    assert world.findtext(".//physics/max_step_size") == "0.0025"
    assert world.findtext(".//scene/shadows") == "false"
    assert world.find(".//plugin[@name='gz::sim::systems::Imu']") is not None
    assert world.find(".//plugin[@name='gz::sim::systems::Sensors']") is None


class Wire:
    """Decode outbound MAVLink bytes, rather than mocking the send method."""
    def __init__(self):
        self.messages = []
        self.decoder = mavutil.mavlink.MAVLink(None)

    def write(self, data):
        message = self.decoder.parse_char(data)
        if message:
            self.messages.append(message)


class Connection:
    def __init__(self, sysid):
        self.target_system, self.target_component = sysid, 1
        self.wire = Wire()
        self.mav = mavutil.mavlink.MAVLink(self.wire, srcSystem=255, srcComponent=0)
        self.queue = deque()
        self.closed = False

    def recv_match(self, **kwargs):
        return self.queue.popleft() if self.queue else None

    def close(self):
        self.closed = True

    def inject(self, message, src=None):
        encoder = mavutil.mavlink.MAVLink(None, srcSystem=src or self.target_system, srcComponent=1)
        decoder = mavutil.mavlink.MAVLink(None)
        self.queue.append(decoder.parse_char(message.pack(encoder)))


def telemetry(v, ned, milliseconds=20000):
    c = v.connection
    c.inject(mavutil.mavlink.MAVLink_heartbeat_message(2, 3, 128, 4, 4, 3))
    c.inject(mavutil.mavlink.MAVLink_local_position_ned_message(milliseconds, *ned, .1, .2, -.3))
    c.inject(mavutil.mavlink.MAVLink_attitude_message(milliseconds, .2, -.1, 1., .1, .2, .3))


def backend_with_telemetry(monkeypatch):
    backend = ArduPilotBackend(SimConfig(simulator="ardupilot"))
    backend.mavutil = mavutil
    backend.vehicles = [Vehicle(Connection(1), np.array([1., 2., .195])),
                        Vehicle(Connection(2), np.array([5., 6., .195]))]
    for i, v in enumerate(backend.vehicles):
        telemetry(v, [i+1., 2., -1.])
    backend._pump()
    backend.start_time = 20.
    return backend


def test_telemetry_origins_and_body_rates_are_independent(monkeypatch):
    backend = backend_with_telemetry(monkeypatch)
    backend.vehicles[0].offset = np.array([.1, .2, .3])
    states = backend.get_states()
    assert np.allclose(states[0].position, [2.9, 2.8, .895])
    assert np.allclose(states[1].position, [7., 8., 1.195])
    assert np.allclose(states[0].velocity, [.2, .1, .3])
    assert np.allclose(states[0].angular_velocity, [.1, -.2, -.3])


def test_actual_mavlink_commands_target_each_drone(monkeypatch):
    backend = backend_with_telemetry(monkeypatch)
    command = VehicleCommand(np.zeros(3), np.zeros(3), [.1, .2, .3], [0, 0, .2])
    backend.apply_commands([command, command])
    for i, v in enumerate(backend.vehicles):
        message = v.connection.wire.messages[-1]
        assert message.get_type() == "SET_POSITION_TARGET_LOCAL_NED"
        assert message.target_system == i + 1
        assert message.target_component == 1
        assert message.coordinate_frame == mavutil.mavlink.MAV_FRAME_LOCAL_NED
        assert np.allclose([message.vx, message.vy, message.vz], [.2, .1, -.3])
        assert message.type_mask & 7 == 7  # position ignored
        assert message.type_mask & 56 == 0  # all three velocity axes used
        assert message.type_mask & 1024   # yaw angle ignored
        assert message.type_mask & 2048 == 0  # yaw rate used


def test_stale_state_and_mode_change_stop_tracking(monkeypatch):
    backend = backend_with_telemetry(monkeypatch)
    backend.vehicles[0].position_received = time.monotonic() - 4.
    with pytest.raises(RuntimeError, match="telemetry stopped"):
        backend.get_states()
    backend.vehicles[0].position_received = time.monotonic()
    backend.vehicles[1].mode = 9
    command = VehicleCommand(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    with pytest.raises(RuntimeError, match="Guided mode"):
        backend.apply_commands([command, command])


def test_cross_vehicle_telemetry_is_not_accepted(monkeypatch):
    backend = backend_with_telemetry(monkeypatch)
    old = backend.vehicles[0].position.copy()
    c = backend.vehicles[0].connection
    c.inject(mavutil.mavlink.MAVLink_local_position_ned_message(21000, 100, 100, 100, 0, 0, 0), src=2)
    backend._pump()
    assert np.allclose(old, backend.vehicles[0].position)


def test_iris_proximity_stops_commands_before_contact(monkeypatch):
    backend = backend_with_telemetry(monkeypatch)
    backend.vehicles[1].position = backend.vehicles[0].position + np.array([.2, 0, 0])
    command = VehicleCommand(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    with pytest.raises(RuntimeError, match="stopping tracking before contact"):
        backend.apply_commands([command, command])
    assert not any(m.get_type() == "SET_POSITION_TARGET_LOCAL_NED"
                   for v in backend.vehicles for m in v.connection.wire.messages)


def test_rejected_takeoff_reports_error(monkeypatch):
    backend = backend_with_telemetry(monkeypatch)
    vehicle = backend.vehicles[0]
    vehicle.connection.inject(mavutil.mavlink.MAVLink_command_ack_message(22, 2))
    with pytest.raises(RuntimeError, match="rejected command 22"):
        backend._ack_command(vehicle, 22, time.monotonic()+1, 0, 0, 0, 0, 0, 0, 1)


def test_cleanup_lands_and_closes_connections(monkeypatch):
    backend = backend_with_telemetry(monkeypatch)
    connections = [v.connection for v in backend.vehicles]
    for c in connections:
        c.inject(mavutil.mavlink.MAVLink_heartbeat_message(2, 3, 0, 9, 4, 3))
    backend.flying = True
    backend.close()
    for c in connections:
        assert c.closed
        land = [m for m in c.wire.messages if m.get_type() == "SET_MODE"]
        assert land[-1].custom_mode == 9
    assert backend.vehicles == []
    assert backend.elapsed_time is None
    backend.close()  # safe repeated cleanup


def test_landing_timeout_still_releases_connections(monkeypatch):
    backend = backend_with_telemetry(monkeypatch)
    connections = [v.connection for v in backend.vehicles]
    backend.cfg.ardupilot_land_timeout = .01
    with pytest.warns(UserWarning, match="Landing did not finish"):
        backend.close()
    assert all(c.closed for c in connections)


def test_cleanup_only_stops_processes_owned_by_run(tmp_path):
    backend = ArduPilotBackend(SimConfig())
    backend.run_dir = tmp_path
    backend.env = None
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        backend._launch("owned", [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path)
        owned = backend.processes[0]
        backend.close()
        assert owned.poll() is not None
        assert other.poll() is None
        assert backend.files == []
    finally:
        other.terminate()
        other.wait(timeout=3)


def test_startup_tracks_and_lands_with_protocol_peer(tmp_path, plugin, monkeypatch):
    """Exercise reset/run/close using a MAVLink peer, without claiming physics QA."""
    connections = []

    def connect(address, **kwargs):
        c = Connection(len(connections) + 1)
        c.armed, c.mode, c.height = False, 0, 0.
        original_write = c.wire.write

        def write(data):
            original_write(data)
            message = c.wire.messages[-1]
            if message.get_type() == "SET_MODE":
                c.mode = message.custom_mode
                if c.mode == 9:
                    c.armed, c.height = False, 0.
            elif message.get_type() == "COMMAND_LONG":
                if message.command == 400:
                    c.armed = True
                elif message.command == 22:
                    c.height = message.param7
                c.inject(mavutil.mavlink.MAVLink_command_ack_message(message.command, 0))
        c.wire.write = write
        connections.append(c)
        return c

    monkeypatch.setattr(mavutil, "mavlink_connection", connect)
    cfg = SimConfig(simulator="ardupilot", duration_sec=.15, output_folder=str(tmp_path / "results"))
    backend = ArduPilotBackend(cfg)
    backend.mavutil = mavutil
    ap = tmp_path / "ap"
    binary = ap / "build/sitl/bin/arducopter"
    monkeypatch.setattr(backend, "_check_dependencies", lambda: (ap, plugin, binary))
    launches = []
    monkeypatch.setattr(backend, "_launch", lambda name, args, cwd: launches.append((name, args)))
    pump = backend._pump
    tick = 20000

    def publish_and_pump():
        nonlocal tick
        tick += 25
        for c in connections:
            c.inject(mavutil.mavlink.MAVLink_heartbeat_message(2, 3, 128 if c.armed else 0, c.mode, 4, 3))
            c.inject(mavutil.mavlink.MAVLink_local_position_ned_message(tick, 0, 0, -c.height, 0, 0, 0))
            c.inject(mavutil.mavlink.MAVLink_attitude_message(tick, 0, 0, np.pi/2, 0, 0, 0))
        pump()
    monkeypatch.setattr(backend, "_pump", publish_and_pump)
    sim = LeaderFollowerSimulation(get_gains("real_eig"), cfg, LeaderTrajectory(), backend=backend)
    log = sim.run()
    assert len(log["t"]) > 0
    # The flight timestamp is refreshed with telemetry before evaluating targets.
    assert np.isclose(log["t"][0], .025)  # one refresh from this 25-ms protocol peer
    assert np.all(log["t"] < cfg.duration_sec)
    assert np.allclose(log["leader_pos"][:, 2], 1.)
    assert [name for name, _ in launches] == ["gazebo", "sitl_0", "sitl_1"]
    assert launches[0][1][:4] == ["gz", "sim", "-r", "-s"]
    for i, c in enumerate(connections):
        args = launches[i+1][1]
        assert args[args.index("-I")+1] == str(i)
        assert args[args.index("--sysid")+1] == str(i+1)
        kinds = {m.get_type() for m in c.wire.messages}
        assert "SET_POSITION_TARGET_LOCAL_NED" in kinds
        assert c.closed and not c.armed


def test_shared_runner_uses_flight_time_after_takeoff(monkeypatch):
    from simulators.kinematic import KinematicBackend

    class FlightClockBackend(KinematicBackend):
        @property
        def elapsed_time(self):
            return self.clock

        def reset(self, xyz, rpy):
            super().reset(xyz, rpy)
            self.clock = 0.  # Takeoff is finished before this clock starts.

        def step(self):
            result = super().step()
            # Simulate a physics/telemetry interval longer than the requested
            # control period; duration must follow this clock, not step count.
            self.clock += .25
            return result

    cfg = SimConfig(simulator="ardupilot", duration_sec=1., ctrl_freq=48)
    backend = FlightClockBackend(cfg)
    sim = LeaderFollowerSimulation(get_gains("real_eig"), cfg, LeaderTrajectory(), backend=backend)
    log = sim.run()
    assert np.allclose(log["t"], [0, .25, .5, .75])
    assert backend.clock == 1.
