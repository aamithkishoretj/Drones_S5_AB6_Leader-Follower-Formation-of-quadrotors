import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simulators import available_simulators
from simulators.low_level import GazeboQuadrotorController
from simulators.base import SimConfig


def test_simulator_registry_has_requested_backends():
    names = available_simulators()
    assert "pybullet" in names
    assert "gazebo" in names
    assert "mujoco" in names
    assert "ardupilot" in names


def test_gazebo_low_level_controller_returns_four_rpm_values():
    from dq_control import Quaternion
    from simulators.base import DroneState, VehicleCommand

    controller = GazeboQuadrotorController()
    state = DroneState(
        position=np.zeros(3),
        attitude=Quaternion.identity(),
        velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
    )
    cmd = VehicleCommand(
        target_pos=np.array([0.0, 0.0, 1.0]),
        target_rpy=np.zeros(3),
        target_vel=np.zeros(3),
        target_rpy_rates=np.zeros(3),
    )
    rpm = controller.compute(state, cmd)
    assert rpm.shape == (4,)
    assert np.all(np.isfinite(rpm))
    assert np.all(rpm >= 0)
    assert np.max(rpm) <= controller.cfg.max_motor_rpm + 1e-9


def test_gazebo_sdf_template_and_bridge_yaml_are_parseable():
    sdf = Path(_ROOT) / "gazebo/models/formation_quadrotor/model.sdf.template"
    bridge = Path(_ROOT) / "gazebo/config/bridge.yaml"
    world = Path(_ROOT) / "gazebo/worlds/leader_follower.sdf.template"

    # XML parser catches malformed tags. Placeholders are plain text in valid XML.
    ET.parse(sdf)
    ET.parse(world)

    data = yaml.safe_load(bridge.read_text())
    assert isinstance(data, list)
    assert len(data) == 4
    assert {item["direction"] for item in data} == {"GZ_TO_ROS", "ROS_TO_GZ"}


def test_default_control_timestep():
    assert np.isclose(SimConfig().control_timestep(), 1 / 48)


def test_gazebo_physics_and_motor_geometry():
    world = ET.parse(Path(_ROOT) / "gazebo/worlds/leader_follower.sdf.template")
    assert world.find(".//plugin[@name='gz::sim::systems::Physics']") is not None
    model = ET.parse(Path(_ROOT) / "gazebo/models/formation_quadrotor/model.sdf.template")
    joints = {j.attrib["name"] for j in model.findall(".//joint")}
    links = {l.attrib["name"] for l in model.findall(".//link")}
    motors = model.findall(".//plugin[@name='gz::sim::systems::MulticopterMotorModel']")
    assert [m.findtext("turningDirection") for m in motors] == ["ccw", "cw", "ccw", "cw"]
    for motor in motors:
        assert motor.findtext("jointName") in joints
        assert motor.findtext("linkName") in links
    mass = sum(float(e.text) for e in model.findall(".//inertial/mass"))
    assert np.isclose(mass, GazeboQuadrotorController().cfg.mass_kg)


def test_gazebo_mixer_torque_directions():
    from dq_control import Quaternion
    from simulators.base import DroneState, VehicleCommand
    controller = GazeboQuadrotorController()
    state = DroneState(np.zeros(3), Quaternion.identity())
    # y thrust produces roll, -x thrust produces pitch; yaw reacts against spin.
    mixer = np.array([[.2, .2, -.2, -.2], [-.2, .2, .2, -.2], [-.016, .016, -.016, .016]])
    commands = [VehicleCommand(np.zeros(3), np.zeros(3), np.array([0., -1., 0.]), np.zeros(3)),
                VehicleCommand(np.zeros(3), np.zeros(3), np.array([1., 0., 0.]), np.zeros(3)),
                VehicleCommand(np.zeros(3), np.array([0., 0., .1]), np.zeros(3), np.zeros(3))]
    for axis, command in enumerate(commands):
        rpm = controller.compute(state, command)
        torques = mixer @ (controller.cfg.motor_constant*(rpm*2*np.pi/60)**2)
        assert torques[axis] > 0
        assert np.allclose(np.delete(torques, axis), 0, atol=1e-10)


def test_repeated_gazebo_pose_preserves_velocity():
    from types import SimpleNamespace
    from simulators.gazebo import GazeboBackend
    backend = GazeboBackend(SimConfig())
    backend.node = SimpleNamespace(poses={n: np.array([0., 0., 1., 0., 0., 0., 1.]) for n in backend.model_names},
                                   pose_stamp={n: 1. for n in backend.model_names})
    backend.states = backend._read_states()
    for n in backend.model_names:
        backend.node.poses[n][0] = .1
        backend.node.pose_stamp[n] = 1.1
    backend.states = backend._read_states()
    assert np.allclose(backend.states[0].velocity, [1, 0, 0])
    repeated = backend._read_states()
    assert np.allclose(repeated[0].velocity, [1, 0, 0])


def test_gazebo_angular_velocity_is_quaternion_sign_invariant():
    from dq_control import Quaternion
    from simulators.gazebo import _quat_angular_velocity
    previous = Quaternion.from_rpy([.1, .2, 3.1])
    current = previous * Quaternion.from_rotvec([0, 0, 1], .01)
    expected = _quat_angular_velocity(previous, current, .02)
    assert np.allclose(expected, [0, 0, .5])
    assert np.allclose(_quat_angular_velocity(previous, -current, .02), expected)
