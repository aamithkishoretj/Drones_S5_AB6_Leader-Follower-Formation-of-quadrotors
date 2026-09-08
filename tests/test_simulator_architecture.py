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
