"""ROS integration checks; skipped when ROS 2 is not installed/sourced."""
import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
from geometry_msgs.msg import PoseStamped
from simulators.gazebo import _GazeboRosNode


def test_ros_adapter_initializes_reads_stamped_pose_and_closes():
    already_initialized = rclpy.ok()
    adapter = _GazeboRosNode(["regression_drone"])
    try:
        assert rclpy.ok()
        msg = PoseStamped()
        msg.pose.position.x = 1.25
        msg.pose.orientation.w = 1.0
        msg.header.stamp.sec = 2
        msg.header.stamp.nanosec = 500000000
        adapter._make_pose_callback("regression_drone")(msg)
        assert np.allclose(adapter.poses["regression_drone"], [1.25, 0, 0, 0, 0, 0, 1])
        assert adapter.pose_stamp["regression_drone"] == 2.5
        adapter.publish_motor_rpm("regression_drone", np.full(4, 60.0))
    finally:
        adapter.destroy()
        adapter.destroy()
    assert rclpy.ok() == already_initialized


def test_adapter_preserves_existing_ros_context():
    own_context = not rclpy.ok()
    if own_context:
        rclpy.init(args=[])
    try:
        adapter = _GazeboRosNode([])
        adapter.destroy()
        assert rclpy.ok()
    finally:
        if own_context:
            rclpy.try_shutdown()


def test_runs_use_separate_command_topics():
    a = _GazeboRosNode(["drone_0"], "/run_a")
    b = _GazeboRosNode(["drone_0"], "/run_b")
    try:
        assert a.pubs["drone_0"].topic_name == "/run_a/drone_0/gazebo/command/motor_speed"
        assert b.pubs["drone_0"].topic_name == "/run_b/drone_0/gazebo/command/motor_speed"
    finally:
        b.destroy()
        a.destroy()
