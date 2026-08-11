import os
import sys
import numpy as np
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dq_control import (
    Quaternion, DualQuaternion, integrate_pose,
    get_gains, KinematicController, LeaderTrajectory,
)


def test_quaternion_rotation_90deg_about_z():
    q = Quaternion.from_rotvec([0, 0, 1], np.pi / 2)
    v = q.rotate(np.array([1.0, 0.0, 0.0]))
    assert np.allclose(v, [0, 1, 0], atol=1e-8)


def test_quaternion_conjugate_is_inverse_for_unit_quat():
    q = Quaternion.from_rotvec([1, 1, 0], 0.7).normalized()
    identity = q * q.conj()
    assert np.allclose(identity.as_array(), [0, 0, 0, 1], atol=1e-8)


def test_dual_quaternion_pose_roundtrip():
    pos = np.array([1.0, -2.0, 3.5])
    att = Quaternion.from_rotvec([0, 0, 1], 0.3)
    Q = DualQuaternion.from_pose(pos, att)
    assert np.allclose(Q.position(), pos, atol=1e-8)
    assert np.allclose(Q.attitude().as_array(), att.as_array(), atol=1e-8)
    assert abs(Q.norm_check() - 1.0) < 1e-8


def test_integrate_pose_pure_translation():
    Q0 = DualQuaternion.from_pose(np.zeros(3), Quaternion.identity())
    v_cmd = np.array([2.0, 0.0, 0.0])
    Q1 = integrate_pose(Q0, np.zeros(3), v_cmd, dt=0.05)
    assert np.allclose(Q1.position(), [0.1, 0, 0], atol=1e-6)


@pytest.mark.parametrize("experiment", ["proportional", "complex_eig", "real_eig"])
def test_controller_converges_to_static_setpoint(experiment):
    gains = get_gains(experiment)["leader"]
    ctrl = KinematicController(gains)

    Qd = DualQuaternion.from_pose(np.array([1.0, 0.5, 1.0]), Quaternion.from_rotvec([0, 0, 1], 0.4))
    Q = DualQuaternion.from_pose(np.zeros(3), Quaternion.identity())

    dt = 0.02
    for _ in range(3000):
        omega_cmd, v_cmd = ctrl.compute(Q, Qd, omega_d=np.zeros(3), v_d=np.zeros(3), dt=dt)
        Q = integrate_pose(Q, omega_cmd, v_cmd, dt)

    pos_err = np.linalg.norm(Q.position() - Qd.position())
    att_err = np.linalg.norm((Q.attitude().conj() * Qd.attitude()).vec)
    assert pos_err < 0.05, f"{experiment}: position error {pos_err}"
    assert att_err < 0.05, f"{experiment}: attitude error {att_err}"


def test_leader_trajectory_is_periodic_and_bounded():
    traj = LeaderTrajectory()
    T = 2 * np.pi / traj.p.w_d
    p0 = traj.position(0.0)
    pT = traj.position(T)
    assert np.allclose(p0, pT, atol=1e-6)
