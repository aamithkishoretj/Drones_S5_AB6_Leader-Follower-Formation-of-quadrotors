import numpy as np
import pytest
from dq_control import Quaternion, DualQuaternion, get_gains
from dq_control.learned_control import (fit_response, LearnedResponse, LearnedDQController,
                                       dual_pose_error, response_state)
from simulators.base import DroneState, SimConfig


def identified_fixture():
    rng = np.random.default_rng(33)
    x, u = rng.normal(size=(1500, 8)), rng.normal(size=(1500, 6))
    A = np.eye(8)*.8
    B = np.r_[np.eye(6)*.2, np.zeros((2, 6))]
    c = np.arange(8)*.001
    y = x@A.T + u@B.T + c
    model = fit_response(x, u, y, {"backend": "kinematic", "dt": 1/48})
    return model, A, B, c


def test_identification_learns_observed_dynamics():
    model, A, B, c = identified_fixture()
    assert np.allclose(model.A, A, atol=1e-5)
    assert np.allclose(model.B, B, atol=1e-5)
    assert np.allclose(model.c, c, atol=1e-5)


def test_dual_pose_error_frames_and_quaternion_sign():
    q = Quaternion.from_rpy([0, 0, np.pi/2])
    current = DualQuaternion.from_pose([1, 2, 3], q)
    desired = DualQuaternion.from_pose([1, 3, 3], q)
    assert np.allclose(dual_pose_error(current, desired), [1, 0, 0, 0, 0, 0], atol=1e-10)
    negated = DualQuaternion.from_pose([1, 3, 3], -q)
    assert np.allclose(dual_pose_error(current, negated), dual_pose_error(current, desired))


def test_model_mismatch_and_missing_fail_explicitly(tmp_path):
    model, *_ = identified_fixture()
    path = tmp_path/"model.npz"
    model.save(path)
    assert np.allclose(LearnedResponse.load(path, "kinematic", 1/48).A, model.A)
    with pytest.raises(ValueError, match="simulator"):
        LearnedResponse.load(path, "gazebo", 1/48)
    with pytest.raises(ValueError, match="timestep"):
        LearnedResponse.load(path, "kinematic", 1/20)
    with pytest.raises(FileNotFoundError, match="Trained model missing"):
        LearnedResponse.load(tmp_path/"missing.npz", "kinematic", 1/48)


def test_learned_model_changes_real_command_and_enforces_limits():
    model, *_ = identified_fixture()
    state = DroneState(np.zeros(3), Quaternion.identity())
    desired = DualQuaternion.from_pose([.3, 0, 0], Quaternion.identity())
    def command(m):
        ctrl = LearnedDQController(m, 1/48)
        return ctrl.compute(state.as_pose(), desired, np.zeros(3), np.zeros(3), 1/48, state)[1]
    first = command(model)
    slower = LearnedResponse(model.A, model.B*.5, model.c, model.limit, model.metadata)
    assert np.linalg.norm(first-command(slower)) > .01
    assert np.all(np.abs(first) <= model.limit[:3]+1e-8)
    assert first[0] > 0


def test_learned_loop_does_not_call_original_controller(tmp_path, monkeypatch):
    from simulators import LeaderFollowerSimulation
    from dq_control import KinematicController, LeaderTrajectory
    model, *_ = identified_fixture()
    path = tmp_path/"model.npz"
    model.save(path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Original controller was called in learned mode")
    monkeypatch.setattr(KinematicController, "compute", forbidden)
    cfg = SimConfig(simulator="kinematic", controller="learned", learned_model=str(path), duration_sec=.2)
    log = LeaderFollowerSimulation(get_gains("real_eig"), cfg, LeaderTrajectory()).run()
    assert len(log["response_u"]) == 9
    assert np.all(np.isfinite(log["response_y"]))
