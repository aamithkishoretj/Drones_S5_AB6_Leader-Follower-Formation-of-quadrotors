"""Learned dynamics actually select commands; no analytical-controller fallback."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dq_control import DualQuaternion, Quaternion, LeaderTrajectory, get_gains
from dq_control.data_driven import IdentifiedDynamics, DataDrivenDQController
from simulators.base import SimConfig
from simulators.kinematic import KinematicBackend
from simulators.leader_follower import LeaderFollowerSimulation


def model():
    W=np.zeros((11,6))
    W[1:5,:4]=.65*np.eye(4)
    W[5:7,4:]=.65*np.eye(2)
    W[-4:,:4]=.35*np.eye(4)
    return IdentifiedDynamics(W,.2,metadata={'simulator':'ardupilot','validation_passed':True})


def test_predictor_is_translation_and_heading_equivariant():
    m=model()
    q=DualQuaternion.from_pose([0,0,2],Quaternion.identity())
    rotated=DualQuaternion.from_pose([7,8,3],Quaternion.from_rpy([0,0,np.pi/2]))
    v1,r1,_=m.step(q,np.array([.2,0,0]),np.zeros(3),np.array([.4,0,0,.1]))
    v2,r2,_=m.step(rotated,np.array([0,.2,0]),np.zeros(3),np.array([0,.4,0,.1]))
    assert np.allclose(v2,[-v1[1],v1[0],v1[2]])
    assert np.isclose(r1,r2)


def test_dual_quaternion_rollout_stays_on_pose_manifold():
    Q=DualQuaternion.from_pose([1,2,3],Quaternion.from_rpy([.1,.2,.3]))
    predicted,_,_=model().rollout(Q,np.zeros(3),np.zeros(3),np.array([.2,.1,0,.2]),10)
    assert np.isclose(predicted.P.norm(),1.)
    assert abs(predicted.P.as_array()@predicted.D.as_array())<1e-10
    assert np.isfinite(predicted.position()).all()


def test_command_changes_when_learned_response_changes():
    Q=DualQuaternion.from_pose([0,0,2],Quaternion.identity())
    target=DualQuaternion.from_pose([.15,0,2],Quaternion.identity())
    slow=model()
    slow.weights[-4:]*=.5
    args=(Q,target,np.zeros(3),np.zeros(3),.05)
    fast_u=DataDrivenDQController(model()).compute(*args,velocity=np.zeros(3),angular_velocity=np.zeros(3))[1]
    slow_u=DataDrivenDQController(slow).compute(*args,velocity=np.zeros(3),angular_velocity=np.zeros(3))[1]
    assert slow_u[0]>fast_u[0]>0
    assert np.linalg.norm(slow_u)<=.8+1e-8


def test_controller_does_not_depend_on_quaternion_sign():
    Q=DualQuaternion.from_pose([0,0,2],Quaternion.identity())
    target=DualQuaternion.from_pose([.1,0,2],Quaternion.from_rpy([0,0,.3]))
    opposite=DualQuaternion.from_pose(target.position(),-target.attitude())
    a=DataDrivenDQController(model()).compute(Q,target,np.zeros(3),np.zeros(3),.05,velocity=np.zeros(3),angular_velocity=np.zeros(3))
    b=DataDrivenDQController(model()).compute(Q,opposite,np.zeros(3),np.zeros(3),.05,velocity=np.zeros(3),angular_velocity=np.zeros(3))
    assert np.allclose(a,b,atol=1e-6)


def test_invalid_or_unvalidated_model_is_rejected(tmp_path):
    m=model()
    m.metadata['validation_passed']=False
    path=tmp_path/'invalid.npz'
    m.save(path)
    with pytest.raises(ValueError,match='held-out'):
        IdentifiedDynamics.load(path)
    with pytest.raises(ValueError,match='Invalid'):
        IdentifiedDynamics(np.ones((3,4)),.2)


def test_shared_flight_loop_uses_learned_controller_only(tmp_path,monkeypatch):
    from dq_control.controller import KinematicController
    def forbidden(*args,**kwargs):
        raise AssertionError('Legacy control law must not run in data-driven flight')
    monkeypatch.setattr(KinematicController,'compute',forbidden)
    path=tmp_path/'learned.npz'
    model().save(path)
    cfg=SimConfig(simulator='ardupilot',controller='data_driven',dynamics_model=str(path),
                  ctrl_freq=10,duration_sec=.5,follow_distance=1.5)
    sim=LeaderFollowerSimulation(get_gains('real_eig'),cfg,LeaderTrajectory(),backend=KinematicBackend(cfg))
    log=sim.run()
    assert sim.leader_ctrl.calls==sim.follower_ctrl.calls==5
    assert np.isfinite(log['leader_prediction_error']).all()
    assert log['leader_dual_quaternion'].shape==(5,8)


def test_fit_refuses_train_validation_overlap(tmp_path):
    from scripts.train_ardupilot_model import train
    path=tmp_path/'same.npz'
    with pytest.raises(ValueError,match='separate flights'):
        train([path],[path],tmp_path/'model.npz',tmp_path/'report.json')
