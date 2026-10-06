"""Identified flight-stack dynamics and dual-quaternion predictive control.

The model learns velocity/yaw-rate response to Guided commands from SITL data.
Pose targets, errors and reconstruction use the existing dual-quaternion algebra.
It is a local finite-feature predictor, not a globally linear quadrotor model.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares

from .quaternion import Quaternion
from .dual_quaternion import DualQuaternion, integrate_pose
from .controller import pose_error


def heading_rotation(Q):
    yaw = Q.attitude().to_rpy()[2]
    return Quaternion.from_rpy([0., 0., yaw]).rotation_matrix()


def yaw_rate(Q, omega):
    roll, pitch, _ = Q.attitude().to_rpy()
    return (np.sin(roll)*omega[1] + np.cos(roll)*omega[2])/max(.1, np.cos(pitch))


def features(s, Q, quadratic=False):
    # Heading-relative velocities avoid learning a different model at every yaw.
    # Pose-dependent tilt features come from the principal part of the DQ.
    up_body = Q.attitude().rotation_matrix().T @ np.array([0., 0., 1.])
    base = np.r_[1., s, up_body[:2]]
    return np.r_[base, base[1:]*np.abs(base[1:])] if quadratic else base


class IdentifiedDynamics:
    def __init__(self, weights, dt, quadratic=False, metadata=None):
        self.weights = np.asarray(weights, float)
        self.dt = float(dt)
        self.quadratic = bool(quadratic)
        self.metadata = metadata or {}
        expected = (17 if quadratic else 11, 6)
        if self.weights.shape != expected or not np.isfinite(self.weights).all() or not np.isfinite(dt) or dt <= 0:
            raise ValueError('Invalid identified model shape, coefficients or sample interval')

    def step(self, Q, velocity, omega, command):
        R = heading_rotation(Q)
        s = np.r_[R.T @ velocity, yaw_rate(Q, omega)]
        u = np.r_[R.T @ command[:3], command[3]]
        sn = np.r_[features(s, Q, self.quadratic), u] @ self.weights
        return R @ sn[:3], float(sn[3]), np.clip(sn[4:6], -.35, .35)

    def rollout(self, Q, velocity, omega, command, steps):
        # Each step predicts tilt as well as velocity and reconstructs a unit DQ;
        # tilt response is learned even though Guided directly commands velocity/yaw.
        for _ in range(steps):
            vn, rn, tilt = self.step(Q, velocity, omega, command)
            _, _, yaw = Q.attitude().to_rpy()
            pitch = -np.arcsin(tilt[0])
            roll = np.arctan2(tilt[1], np.sqrt(max(.01, 1-float(tilt@tilt))))
            current_rate = yaw_rate(Q, omega)
            attitude = Quaternion.from_rpy([roll, pitch, yaw + .5*(current_rate+rn)*self.dt])
            Q = DualQuaternion.from_pose(Q.position()+.5*(velocity+vn)*self.dt, attitude)
            velocity = vn
            omega = np.array([-np.sin(pitch)*rn, np.sin(roll)*np.cos(pitch)*rn,
                              np.cos(roll)*np.cos(pitch)*rn])
        return Q, velocity, omega

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, weights=self.weights, dt=self.dt, quadratic=self.quadratic,
                            state_transition=self.weights[1:7].T,
                            control_matrix=self.weights[-4:].T,
                            metadata_json=np.array(json.dumps(self.metadata)))

    @classmethod
    def load(cls, path):
        if not Path(path).is_file():
            raise FileNotFoundError('Trained dynamics model missing. Run bash scripts/train_data_driven.sh first.')
        with np.load(path, allow_pickle=False) as a:
            model = cls(a['weights'], float(a['dt']), bool(a['quadratic']), json.loads(str(a['metadata_json'])))
        if model.metadata.get('simulator') != 'ardupilot':
            raise ValueError('Model must be trained from ArduPilot SITL data')
        if not model.metadata.get('validation_passed'):
            raise ValueError('Model has not passed held-out flight prediction validation')
        return model


class DataDrivenDQController:
    """Receding-horizon controller; learned dynamics choose every applied command.

    No KinematicController calculation is blended into this controller. The
    optimization minimizes a DQ pose error after rolling out learned dynamics.
    """
    def __init__(self, model, horizon_steps=5):
        self.model = model
        self.steps = horizon_steps
        self.previous = np.zeros(4)
        self.calls = 0
        self.last_prediction_error = 0.

    def compute(self, Q, Qd, omega_d, v_d, dt, *, velocity, angular_velocity):
        horizon = self.steps*self.model.dt
        target = integrate_pose(Qd, omega_d, v_d, horizon)
        rate_d = yaw_rate(Qd, omega_d)
        def residual(command):
            predicted, vn, wn = self.model.rollout(Q, velocity, angular_velocity, command, self.steps)
            dq, dp = pose_error(predicted, target)
            sign = 1 if dq.scalar >= 0 else -1
            return np.r_[dp, 1.2*sign*dq.vec,
                         .3*(vn-v_d), .2*(yaw_rate(predicted, wn)-rate_d),
                         .08*(command-self.previous)]
        limits = np.array([.8, .8, .45, .65])
        result = least_squares(residual, np.clip(self.previous, -limits+1e-7, limits-1e-7),
                               bounds=(-limits, limits), max_nfev=12, ftol=1e-4,
                               xtol=1e-4, gtol=1e-4)
        if not np.isfinite(result.x).all() or not np.isfinite(result.fun).all():
            raise RuntimeError('Data-driven DQ optimization produced non-finite commands')
        command = result.x
        if np.linalg.norm(command[:3]) > .8:
            command[:3] *= .8/np.linalg.norm(command[:3])
        self.previous = command.copy()
        self.calls += 1
        self.last_prediction_error = float(np.linalg.norm(result.fun[:3]))
        roll, pitch, _ = Q.attitude().to_rpy()
        rate = command[3]
        body_rate = np.array([-np.sin(pitch)*rate, np.sin(roll)*np.cos(pitch)*rate,
                              np.cos(roll)*np.cos(pitch)*rate])
        return body_rate, command[:3].copy()
