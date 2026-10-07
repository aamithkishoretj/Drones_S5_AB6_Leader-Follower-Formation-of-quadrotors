"""Identified-dynamics predictive control with dual-quaternion pose errors.

The fitted predictor models the *closed-loop backend* including its existing
motor controller. No mass, inertia, thrust coefficient or teacher action
targets enter fitting. Units: metres, seconds, radians; quaternion xyzw.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear

from .quaternion import Quaternion

SCHEMA = "body_twist_gravity_v1"


def response_state(state):
    """Body linear velocity, body angular velocity, gravity direction xy."""
    R = state.attitude.rotation_matrix()
    return np.r_[R.T @ state.velocity, state.angular_velocity, (R.T @ [0., 0., 1.])[:2]]


def response_input(state, omega, velocity):
    return np.r_[state.attitude.rotation_matrix().T @ velocity, omega]


def dual_pose_error(current, desired):
    """Relative unit dual quaternion current* o desired, in current frame."""
    relative = current.conj() * desired
    q = relative.attitude()
    if q.scalar < 0:
        q = -q
    n = np.linalg.norm(q.vec)
    rotation = q.vec*(2*np.arctan2(n, q.scalar)/n) if n > 1e-12 else np.zeros(3)
    return np.r_[relative.position(), rotation]


@dataclass
class LearnedResponse:
    A: np.ndarray
    B: np.ndarray
    c: np.ndarray
    limit: np.ndarray
    metadata: dict

    def predict(self, x, u):
        return np.asarray(x) @ self.A.T + np.asarray(u) @ self.B.T + self.c

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, A=self.A, B=self.B, c=self.c, limit=self.limit,
                 metadata=np.asarray(json.dumps(self.metadata)))

    @classmethod
    def load(cls, path, backend, dt):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Trained model missing: {path}. Run scripts/train_data_driven.py --simulator {backend} first.")
        with np.load(path, allow_pickle=False) as data:
            model = cls(*(data[k].copy() for k in ("A", "B", "c", "limit")),
                        json.loads(str(data["metadata"])))
        if model.metadata.get("schema") != SCHEMA or model.metadata.get("backend") != backend:
            raise ValueError("Model feature schema or simulator does not match this run")
        if not np.isclose(model.metadata["dt"], dt):
            raise ValueError("Model control timestep differs; retrain at this --ctrl_freq")
        for value, shape in [(model.A, (8, 8)), (model.B, (8, 6)), (model.c, (8,)), (model.limit, (6,))]:
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError("Invalid learned model arrays")
        if np.any(model.limit <= 0):
            raise ValueError("Invalid learned command bounds")
        return model


def fit_response(x, u, y, metadata, ridge=1e-3):
    """Standardized ridge system identification: next state is the label."""
    x, u, y = (np.asarray(a, float) for a in (x, u, y))
    if x.ndim != 2 or x.shape[1] != 8 or u.shape != (len(x), 6) or y.shape != x.shape or len(x) < 100:
        raise ValueError("Need at least 100 aligned state/action/next-state samples")
    if not all(np.all(np.isfinite(a)) for a in (x, u, y)) or ridge <= 0:
        raise ValueError("Finite data and positive ridge required")
    features = np.c_[x, u]
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale = np.maximum(scale, 1e-5)
    z = (features-mean)/scale
    target_mean = y.mean(axis=0)
    weights = np.linalg.solve(z.T @ z + ridge*np.eye(14), z.T @ (y-target_mean)) / scale[:, None]
    bias = target_mean - mean @ weights
    # Bounds come from observed action magnitudes with a fixed minimum to
    # accommodate very quiet axes; a simulator-specific model is mandatory.
    limit = np.maximum(np.quantile(np.abs(u), .995, axis=0), [.3, .3, .3, .2, .2, .3])
    meta = dict(metadata, schema=SCHEMA, samples=len(x), ridge=ridge,
                feature_std=features.std(axis=0).tolist(),
                feature_rank=int(np.linalg.matrix_rank(z)))
    return LearnedResponse(weights[:8].T, weights[8:].T, bias, limit, meta)


class LearnedDQController:
    """Bounded, constant-input receding-horizon optimizer.

    Uses learned A/B/c to predict body twist. Integrating predicted twist
    gives a local first-order forecast of the dual-quaternion pose error.
    A single command is held over the prediction horizon, then replanned
    every sample. This is move-blocked MPC, not a full nonlinear DQ solver.
    """
    def __init__(self, model, dt, horizon=24):
        self.model, self.dt, self.horizon = model, dt, horizon
        self.previous = np.zeros(6)
        self.saturated_steps = 0
        self.solve_count = 0
        self.maps = []
        F, G, c = np.eye(8), np.zeros((8, 6)), np.zeros(8)
        E, H, h = np.zeros((6, 8)), np.zeros((6, 6)), np.zeros(6)
        self.pose_weight = np.array([4., 4., 5., .5, .5, 1.5])
        self.twist_weight = np.array([.3, .3, .4, .05, .05, .15])
        rows = []
        for j in range(1, horizon+1):
            F, G, c = model.A @ F, model.A @ G + model.B, model.A @ c + model.c
            E, H, h = E+dt*F[:6], H+dt*G[:6], h+dt*c[:6]
            self.maps.append((j, F.copy(), c.copy(), E.copy(), h.copy()))
            rows.extend([self.pose_weight[:, None]*H, self.twist_weight[:, None]*G[:6]])
        self.regularization, self.smoothing = .12, .2
        rows.extend([self.regularization*np.eye(6), self.smoothing*np.eye(6)])
        self.matrix = np.vstack(rows)

    def compute(self, Q, Qd, omega_d, v_d, dt, measured_state):
        if not np.isclose(dt, self.dt):
            raise ValueError("Control timestep changed after model initialization")
        x = response_state(measured_state)
        error = dual_pose_error(Q, Qd)
        R = Q.attitude().rotation_matrix()
        angular_ff = (Q.attitude().conj()*Qd.attitude()*Quaternion.pure(omega_d)
                      *Qd.attitude().conj()*Q.attitude()).vec
        feedforward = np.r_[R.T @ v_d, angular_ff]
        targets = []
        for j, F, c, E, h in self.maps:
            targets.extend([self.pose_weight*(error+j*dt*feedforward-E@x-h),
                            self.twist_weight*(feedforward-(F@x+c)[:6])])
        targets.extend([self.regularization*feedforward, self.smoothing*self.previous])
        solution = lsq_linear(self.matrix, np.concatenate(targets),
                              bounds=(-self.model.limit, self.model.limit), method="bvls", tol=1e-7)
        if not solution.success or not np.all(np.isfinite(solution.x)):
            raise RuntimeError("Learned controller optimization failed; no analytic fallback was used")
        self.previous = solution.x
        self.solve_count += 1
        self.saturated_steps += int(np.any(np.abs(solution.x) >= .999*self.model.limit))
        return solution.x[3:], R @ solution.x[:3]


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
