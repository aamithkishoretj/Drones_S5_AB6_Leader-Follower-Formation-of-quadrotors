"""Linear position/velocity Kalman filter; quaternion attitude stays untouched."""
import numpy as np


class PositionKalmanFilter:
    def __init__(self, dt, measurement_std=0.03, acceleration_std=0.5):
        if not np.all(np.isfinite([dt, measurement_std, acceleration_std])) or dt <= 0 or measurement_std <= 0 or acceleration_std < 0:
            raise ValueError("Require dt > 0, measurement_std > 0, acceleration_std >= 0")
        self.F = np.eye(6)
        self.F[:3, 3:] = dt*np.eye(3)
        self.H = np.c_[np.eye(3), np.zeros((3, 3))]
        G = np.r_[0.5*dt**2*np.eye(3), dt*np.eye(3)]
        self.Q = acceleration_std**2 * G @ G.T
        self.R = measurement_std**2*np.eye(3)
        self.x = None
        self.P = np.eye(6)

    def update(self, position):
        z = np.asarray(position, float).reshape(3)
        if not np.all(np.isfinite(z)):
            raise ValueError("Position measurements must be finite")
        if self.x is None:
            self.x = np.r_[z, np.zeros(3)]
            self.P[:3, :3] = self.R
        else:
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            S = self.H @ self.P @ self.H.T + self.R
            K = np.linalg.solve(S, self.H @ self.P).T
            self.x += K @ (z - self.H @ self.x)
            residual = np.eye(6) - K @ self.H
            self.P = residual @ self.P @ residual.T + K @ self.R @ K.T
            self.P = (self.P + self.P.T)/2
        return self.x[:3].copy(), self.x[3:].copy()
