"""Local hover analysis of the existing controller, plus numerical tools."""
import numpy as np
from scipy.linalg import expm, block_diag
from scipy.integrate import solve_ivp

from .quaternion import Quaternion
from .dual_quaternion import DualQuaternion, pose_derivative


def hover_linearization(gains):
    """State [attitude-error vector, eta vector, position error, xi].

    Fixed desired pose, identity attitude, positive quaternion branch,
    ideal instantaneous twist actuation. Not the full quadrotor dynamics.
    """
    g = gains
    attitude = 0.5*np.block([[g.Kw_p, g.Kw_i], [-g.Kw_i, g.K_eta]])
    position = np.block([[g.Kv_p, g.Kv_i], [-g.Kv_i, g.K_xi]])
    return block_diag(attitude, position)


def predict_linear(A, x0, time):
    """Exact continuous-time LTI prediction x(t)=exp(A*t)x(0)."""
    if not np.isfinite(time) or time < 0:
        raise ValueError("Prediction time must be finite and nonnegative")
    return expm(np.asarray(A)*time) @ np.asarray(x0)


def taylor_approximations(f, x, delta, step=1e-4):
    """Scalar multivariate f: centered finite-difference gradient and Hessian."""
    x, delta = np.asarray(x, float), np.asarray(delta, float)
    if x.ndim != 1 or delta.shape != x.shape or step <= 0:
        raise ValueError("Matching vectors and positive step required")
    n, value = len(x), float(f(x))
    gradient, hessian = np.zeros(n), np.zeros((n, n))
    basis = step*np.eye(n)
    for i, ei in enumerate(basis):
        gradient[i] = (f(x+ei)-f(x-ei))/(2*step)
        hessian[i, i] = (f(x+ei)-2*value+f(x-ei))/step**2
        for j in range(i):
            ej = basis[j]
            hessian[i, j] = hessian[j, i] = (f(x+ei+ej)-f(x+ei-ej)-f(x-ei+ej)+f(x-ei-ej))/(4*step**2)
    first = value + gradient @ delta
    return first, first + 0.5*delta @ hessian @ delta


def integrate_nonlinear_pose(pose, omega, velocity, duration):
    """Adaptive RK45 solve of the existing dual-quaternion pose ODE.

    Constant body omega and inertial velocity; original controller's Euler
    integration is retained. This is an independent numerical demonstration.
    """
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("duration must be finite and nonnegative")
    if duration == 0:
        return DualQuaternion.from_pose(pose.position(), pose.attitude())
    def rhs(t, state):
        Q = DualQuaternion(Quaternion.from_array(state[:4]), Quaternion.from_array(state[4:]))
        derivative = pose_derivative(Q, omega, velocity)
        return np.r_[derivative.P.as_array(), derivative.D.as_array()]
    result = solve_ivp(rhs, [0, duration], np.r_[pose.P.as_array(), pose.D.as_array()],
                       rtol=1e-9, atol=1e-11)
    if not result.success:
        raise RuntimeError(result.message)
    final = result.y[:, -1]
    Q = DualQuaternion(Quaternion.from_array(final[:4]), Quaternion.from_array(final[4:]))
    return DualQuaternion.from_pose(Q.position(), Q.attitude().normalized())
