import numpy as np
import pytest
from dq_control import Quaternion, DualQuaternion, get_gains, KinematicController, ControllerState
from dq_control.course_trajectories import BSplineTrajectory, InterpolatedPoseTrajectory, slerp
from dq_control.estimation import PositionKalmanFilter
from dq_control.course_analysis import hover_linearization, predict_linear, taylor_approximations, integrate_nonlinear_pose
from simulators import LeaderFollowerSimulation, SimConfig


def test_slerp_shortest_arc_and_antipodal():
    a = Quaternion.from_rotvec([0, 0, 1], np.deg2rad(170))
    b = Quaternion.from_rotvec([0, 0, 1], np.deg2rad(-170))
    middle = slerp(a, b, 0.5)
    assert np.allclose(middle.rotate([1, 0, 0]), [-1, 0, 0])
    assert np.allclose(slerp(a, -a, .5).rotation_matrix(), a.rotation_matrix())
    assert middle.norm() == pytest.approx(1)
    with pytest.raises(ValueError):
        slerp(Quaternion((0, 0, 0), 0), b, 0.5)


@pytest.mark.parametrize("dimension", [2, 3])
def test_spline_endpoints_and_derivatives(dimension):
    points = np.array([[0, 0, 1], [1, 0, 1], [2, 1, 1.5], [3, 0, 1]])[:, :dimension]
    traj = BSplineTrajectory(points, duration=10)
    assert np.allclose(traj.position(0)[:dimension], points[0])
    assert np.allclose(traj.position(10)[:dimension], points[-1])
    assert np.allclose(traj.velocity(0), 0)
    assert np.allclose(traj.acceleration(10), 0)
    t, h = 4., 1e-4
    assert np.allclose((traj.position(t+h)-traj.position(t-h))/(2*h), traj.velocity(t), atol=1e-7)
    assert np.allclose((traj.velocity(t+h)-traj.velocity(t-h))/(2*h), traj.acceleration(t), atol=1e-7)
    assert (traj.yaw(t+h)-traj.yaw(t-h))/(2*h) == pytest.approx(traj.angular_velocity(t)[2], abs=1e-7)
    assert np.allclose(traj.velocity(11), 0)


def test_pose_interpolation_twist_matches_orientation_derivative():
    a = Quaternion.from_rpy([.2, -.1, .3])
    b = Quaternion.from_rpy([-.3, .4, 1.2])
    traj = InterpolatedPoseTrajectory([0, 0, 1], [2, 1, 2], a, b, 10)
    t, h = 3., 1e-5
    qdot = Quaternion.from_array((traj.attitude(t+h).as_array()-traj.attitude(t-h).as_array())/(2*h))
    omega = (traj.attitude(t).conj()*qdot).vec*2
    assert np.allclose(omega, traj.angular_velocity(t), atol=1e-7)
    assert np.allclose(traj.attitude(10).rotation_matrix(), b.rotation_matrix())


def test_kalman_reduces_seeded_measurement_error_and_preserves_covariance():
    rng = np.random.default_rng(7)
    filt = PositionKalmanFilter(.02, measurement_std=.1, acceleration_std=.1)
    truth = np.arange(500)[:, None]*.02*np.array([[.4, -.2, .1]])
    noisy = truth + rng.normal(0, .1, truth.shape)
    estimates = np.array([filt.update(z)[0] for z in noisy])
    assert np.mean((estimates[50:]-truth[50:])**2) < .3*np.mean((noisy[50:]-truth[50:])**2)
    assert np.min(np.linalg.eigvalsh(filt.P)) >= -1e-12
    assert np.allclose(filt.x[3:], [.4, -.2, .1], atol=.06)


def test_hover_linearization_matches_controller_finite_difference():
    g = get_gains("real_eig")["leader"]
    def derivative(x):
        q = Quaternion(x[:3], np.sqrt(1-x[:3]@x[:3]))
        eta = Quaternion(x[3:6], np.sqrt(1-x[3:6]@x[3:6]))
        controller = KinematicController(g, ControllerState(xi=x[9:].copy(), eta=eta))
        dt = 1e-5
        omega, velocity = controller.compute(DualQuaternion.from_pose(x[6:9], q), DualQuaternion.identity(), np.zeros(3), np.zeros(3), dt)
        qdot = (q*Quaternion.pure(omega)).vec*.5
        return np.r_[qdot, (controller.state.eta.vec-eta.vec)/dt, velocity, (controller.state.xi-x[9:])/dt]
    eps = 1e-6
    numerical = np.column_stack([(derivative(eps*e)-derivative(-eps*e))/(2*eps) for e in np.eye(12)])
    assert np.allclose(numerical, hover_linearization(g), atol=1e-7)
    assert np.max(np.linalg.eigvals(hover_linearization(g)).real) < 0
    assert np.allclose(predict_linear(-np.eye(2), [1, 2], 1), np.exp(-1)*np.array([1, 2]))


def test_second_order_taylor_and_nonlinear_pose_solution():
    f = lambda x: x[0]**2 + 3*x[0]*x[1] + 2*x[1]**2
    x, d = np.array([.3, .5]), np.array([.2, -.1])
    first, second = taylor_approximations(f, x, d)
    assert second == pytest.approx(f(x+d), abs=1e-7)
    Q = integrate_nonlinear_pose(DualQuaternion.identity(), np.array([0., 0, 1]), np.array([1., 2, 3]), 2)
    assert np.allclose(Q.position(), [2, 4, 6], atol=1e-7)
    assert np.allclose(Q.attitude().rotation_matrix(), Quaternion.from_rotvec([0, 0, 1], 2).rotation_matrix(), atol=1e-7)


def test_optimization_bounds_and_unconstrained_solution():
    pytest.importorskip("cvxpy")
    from dq_control.optimization import smooth_control_points
    points = np.array([[0., 0], [1, 2], [2, -2], [3, 0]])
    unconstrained = smooth_control_points(points, preserve_endpoints=False)
    D = np.diff(np.eye(4), n=2, axis=0)
    assert np.allclose(unconstrained, np.linalg.solve(np.eye(4)+D.T@D, points), atol=1e-5)
    bounded = smooth_control_points(points, bounds=([0, -.5], [3, .5]))
    assert np.allclose(bounded[[0, -1]], points[[0, -1]], atol=1e-5)
    assert np.max(np.abs(bounded[:, 1])) <= .50001
    with pytest.raises(ValueError, match="failed"):
        smooth_control_points(points, bounds=([0, 1], [3, 2]))


@pytest.mark.parametrize("kalman", [False, True])
def test_full_formation_loop_with_course_trajectory(kalman):
    traj = BSplineTrajectory([[0, 0, 1], [1, 0, 1], [2, .5, 1.5], [3, 0, 1]], 10)
    cfg = SimConfig(simulator="kinematic", duration_sec=10, kalman=kalman,
                    position_noise_std=.01 if kalman else 0, seed=10)
    log = LeaderFollowerSimulation(get_gains("real_eig"), cfg, traj).run()
    assert log["leader_pos"].shape == (480, 3)
    assert all(np.all(np.isfinite(v)) for v in log.values())
    assert np.max(np.linalg.norm(log["leader_pos"]-log["leader_pos_d"], axis=1)) < .2
    assert np.linalg.norm(log["follower_pos"][-1]-log["follower_pos_d"][-1]) < .1
