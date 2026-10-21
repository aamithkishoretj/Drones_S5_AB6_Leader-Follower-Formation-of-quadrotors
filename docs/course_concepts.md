# Course concepts in the dual-quaternion formation project

The main implementation now uses [data-driven predictive control](data_driven.md)
with dual-quaternion pose error, learned future-state prediction, and bounded
optimization. Quaternion algebra is preserved. The original controller is an
explicit `--controller analytic` baseline. Recorded-path following remains the
default. The examples below are optional supporting demonstrations; they are
not what makes the main controller data-driven.

## Run the extensions

From the repository root in PowerShell, use the existing virtual environment:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements.txt -r requirements-course.txt
.venv/Scripts/python.exe -B scripts/run_experiment.py --controller analytic --experiment real_eig --simulator kinematic --trajectory bspline --duration 30 --optimize_path --kalman --position_noise_std 0.03
.venv/Scripts/python.exe -B scripts/run_experiment.py --controller analytic --experiment real_eig --simulator kinematic --trajectory interpolated --duration 30
.venv/Scripts/python.exe -B scripts/analyze_course_concepts.py
.venv/Scripts/python.exe -B -m pytest -q
```

`kinematic` is an ideal velocity-actuated model for checking the formation law;
it does not simulate mass, motors, thrust limits, or quadrotor underactuation.
Choose `--simulator mujoco`, `pybullet`, or `gazebo` for an installed physics
backend. Their existing actuation differences still apply: MuJoCo stabilizes
attitude rather than following the complete requested attitude trajectory.
The new trajectories have only been validated with the ideal kinematic backend
unless a separate physics run is explicitly reported.

Edit `configs/course_bspline.json` or `configs/course_interpolated.json`, or
pass another file with `--trajectory_file`. Duration sets the time to travel
the path. Start/end velocities and accelerations are zero; after completion,
the reference holds its final pose. Very short durations demand faster motion.
The default formation follows the leader's recorded route with a 0.8 m
along-path gap. See [path-following details](path_following.md). Explicit
`--follower_offset_mode world` restores the paper's fixed offset, while `body`
uses the existing rotating-offset behavior and reference-heading handling.

## How the topics map to the code

| Topic in the image | Working project component |
| --- | --- |
| Quaternion norm, conjugate, inverse, product, sandwich | Existing `quaternion.py`; numerical examples in `analyze_course_concepts.py` |
| LERP / SLERP | `course_trajectories.py`: linearly interpolate position endpoints and interpolate unit orientations along the shortest arc |
| Differential-equation classification | Existing pose/controller equations; classification below |
| First/second-order multivariate approximation | `course_analysis.taylor_approximations`; rotated formation-offset example |
| Taylor-series failure | Examples and limits below |
| Linear-system stability | `hover_linearization` and its eigenvalues for both drones and all gain sets |
| Future-state prediction | `predict_linear`: matrix-exponential solution |
| Nonlinear differential equation solution | `integrate_nonlinear_pose`: adaptive RK45 of the existing pose derivative |
| Unconstrained/constrained convex optimization | `optimization.smooth_control_points`, using CVXPY, the Python modeling package |
| 2D/3D B-splines and knots | `BSplineTrajectory`, including explicit clamped cubic knot vectors |
| Linear Kalman filtering | `PositionKalmanFilter`, optional in the actual formation loop |

“MCQ type” is an assessment format rather than an algorithm. Review questions
are included at the end; it does not need a runtime controller feature.

## Interpolation and knots

LERP interpolates Euclidean positions. SLERP interpolates orientations on the
unit quaternion sphere, accounting for the equivalence of q and -q. We use
the quintic time map `u(s)=10s^3-15s^4+6s^5`, `s=t/duration`, so endpoint motion
starts and stops smoothly. This changes timing, not the geometric straight
position path or shortest orientation arc. Angular velocity is derived from
the same SLERP rotation vector, rather than differencing Euler angles.

For n cubic B-spline control points, the knot vector has n+4 entries, starts
with four zeros, and ends with four ones. Simple interior knots preserve C2
geometric continuity. Clamping interpolates the first and last control points;
interior control points generally are not points on the curve. Two-dimensional
inputs are lifted to the specified `altitude`. Velocity and acceleration use
analytic spline derivatives plus the chain rule for the time map. Tangent yaw
requires a nonzero horizontal tangent; degenerate tangents use zero yaw rate
and may have heading discontinuities. Choose control points without horizontal
cusps when using tangent heading.

## Equation classification, linearization, and prediction

The pose kinematics and controller integral-state equations form a first-order
ODE system. Derivatives enter to the first power (degree one). The closed-loop
system is nonlinear because of quaternion products, frame rotations, and sign
terms. This does not make it a second-order ODE: products of states are not
derivatives of higher order.

At a fixed hover reference with identity desired attitude, positive quaternion
branch, and ideal instantaneous twist actuation, define the small-error state
`x=[e_q, eta_vec, e_p, xi]`. The linear model is block diagonal with

```text
A_att = 0.5 * [[Kw_p,  Kw_i],
               [-Kw_i, K_eta]]
A_pos =       [[Kv_p,  Kv_i],
               [-Kv_i, K_xi]]
```

All eigenvalues with negative real parts imply local exponential stability
of this linear model. Positive real parts imply instability. Eigenvalues on
the imaginary axis require more analysis; eigenvalues alone do not settle
stability in that case. For the proportional-only configuration, inactive
integral states introduce zero eigenvalues, although pose-error modes decay.
The analysis reports that distinction rather than declaring the augmented
system asymptotically stable.

`x(t)=expm(A*t)x(0)` predicts the unforced LTI error state. This is a local hover
prediction, not a guarantee for a curved reference, a noisy estimator, actuator
saturation, or any physics backend. RK45 separately integrates the original
dual-quaternion pose ODE for a prescribed constant twist. The simulation's
existing Euler pose integration remains unchanged.

## Taylor approximations and their limits

For a scalar multivariate function, the tool evaluates
`f(x+d) ≈ f(x) + grad(f)^T d` and adds `0.5*d^T Hessian(f)*d` for second order.
The example `f(r,psi)=r*cos(psi)` is the x component of a rotated offset.
Derivatives are centered numerical differences, so step size and numerical
roundoff affect accuracy. This helper assumes sufficient smoothness near x.

Examples where Taylor expansions cannot represent the function locally:

- `abs(x)` at zero: the first derivative does not exist.
- `exp(-1/x^2)` for nonzero x, extended by zero at x=0: all derivatives at
  zero vanish, but the function is positive at every nonzero x. Smoothness
  alone does not imply equality to the Taylor series.
- `1/(1-x)` expanded at zero: the geometric series converges only for |x|<1;
  using it outside that radius fails.

In this project, do not linearize across the quaternion sign switch or a
singular tangent heading and expect one smooth Taylor model to remain valid.

## Convex optimization

The optional planner minimizes
`||P-P_raw||_F^2 + weight*||D2*P||_F^2`. It smooths control points before the
trajectory enters the unchanged dual-quaternion controller. It is not an MPC
controller or an optimization over dual quaternions.

The CLI fixes endpoint positions and applies the box bounds from the JSON
when present. API callers can use `preserve_endpoints=False, bounds=None`
for a genuinely unconstrained least-squares problem. A nonnegative weight
makes the objective convex. Infeasible constraints raise an explicit error.
Control-point box bounds also bound the B-spline reference through its convex
hull property; they do not constrain the real drone's tracking error, motor
commands, inter-drone spacing, or obstacles. Solver feasibility uses numerical
tolerance. The JSON bounds describe the leader reference only.

## Kalman estimation and logs

The linear state is `[px,py,pz,vx,vy,vz]`, with constant-velocity transition F
and position observation H. Process covariance assumes independent random
acceleration per interval. The filter uses a Joseph covariance update.
`--kalman_measurement_std` sets the assumed measurement uncertainty;
`--position_noise_std` independently sets injected simulation noise. Match them
for a consistent noise experiment. `--kalman_acceleration_std` trades smoothness
against responsiveness to acceleration. Initial velocity is unknown and set
to zero with nonzero uncertainty.

Filtering affects the states supplied to both controllers and the leader
measurements used to construct the follower reference. Attitude and angular
velocity are taken from the backend unchanged; this is not an attitude EKF.
With noise but no filter, backend velocity remains available while positions
are noisy. Thus this mode isolates position-measurement noise rather than
simulating a complete noisy sensor suite.

Logs retain true `leader_pos`/`follower_pos`, plus measured and estimated
positions. Tracking errors therefore remain errors of actual vehicle motion.
With filtering disabled, “estimated” denotes the state fed to the controller.
`run_options_json` records flags and seed. Course runs also save resolved spline
control points/knots or interpolation endpoints/attitudes, including the result
of optional optimization. Default noise is zero and filtering
is disabled. Select `--follower_offset_mode world` to reproduce the original
fixed-offset formation as well.

## Quick review questions

1. Does a B-spline pass through all its control points? **No.** Clamped
   endpoints are interpolated; interior points generally are not.
2. Is the inverse always the conjugate? **Only for unit quaternions.**
3. Does q and -q represent different attitudes? **No.**
4. Do zero eigenvalues establish asymptotic stability? **No.**
5. Is this position Kalman filter nonlinear? **No.** Its transition and
   measurement models are linear; the surrounding formation controller is nonlinear.

Implementation references: [SciPy BSpline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html)
and [CVXPY quadratic programs](https://www.cvxpy.org/examples/basic/quadratic_program.html).
