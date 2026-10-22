"""Optional CVXPY convex smoothing of B-spline control points."""
import numpy as np


def smooth_control_points(points, weight=1.0, bounds=None, preserve_endpoints=True):
    """Minimize fidelity + squared second differences; optionally constrain a box.

    Endpoints remain fixed by default. Set preserve_endpoints=False and omit
    bounds for the unconstrained problem. A clamped B-spline lies inside the
    convex hull of its control points, so box bounds also bound its curve.
    This is not collision avoidance or a quadrotor feasibility guarantee.
    """
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError("Install optional optimization dependencies: pip install -r requirements-course.txt") from exc
    points = np.asarray(points, float)
    if points.ndim != 2 or len(points) < 4 or points.shape[1] not in (2, 3) or not np.all(np.isfinite(points)):
        raise ValueError("Provide at least four finite 2D or 3D points")
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("weight must be nonnegative")
    x = cp.Variable(points.shape)
    objective = cp.sum_squares(x-points) + weight*cp.sum_squares(x[2:]-2*x[1:-1]+x[:-2])
    constraints = [x[0] == points[0], x[-1] == points[-1]] if preserve_endpoints else []
    if bounds is not None:
        lower, upper = np.asarray(bounds, float)
        if lower.shape != (points.shape[1],) or not np.all(np.isfinite([lower, upper])) or np.any(lower > upper):
            raise ValueError("Bounds must match point dimension with lower <= upper")
        constraints += [x >= lower, x <= upper]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or x.value is None:
        raise ValueError(f"Control-point optimization failed: {problem.status}")
    result = np.asarray(x.value)
    if bounds is not None and (np.any(result < lower-1e-5) or np.any(result > upper+1e-5)):
        raise ValueError("Solver result violates requested bounds")
    return result
