"""Run numerical course demonstrations without a physics-engine dependency."""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dq_control import Quaternion, DualQuaternion, get_gains, EXPERIMENTS
from dq_control.course_analysis import hover_linearization, predict_linear, taylor_approximations, integrate_nonlinear_pose
from dq_control.course_trajectories import lerp, slerp


def main():
    report = {}
    for experiment in EXPERIMENTS:
        report[experiment] = {}
        for vehicle, gains in get_gains(experiment).items():
            A = hover_linearization(gains)
            poles = np.linalg.eigvals(A)
            x0 = np.zeros(12)
            x0[[0, 6]] = 0.1
            report[experiment][vehicle] = {
                "hover_poles": [str(complex(p)) for p in poles],
                "full_augmented_system_hurwitz": bool(np.max(poles.real) < -1e-10),
                "state_after_5_seconds": predict_linear(A, x0, 5).tolist(),
                "note": "Proportional-only gains leave unused integral states at zero eigenvalues; pose errors still decay. Local ideal-actuation analysis only."
            }
    # Rotated offset x*cos(yaw): two-variable nonlinear formation geometry.
    f = lambda x: x[0]*np.cos(x[1])
    x, delta = np.array([1.85, 0.4]), np.array([0.1, 0.15])
    first, second = taylor_approximations(f, x, delta)
    report["taylor_rotated_offset"] = {"exact": f(x+delta), "first_order": first, "second_order": second}
    q = Quaternion.from_rotvec([0, 0, 1], np.pi/2)
    report["quaternion"] = {"norm": q.norm(), "inverse_product": (q*q.inv()).as_array().tolist(),
        "sandwich_rotated_x": q.rotate([1, 0, 0]).tolist(),
        "slerp_midpoint": slerp(Quaternion.identity(), q, 0.5).as_array().tolist(),
        "lerp_midpoint": lerp([0, 0, 1], [2, 1, 1], 0.5).tolist()}
    pose = integrate_nonlinear_pose(DualQuaternion.identity(), np.array([0, 0, 1.]), np.array([1., 0, 0]), 2.)
    report["rk45_pose_after_2_seconds"] = {"position": pose.position().tolist(), "quaternion": pose.attitude().as_array().tolist()}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
