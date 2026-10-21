import threading
import numpy as np

from dq_control import Quaternion, LeaderTrajectory, get_gains
from dq_control.path_following import PathFollowerTrajectory
from simulators import LeaderFollowerSimulation, SimConfig
from simulators.gazebo import GazeboBackend


def test_increasing_gap_can_recover_older_turn():
    q = Quaternion.identity()
    path = PathFollowerTrajectory(.5, [-.5, 0, 1], [0, 0, 1], q, history_distance=3.)
    for p in [[1, 0, 1], [2, 0, 1], [2, 1, 1], [2, 2, 1]]:
        path.update(0, p, q, .1)
    path.distance = 2.5
    pose, _, _ = path.update(0, [2, 2, 1], q, .1)
    assert np.allclose(pose.position(), [1.5, 0, 1])


def test_live_changes_reach_reference_and_vehicle_without_restart():
    live = {"lock": threading.Lock(), "values": {}}
    sim = LeaderFollowerSimulation(get_gains("real_eig"),
        SimConfig(simulator="kinematic", duration_sec=10), LeaderTrajectory())
    positions = []
    def callback(t, states):
        if t >= 1:
            with live["lock"]:
                live["values"].update(altitude=1.6, route_radius=1.2,
                                     leader_speed=.15, follow_distance=1.5)
        positions.append(states[0].position.copy())
    sim.run(step_callback=callback, live_parameters=live)
    assert abs(positions[-1][2]-1.6) < .1
    assert np.isclose(sim.leader_traj.p.r_x, 1.2)
    assert np.isclose(sim.leader_traj.p.r_y, 1.2*.65/.85)
    assert np.isclose(sim.follower_traj.distance, 1.5)
    assert np.isclose(sim.live_applied["leader_speed"], .15)


def test_windows_gazebo_forwards_trails_to_worker(monkeypatch):
    backend = GazeboBackend(SimConfig(simulator="gazebo", gui=True))
    backend._wsl_worker = object()
    requests = []
    monkeypatch.setattr(backend, "_wsl_request", requests.append)
    backend.draw_trails([[0, 0, 1], [1, 0, 1]])
    assert requests == [{"op": "draw_trails", "positions": [[0., 0., 1.], [1., 0., 1.]]}]
    assert backend._trail_worker is None
