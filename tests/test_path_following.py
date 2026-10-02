import numpy as np
from dq_control import Quaternion
from dq_control.path_following import PathFollowerTrajectory
from simulators.trails import FlightTrails


def test_follower_replays_corner_instead_of_offset_curve():
    q = Quaternion.identity()
    follower = PathFollowerTrajectory(1., [-1, 0, 1], [0, 0, 1], q)
    for pos, expected in [([1, 0, 1], [0, 0, 1]), ([1, 1, 1], [1, 0, 1]),
                          ([1, 2, 1], [1, 1, 1]), ([2, 2, 1], [1, 2, 1])]:
        pose, _, _ = follower.update(0, pos, q, 1.)
        assert np.allclose(pose.position(), expected)
    pose, omega, velocity = follower.update(1, [2, 2, 1], q, 1.)
    assert np.allclose(pose.position(), [1, 2, 1])
    assert np.allclose(velocity, 0)


def test_history_bounded():
    q = Quaternion.identity()
    follower = PathFollowerTrajectory(.5, [-.5, 0, 0], [0, 0, 0], q)
    for x in np.arange(.1, 100, .1):
        follower.update(0, [x, 0, 0], q, .1)
    assert len(follower.history) <= 8
    assert np.allclose(follower.previous_position, [99.4, 0, 0])


def test_crossing_replays_chronological_branch_in_3d():
    q = Quaternion.identity()
    follower = PathFollowerTrajectory(.5, [-.5, 0, 1], [0, 0, 1], q)
    for position in [[1, 0, 1], [1, 1, 2], [0, 1, 2], [0, 0, 1]]:
        pose, _, _ = follower.update(0, position, q, 1.)
    # On returning to the start, stay on the just-flown descending segment,
    # rather than switching back to the earlier eastbound branch.
    assert np.allclose(pose.position(), [0, .5/np.sqrt(2), 1+.5/np.sqrt(2)])


def test_trail_sampling_and_marker():
    trail = FlightTrails(max_points=3, spacing=.1)
    for x in [0, .01, .2, .4, .6]:
        trail.update([[x, 0, 1], [x, 1, 1]])
    assert len(trail.points[0]) == 3
    marker = trail.marker(0)
    assert "type:LINE_STRIP" in marker
    assert marker.count("point {") == 3
    assert 'id:1' in marker
    assert 'id:2' in trail.marker(1)
