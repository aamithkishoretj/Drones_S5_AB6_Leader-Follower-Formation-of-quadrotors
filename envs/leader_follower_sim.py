"""
Backward-compatible entry point for the original PyBullet environment name.

New code should import `simulators.LeaderFollowerSimulation` and select a
backend through `SimConfig.simulator`.
"""
from __future__ import annotations

from simulators import LeaderFollowerSimulation, SimConfig


class LeaderFollowerSim:
    """Compatibility wrapper preserving the original constructor and API."""

    def __init__(self, gains, cfg: SimConfig = SimConfig(), leader_traj=None):
        self.cfg = cfg
        self._runner = LeaderFollowerSimulation(
            gains=gains,
            cfg=cfg,
            leader_traj=leader_traj,
        )

    def run(self):
        return self._runner.run()
