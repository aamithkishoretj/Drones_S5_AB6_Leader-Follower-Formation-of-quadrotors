"""
Bridges the paper's dual-quaternion kinematic controller (dq_control/) to
gym-pybullet-drones' CtrlAviary + DSLPIDControl.

Cascade per control step (see README "Design notes" for the rationale):

    1. dq_control.controller.KinematicController.compute(...)   -> (omega_cmd, v_cmd)
       This is exactly eq. (5)-(9) of the paper: a *kinematic* twist command.

    2. dq_control.dual_quaternion.integrate_pose(...)            -> next-step
       target pose (position + attitude), by integrating the twist for one
       control timestep (eq. 3). This plays the role of the instantaneous
       position/attitude setpoint that the real Bebop 2's onboard firmware
       would track given the same velocity/attitude-rate commands.

    3. gym_pybullet_drones.control.DSLPIDControl.computeControlFromState(...)
       consumes that target pose and outputs motor RPMs -- this stands in
       for the Bebop's low-level (onboard) controller.

This file intentionally follows the exact same obs/action pattern used in
gym-pybullet-drones' own examples (e.g. `examples/pid.py`), so it should be
robust across recent versions of the package. If your installed version
renamed a class/kwarg, the two `import` lines at the top are the only place
you should need to touch.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import os
import sys
import collections
import collections.abc
import numpy as np

# Py3.10+ moved these ABCs out of `collections` into `collections.abc`.
# The old `gym` package (dependency of gym_pybullet_drones 0.6.0 / v1.0.0)
# still references collections.Mapping directly, so restore the old aliases
# before anything downstream imports `gym`.
for _name in ("Mapping", "MutableMapping", "Sequence", "Set", "Callable"):
    if not hasattr(collections, _name) and hasattr(collections.abc, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

# --- make sure gym-pybullet-drones is importable -----------------------------
def _add_sibling_to_path():
    _here = os.path.dirname(os.path.abspath(__file__))
    _sibling = os.path.abspath(os.path.join(_here, "..", "..", "gym-pybullet-drones"))
    if os.path.isdir(_sibling) and _sibling not in sys.path:
        sys.path.insert(0, _sibling)


# gym-pybullet-drones has two generations of API:
#   NEW (>=2.0, gymnasium-based): enums live in gym_pybullet_drones.utils.enums,
#       CtrlAviary uses pyb_freq/ctrl_freq kwargs, reset()->(obs,info),
#       step()->(obs,reward,terminated,truncated,info).
#   OLD (<=1.0.0 / package version 0.6.0, plain `gym`-based): enums live in
#       gym_pybullet_drones.envs.BaseAviary, CtrlAviary uses freq/aggregate_phy_steps,
#       reset()->obs, step()->(obs,reward,done,info).
# We detect which one is installed once, at import time, and normalize the
# differences behind the GPD_API_VERSION flag + the _reset_env/_step_env
# helpers below, so the rest of this file doesn't need to care.
try:
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
    GPD_API_VERSION = "new"
except ImportError:
    _add_sibling_to_path()
    try:
        from gym_pybullet_drones.utils.enums import DroneModel, Physics
        from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
        from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
        GPD_API_VERSION = "new"
    except ImportError:
        # Old (v1.0.0 / 0.6.0) layout: enums are defined in BaseAviary.
        from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
        from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
        from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
        GPD_API_VERSION = "old"

print(f"[leader_follower_sim] detected gym-pybullet-drones API generation: {GPD_API_VERSION}")

from dq_control import (
    Quaternion,
    DualQuaternion,
    integrate_pose,
    KinematicController,
    LeaderTrajectory,
    FollowerTrajectory,
)


@dataclass
class SimConfig:
    duration_sec: float = 30.0
    pyb_freq: int = 240          # physics steps / s
    ctrl_freq: int = 48          # our kinematic controller + low-level PID steps / s
    gui: bool = False
    drone_model: "DroneModel" = None  # filled in __post_init__ (avoids import at module load)
    x_offset: float = 1.85       # follower offset along x, eq. (16)
    output_folder: str = "results"

    def __post_init__(self):
        if self.drone_model is None:
            self.drone_model = DroneModel.CF2X


def _get_drone_obs(obs, i: int) -> np.ndarray:
    """Normalizes the various obs container formats seen across
    gym-pybullet-drones versions:
      - a plain (N, 20) array
      - a dict keyed by int drone index -> (20,) array
      - a dict keyed by str drone index -> (20,) array
      - a dict keyed by int/str drone index -> {"state": (20,) array, "neighbors": ...}
        (the old ObservationType.KIN "Dict" space format used by some
        RL-oriented aviaries in gym-pybullet-drones v1.0.0 / 0.6.0)
    """
    if isinstance(obs, dict):
        if i in obs:
            row = obs[i]
        elif str(i) in obs:
            row = obs[str(i)]
        else:
            raise KeyError(
                f"Could not find drone index {i} in obs dict; "
                f"available keys: {list(obs.keys())}"
            )
    else:
        row = obs[i]

    # Nested Dict-space observation: {"state": array(20,), "neighbors": ...}
    if isinstance(row, dict):
        if "state" in row:
            row = row["state"]
        else:
            raise KeyError(
                f"Drone {i}'s obs is a dict without a 'state' key; "
                f"available keys: {list(row.keys())}"
            )

    result = np.asarray(row, dtype=float).reshape(-1)
    if result.size < 13:
        raise ValueError(
            f"Drone {i}'s obs vector has only {result.size} elements after "
            f"normalization (expected >=13: pos[3]+quat[4]+rpy[3]+vel[3]...); "
            f"raw obs was: {row!r}"
        )
    return result


def _quat_xyzw_from_obs(obs_row: np.ndarray) -> Quaternion:
    """gym-pybullet-drones kin obs layout: [x,y,z, qx,qy,qz,qw, r,p,y, vx,vy,vz, wx,wy,wz, rpms(4)]."""
    return Quaternion.from_array(obs_row[3:7])


def _pos_from_obs(obs_row: np.ndarray) -> np.ndarray:
    return obs_row[0:3].copy()


def _vel_from_obs(obs_row: np.ndarray) -> np.ndarray:
    return obs_row[10:13].copy()


class LeaderFollowerSim:
    """Runs the two-drone dual-quaternion leader-follower experiment."""

    def __init__(
        self,
        gains: dict,
        cfg: SimConfig = SimConfig(),
        leader_traj: LeaderTrajectory | None = None,
    ):
        self.cfg = cfg
        self.leader_traj = leader_traj or LeaderTrajectory()
        self.follower_traj = FollowerTrajectory(x_offset=cfg.x_offset)

        self.leader_ctrl = KinematicController(gains["leader"])
        self.follower_ctrl = KinematicController(gains["follower"])

        init_leader = self.leader_traj.position(0.0)
        init_follower = init_leader + np.array([cfg.x_offset, 0.0, 0.0])
        self.initial_xyzs = np.array([init_leader, init_follower])
        self.initial_rpys = np.zeros((2, 3))

        common_kwargs = dict(
            drone_model=cfg.drone_model,
            num_drones=2,
            initial_xyzs=self.initial_xyzs,
            initial_rpys=self.initial_rpys,
            physics=Physics.PYB,
            gui=cfg.gui,
            record=False,
            obstacles=False,
            user_debug_gui=False,
        )
        if GPD_API_VERSION == "new":
            self.env = CtrlAviary(
                **common_kwargs,
                pyb_freq=cfg.pyb_freq,
                ctrl_freq=cfg.ctrl_freq,
                output_folder=cfg.output_folder,
            )
        else:
            # Old API: single `freq` (physics rate) + `aggregate_phy_steps`
            # (number of physics steps per control step). ctrl_freq is derived.
            aggregate_phy_steps = max(1, cfg.pyb_freq // cfg.ctrl_freq)
            self.env = CtrlAviary(
                **common_kwargs,
                freq=cfg.pyb_freq,
                aggregate_phy_steps=aggregate_phy_steps,
            )
            cfg.ctrl_freq = cfg.pyb_freq // aggregate_phy_steps  # keep it consistent

        self.pid = [DSLPIDControl(drone_model=cfg.drone_model) for _ in range(2)]
        self.ctrl_timestep = 1.0 / cfg.ctrl_freq

    # ------------------------------------------------------------------
    def _reset_env(self):
        """Normalizes gym (old, obs only) vs gymnasium (new, obs+info) reset()."""
        result = self.env.reset()
        if GPD_API_VERSION == "new":
            obs, _info = result
            return obs
        return result

    def _step_env(self, action):
        """Normalizes gym (old, 4-tuple) vs gymnasium (new, 5-tuple) step()."""
        result = self.env.step(action)
        if GPD_API_VERSION == "new":
            obs, reward, terminated, truncated, info = result
            return obs
        obs, reward, done, info = result
        return obs

    # ------------------------------------------------------------------
    def _target_from_twist(self, Q_current: DualQuaternion, omega_cmd, v_cmd):
        """Integrate the kinematic twist one control step to get a target pose."""
        Q_target = integrate_pose(Q_current, omega_cmd, v_cmd, self.ctrl_timestep)
        return Q_target.position(), Q_target.attitude().to_rpy()

    def run(self):
        cfg = self.cfg
        num_steps = int(cfg.duration_sec * cfg.ctrl_freq)

        self.log: dict = {k: [] for k in (
            "t",
            "leader_pos", "leader_pos_d", "leader_rpy", "leader_rpy_d",
            "follower_pos", "follower_pos_d", "follower_rpy", "follower_rpy_d",
        )}

        obs = self._reset_env()
        # Some gym-pybullet-drones versions (e.g. 0.6.0 / v1.0.0) use a Dict
        # action space keyed by *string* drone indices ("0", "1", ...) instead
        # of a plain (NUM_DRONES, 4) array. Detect and build the right container.
        action_is_dict = hasattr(self.env.action_space, "spaces")
        action = {"0": np.zeros(4), "1": np.zeros(4)} if action_is_dict else np.zeros((2, 4))

        for step in range(num_steps):
            t = step * self.ctrl_timestep

            leader_pos = _pos_from_obs(_get_drone_obs(obs, 0))
            leader_att = _quat_xyzw_from_obs(_get_drone_obs(obs, 0))
            follower_pos = _pos_from_obs(_get_drone_obs(obs, 1))
            follower_att = _quat_xyzw_from_obs(_get_drone_obs(obs, 1))

            Q_leader = DualQuaternion.from_pose(leader_pos, leader_att)
            Q_follower = DualQuaternion.from_pose(follower_pos, follower_att)

            # --- leader: analytic desired trajectory ------------------------
            Qd_leader = self.leader_traj.desired_pose(t)
            omega_d_L, v_d_L = self.leader_traj.desired_twist(t)
            omega_cmd_L, v_cmd_L = self.leader_ctrl.compute(
                Q_leader, Qd_leader, omega_d_L, v_d_L, self.ctrl_timestep
            )
            target_pos_L, target_rpy_L = self._target_from_twist(Q_leader, omega_cmd_L, v_cmd_L)

            # --- follower: desired trajectory built from *measured* leader --
            Qd_follower, omega_d_F, v_d_F = self.follower_traj.update(
                t, leader_pos, leader_att, self.ctrl_timestep
            )
            omega_cmd_F, v_cmd_F = self.follower_ctrl.compute(
                Q_follower, Qd_follower, omega_d_F, v_d_F, self.ctrl_timestep
            )
            target_pos_F, target_rpy_F = self._target_from_twist(Q_follower, omega_cmd_F, v_cmd_F)

            # --- low-level PID -> RPMs (stands in for the Bebop firmware) ---
            rpm0, _, _ = self.pid[0].computeControlFromState(
                control_timestep=self.ctrl_timestep,
                state=_get_drone_obs(obs, 0),
                target_pos=target_pos_L,
                target_rpy=target_rpy_L,
            )
            rpm1, _, _ = self.pid[1].computeControlFromState(
                control_timestep=self.ctrl_timestep,
                state=_get_drone_obs(obs, 1),
                target_pos=target_pos_F,
                target_rpy=target_rpy_F,
            )
            if action_is_dict:
                action["0"] = rpm0
                action["1"] = rpm1
            else:
                action[0, :] = rpm0
                action[1, :] = rpm1

            obs = self._step_env(action)

            self.log["t"].append(t)
            self.log["leader_pos"].append(leader_pos)
            self.log["leader_pos_d"].append(Qd_leader.position())
            self.log["leader_rpy"].append(leader_att.to_rpy())
            self.log["leader_rpy_d"].append(Qd_leader.attitude().to_rpy())
            self.log["follower_pos"].append(follower_pos)
            self.log["follower_pos_d"].append(Qd_follower.position())
            self.log["follower_rpy"].append(follower_att.to_rpy())
            self.log["follower_rpy_d"].append(Qd_follower.attitude().to_rpy())

            if cfg.gui:
                self.env.render()

        self.env.close()
        return {k: np.array(v) for k, v in self.log.items()}