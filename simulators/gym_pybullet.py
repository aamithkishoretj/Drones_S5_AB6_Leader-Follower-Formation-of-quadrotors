"""gym-pybullet-drones backend for the original simulation."""

from __future__ import annotations

import collections
import collections.abc
import os
import sys
import numpy as np

from .base import DroneState, SimConfig, SimulationBackend, VehicleCommand
from .dependencies import sibling_repo


for _name in ("Mapping", "MutableMapping", "Sequence", "Set", "Callable"):
    if not hasattr(collections, _name) and hasattr(collections.abc, _name):
        setattr(collections, _name, getattr(collections.abc, _name))


GPD_API_VERSION = None
DroneModel = Physics = CtrlAviary = DSLPIDControl = None


def _load_gpd():
    global GPD_API_VERSION, DroneModel, Physics, CtrlAviary, DSLPIDControl
    if GPD_API_VERSION is not None:
        return

    try:
        from gym_pybullet_drones.utils.enums import DroneModel as _DroneModel, Physics as _Physics
        from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary as _CtrlAviary
        from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl as _DSLPIDControl
        DroneModel, Physics = _DroneModel, _Physics
        CtrlAviary, DSLPIDControl = _CtrlAviary, _DSLPIDControl
        GPD_API_VERSION = "new"
        return
    except ImportError:
        pass

    sibling = str(sibling_repo("gym-pybullet-drones"))
    if os.path.isdir(sibling) and sibling not in sys.path:
        sys.path.insert(0, sibling)

    try:
        from gym_pybullet_drones.utils.enums import DroneModel as _DroneModel, Physics as _Physics
        from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary as _CtrlAviary
        from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl as _DSLPIDControl
        DroneModel, Physics = _DroneModel, _Physics
        CtrlAviary, DSLPIDControl = _CtrlAviary, _DSLPIDControl
        GPD_API_VERSION = "new"
        return
    except ImportError:
        pass

    try:
        from gym_pybullet_drones.envs.BaseAviary import DroneModel as _DroneModel, Physics as _Physics
        from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary as _CtrlAviary
        from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl as _DSLPIDControl
        DroneModel, Physics = _DroneModel, _Physics
        CtrlAviary, DSLPIDControl = _CtrlAviary, _DSLPIDControl
        GPD_API_VERSION = "old"
    except ImportError as exc:
        raise RuntimeError(
            "gym-pybullet-drones is required for --simulator pybullet. "
            "Install it or clone it as a sibling repository with:\n"
            "  python scripts/setup_simulators.py --simulator pybullet"
        ) from exc


def _get_drone_obs(obs, i: int) -> np.ndarray:
    if isinstance(obs, dict):
        if i in obs:
            row = obs[i]
        elif str(i) in obs:
            row = obs[str(i)]
        else:
            raise KeyError(f"Could not find drone index {i}; keys={list(obs.keys())}")
    else:
        row = obs[i]

    if isinstance(row, dict):
        row = row.get("state", row)

    result = np.asarray(row, dtype=float).reshape(-1)
    if result.size < 16:
        raise ValueError(f"Drone {i} observation has {result.size} values; expected >=16")
    return result


class GymPyBulletDronesBackend(SimulationBackend):
    name = "pybullet"

    def __init__(self, cfg: SimConfig):
        _load_gpd()
        self.cfg = cfg
        self.env = None
        self.pid = None
        self.obs = None
        self.action = None
        self.action_is_dict = False
        self.ctrl_timestep = cfg.control_timestep()

        if cfg.drone_model is None:
            cfg.drone_model = DroneModel.CF2X

    def reset(self, initial_xyzs: np.ndarray, initial_rpys: np.ndarray) -> None:
        common_kwargs = dict(
            drone_model=self.cfg.drone_model,
            num_drones=2,
            initial_xyzs=np.asarray(initial_xyzs, dtype=float),
            initial_rpys=np.asarray(initial_rpys, dtype=float),
            physics=Physics.PYB,
            gui=self.cfg.gui,
            record=False,
            obstacles=False,
            user_debug_gui=False,
        )
        if GPD_API_VERSION == "new":
            self.env = CtrlAviary(
                **common_kwargs,
                pyb_freq=self.cfg.pyb_freq,
                ctrl_freq=self.cfg.ctrl_freq,
                output_folder=self.cfg.output_folder,
            )
        else:
            aggregate_phy_steps = max(1, self.cfg.pyb_freq // self.cfg.ctrl_freq)
            self.env = CtrlAviary(
                **common_kwargs,
                freq=self.cfg.pyb_freq,
                aggregate_phy_steps=aggregate_phy_steps,
            )
            self.cfg.ctrl_freq = self.cfg.pyb_freq // aggregate_phy_steps
            self.ctrl_timestep = 1.0 / self.cfg.ctrl_freq

        self.pid = [DSLPIDControl(drone_model=self.cfg.drone_model) for _ in range(2)]
        result = self.env.reset()
        self.obs = result[0] if GPD_API_VERSION == "new" else result
        self.action_is_dict = hasattr(self.env.action_space, "spaces")
        self.action = {"0": np.zeros(4), "1": np.zeros(4)} if self.action_is_dict else np.zeros((2, 4))

    @staticmethod
    def _state_from_obs(row: np.ndarray) -> DroneState:
        attitude = __import__("dq_control", fromlist=["Quaternion"]).Quaternion.from_array(row[3:7])
        return DroneState(
            position=row[0:3],
            attitude=attitude,
            velocity=row[10:13],
            angular_velocity=row[13:16],
            motor_rpm=row[16:20] if row.size >= 20 else np.zeros(4),
        )

    def get_states(self) -> list[DroneState]:
        if self.obs is None:
            raise RuntimeError("PyBullet backend has not been reset")
        return [
            self._state_from_obs(_get_drone_obs(self.obs, 0)),
            self._state_from_obs(_get_drone_obs(self.obs, 1)),
        ]

    @staticmethod
    def _state_to_gpd_array(state: DroneState) -> np.ndarray:
        rpy = state.attitude.to_rpy()
        row = np.zeros(20, dtype=float)
        row[0:3] = state.position
        row[3:7] = state.attitude.as_array()
        row[7:10] = rpy
        row[10:13] = state.velocity
        row[13:16] = state.angular_velocity
        row[16:20] = state.motor_rpm
        return row

    def apply_commands(self, commands: list[VehicleCommand]) -> None:
        states = self.get_states()
        rpm_values = []
        for i, cmd in enumerate(commands):
            rpm, _, _ = self.pid[i].computeControlFromState(
                control_timestep=self.ctrl_timestep,
                state=self._state_to_gpd_array(states[i]),
                target_pos=cmd.target_pos,
                target_rpy=cmd.target_rpy,
                target_vel=cmd.target_vel,
                target_rpy_rates=cmd.target_rpy_rates,
            )
            rpm_values.append(np.asarray(rpm, dtype=float))

        if self.action_is_dict:
            for i, rpm in enumerate(rpm_values):
                self.action[str(i)] = rpm
        else:
            for i, rpm in enumerate(rpm_values):
                self.action[i, :] = rpm

    def step(self) -> list[DroneState]:
        if self.env is None:
            raise RuntimeError("PyBullet backend has not been reset")

        result = self.env.step(self.action)
        self.obs = result[0] if GPD_API_VERSION == "new" else result[0]
        if self.cfg.gui:
            self.env.render()
        return self.get_states()

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
