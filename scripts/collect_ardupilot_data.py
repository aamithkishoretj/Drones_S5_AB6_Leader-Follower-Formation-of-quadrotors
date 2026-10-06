#!/usr/bin/env python3
"""Generate isolated, persistently excited SITL flights for identification."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dq_control import DualQuaternion, Quaternion, KinematicController, get_gains
from simulators.base import SimConfig, VehicleCommand
from simulators.ardupilot import ArduPilotBackend


def collect(duration, seed, output, ardupilot_path=None):
    cfg = SimConfig(simulator='ardupilot', ctrl_freq=24, duration_sec=duration,
                    output_folder=str(output), ardupilot_low_resource=True,
                    ardupilot_startup_timeout=180, ardupilot_path=ardupilot_path)
    backend = ArduPilotBackend(cfg)
    anchors = np.array([[0., 0., 2.2], [10., 0., 2.2]])
    controllers = [KinematicController(get_gains('real_eig')['leader']) for _ in range(2)]
    rng = np.random.default_rng(seed)
    phases = rng.uniform(-np.pi, np.pi, (2, 4))
    rows = []
    backend.reset(anchors, np.zeros((2, 3)))
    try:
        previous = 0.
        while backend.elapsed_time < duration:
            states = backend.get_states()
            t = backend.elapsed_time
            dt = max(t - previous, 1/24)
            previous = t
            commands = []
            for i, state in enumerate(states):
                frequencies = np.array([.14, .19, .23, .17]) * (1 + .08*seed + .1*i)
                angles = frequencies*t + phases[i]
                amplitudes = np.array([1.7, 1.5, .35, .8])
                target = anchors[i] + amplitudes[:3]*np.sin(angles[:3])
                velocity = amplitudes[:3]*frequencies[:3]*np.cos(angles[:3])
                yaw = amplitudes[3]*np.sin(angles[3])
                yaw_rate = amplitudes[3]*frequencies[3]*np.cos(angles[3])
                Qd = DualQuaternion.from_pose(target, Quaternion.from_rpy([0, 0, yaw]))
                omega, vel = controllers[i].compute(state.as_pose(), Qd, np.array([0., 0., yaw_rate]), velocity, dt)
                # Independent smooth excitation breaks command/state correlation.
                excitation = .16*np.sin(np.array([.83, 1.13, .71, 1.37])*t + phases[i])
                vel += excitation[:3]
                omega[2] += excitation[3]
                if np.linalg.norm(vel) > .7:
                    vel *= .7/np.linalg.norm(vel)
                command = VehicleCommand(target, [0, 0, yaw], vel, omega)
                commands.append(command)
            backend.apply_commands(commands)
            rows.append((np.array([v.boot_time-backend.start_time for v in backend.vehicles]),
                         np.array([np.r_[s.position, s.attitude.as_array(), s.velocity, s.angular_velocity] for s in states]),
                         np.array([np.r_[s.as_pose().P.as_array(), s.as_pose().D.as_array()] for s in states]),
                         backend.last_applied_commands.copy()))
            backend.step()
    finally:
        backend.close()
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'training_seed_{seed}.npz'
    times, states, poses, inputs = map(np.array, zip(*rows))
    metadata = dict(schema=1, simulator='ardupilot', seed=seed, duration=duration,
                    state_columns=['px','py','pz','qx','qy','qz','qw','vx','vy','vz','wx','wy','wz'],
                    input_columns=['vx_command_enu','vy_command_enu','vz_command_enu','yaw_rate_enu'],
                    pose='unit dual quaternion, xyzw principal then xyzw dual',
                    input_capture='Exact post-limiter command recorded by the backend at transmission',
                    source='ArduPilot estimated telemetry, not independent Gazebo ground truth')
    np.savez_compressed(path, t=times, state=states, dual_quaternion=poses, u=inputs,
                        metadata_json=np.array(json.dumps(metadata)))
    print(f'[dataset] Saved {len(rows)} paired vehicle samples to {path}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--duration', type=float, default=60.)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--output', type=Path, default=ROOT/'results/data_driven')
    p.add_argument('--ardupilot_path', help='Existing SITL source/build root')
    a = p.parse_args()
    if not np.isfinite(a.duration) or a.duration <= 0:
        p.error('duration must be finite and positive')
    collect(a.duration, a.seed, a.output, a.ardupilot_path)
