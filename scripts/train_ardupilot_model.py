#!/usr/bin/env python3
"""Fit measured command-response dynamics; evaluate on entire held-out flights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dq_control import DualQuaternion, Quaternion
from dq_control.data_driven import IdentifiedDynamics, heading_rotation, yaw_rate, features


def pairs(paths, dt):
    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as a:
            if json.loads(str(a['metadata_json'])).get('simulator') != 'ardupilot':
                raise ValueError('Training data must come from ArduPilot')
            for vehicle in range(2):
                t, state, u = a['t'][:, vehicle], a['state'][:, vehicle], a['u'][:, vehicle]
                # Keep the last sample if telemetry timestamps repeat.
                keep = np.r_[np.diff(t)>0, True]
                t, state, u = t[keep], state[keep], u[keep]
                if np.any(np.diff(t)<=0) or np.max(np.diff(t))>.5:
                    raise ValueError('Non-monotonic or stalled telemetry in dataset')
                # Exclude the initial settling after takeoff/teacher engagement.
                # The identified model is intended for airborne tracking.
                grid = np.arange(max(t[0], 5.), t[-1]-dt, dt)
                rotations = Slerp(t, Rotation.from_quat(state[:, 3:7]))
                for start in grid:
                    end = start+dt
                    p = np.array([np.interp(start, t, state[:, j]) for j in range(3)])
                    q = Quaternion.from_array(rotations(start).as_quat())
                    Q = DualQuaternion.from_pose(p, q)
                    velocity = np.array([np.interp(start,t,state[:,j]) for j in range(7,10)])
                    omega = np.array([np.interp(start,t,state[:,j]) for j in range(10,13)])
                    vn = np.array([np.interp(end,t,state[:,j]) for j in range(7,10)])
                    qn = Quaternion.from_array(rotations(end).as_quat())
                    wn = np.array([np.interp(end,t,state[:,j]) for j in range(10,13)])
                    # Time-weighted zero-order-held applied input, not requested input.
                    edges = np.r_[start, t[(t>start)&(t<end)], end]
                    index = np.searchsorted(t, edges[:-1], side='right')-1
                    command = np.sum(u[index]*np.diff(edges)[:,None],axis=0)/dt
                    R = heading_rotation(Q)
                    s = np.r_[R.T @ velocity, yaw_rate(Q,omega)]
                    target = np.r_[R.T @ vn, yaw_rate(DualQuaternion.from_pose(p,qn),wn),
                                   (qn.rotation_matrix().T @ np.array([0.,0.,1.]))[:2]]
                    local_command = np.r_[R.T @ command[:3],command[3]]
                    records.append((Q,s,local_command,target))
    if not records:
        raise ValueError('No transitions available')
    return records


def design(records, quadratic):
    return np.array([np.r_[features(s,Q,quadratic),u] for Q,s,u,y in records]), np.array([y for Q,s,u,y in records])


def train(train_paths, validation_paths, model_path, report_path, dt=.2):
    if {p.resolve() for p in train_paths} & {p.resolve() for p in validation_paths}:
        raise ValueError('Training and validation must be separate flights')
    hashes = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in train_paths+validation_paths}
    if {hashes[p] for p in train_paths} & {hashes[p] for p in validation_paths}:
        raise ValueError('Training and validation contain identical flight data')
    train_rows, val_rows = pairs(train_paths,dt), pairs(validation_paths,dt)
    reports, candidates = [], []
    for quadratic in (False, True):
        X,Y = design(train_rows,quadratic)
        V,T = design(val_rows,quadratic)
        scale = np.maximum(np.sqrt(np.mean(X**2,axis=0)),.02)
        for ridge in (.001,.01,.1,1.):
            Z = X/scale
            W = np.linalg.solve(Z.T@Z+ridge*np.eye(Z.shape[1]), Z.T@Y)/scale[:,None]
            # Shrink learned state increments toward persistence to avoid
            # fitting sensor jitter. Validation selects the regularization;
            # every deployed coefficient still comes from measured dynamics.
            prior = np.zeros_like(W)
            prior[1:7] = np.eye(6)
            for response_blend in (.4, .6, .8, 1.):
                candidate = prior + response_blend*(W-prior)
                predicted = V@candidate
                rmse = np.sqrt(np.mean((predicted-T)**2,axis=0))
                baseline = V[:,1:7]
                persistence = np.sqrt(np.mean((baseline-T)**2,axis=0))
                score = float(np.mean(rmse/np.maximum(persistence,1e-6)))
                spectral_radius = float(np.max(np.abs(np.linalg.eigvals(candidate[1:7].T))))
                control_rank = int(np.linalg.matrix_rank(candidate[-4:]))
                passed = score<1 and bool(np.all(rmse<persistence))
                info = dict(quadratic=quadratic,ridge=ridge,response_blend=response_blend,
                            rmse=rmse.tolist(),persistence_rmse=persistence.tolist(),
                            normalized_score=score,spectral_radius=spectral_radius,
                            control_rank=control_rank,validation_passed=passed)
                reports.append(info)
                if spectral_radius<1. and control_rank==4 and passed:
                    candidates.append((score,candidate,info))
    if not candidates:
        report_path.parent.mkdir(parents=True,exist_ok=True)
        report_path.write_text(json.dumps(dict(validation_passed=False,candidates=reports),indent=2))
        raise RuntimeError('No stable, controllable identified model; collect richer data')
    score,W,best = min(candidates,key=lambda x:x[0])
    passed = score<1 and all(np.array(best['rmse']) < np.array(best['persistence_rmse']))
    metadata = dict(schema=1,simulator='ardupilot',validation_passed=bool(passed),
                    training_files=[str(p.relative_to(ROOT)) for p in train_paths],
                    validation_files=[str(p.relative_to(ROOT)) for p in validation_paths],
                    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in train_paths+validation_paths},
                    feature_order=['constant','heading_vx','heading_vy','vz','yaw_rate','body_up_x','body_up_y'],
                    quadratic_features='state*abs(state)' if best['quadratic'] else None,
                    input_order=['heading_vx_command','heading_vy_command','vz_command','yaw_rate_command'],
                    training_transitions=len(train_rows),validation_transitions=len(val_rows),metrics=best,
                    settling_seconds_excluded=5.,
                    scope='Estimated SITL state response to Guided commands; not motor-level dynamics')
    report = dict(selected=metadata,candidates=reports)
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2))
    if not passed:
        raise RuntimeError(f'Validation failed; report saved to {report_path}')
    IdentifiedDynamics(W,dt,best['quadratic'],metadata).save(model_path)
    print(json.dumps(metadata,indent=2))
    print(f'[model] Saved validated model to {model_path}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,nargs='+',required=True)
    p.add_argument('--validation',type=Path,nargs='+',required=True)
    p.add_argument('--model',type=Path,default=ROOT/'models/ardupilot_dq_dynamics.npz')
    p.add_argument('--report',type=Path,default=ROOT/'models/ardupilot_dq_validation.json')
    a=p.parse_args()
    train([x.resolve() for x in a.train],[x.resolve() for x in a.validation],a.model,a.report)
