#!/usr/bin/env python3
"""Independent one-second open-loop forecasts with measured future inputs."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dq_control import DualQuaternion, Quaternion
from dq_control.data_driven import IdentifiedDynamics, yaw_rate


def validate(model_path, data_path, output):
    model=IdentifiedDynamics.load(model_path)
    results=[]
    with np.load(data_path,allow_pickle=False) as a:
        for vehicle in range(2):
            t,s,u=a['t'][:,vehicle],a['state'][:,vehicle],a['u'][:,vehicle]
            keep=np.r_[np.diff(t)>0,True]
            t,s,u=t[keep],s[keep],u[keep]
            rotations=Slerp(t,Rotation.from_quat(s[:,3:7]))
            def sample(time):
                position=np.array([np.interp(time,t,s[:,j]) for j in range(3)])
                Q=DualQuaternion.from_pose(position,Quaternion.from_array(rotations(time).as_quat()))
                velocity=np.array([np.interp(time,t,s[:,j]) for j in range(7,10)])
                omega=np.array([np.interp(time,t,s[:,j]) for j in range(10,13)])
                return Q,velocity,omega
            for start in np.arange(5.,t[-1]-1.,1.):
                Q,v,w=sample(start)
                pred,pv,pw=Q,v,w
                for j in range(5):
                    left,right=start+j*model.dt,start+(j+1)*model.dt
                    edges=np.r_[left,t[(t>left)&(t<right)],right]
                    indices=np.searchsorted(t,edges[:-1],side='right')-1
                    command=np.sum(u[indices]*np.diff(edges)[:,None],axis=0)/model.dt
                    pred,pv,pw=model.rollout(pred,pv,pw,command,1)
                actual,_,_=sample(start+1.)
                expected_yaw=actual.attitude().to_rpy()[2]
                learned_yaw=pred.attitude().to_rpy()[2]
                baseline_yaw=Q.attitude().to_rpy()[2]+yaw_rate(Q,w)
                wrap=lambda angle: np.arctan2(np.sin(angle),np.cos(angle))
                results.append((actual.position(),pred.position(),Q.position()+v,
                                wrap(learned_yaw-expected_yaw),wrap(baseline_yaw-expected_yaw)))
    actual,learned,baseline=np.array([r[0] for r in results]),np.array([r[1] for r in results]),np.array([r[2] for r in results])
    metrics=dict(forecast_seconds=1.,windows=len(results),
                 position_rmse_m=float(np.sqrt(np.mean(np.sum((learned-actual)**2,axis=1)))),
                 constant_velocity_position_rmse_m=float(np.sqrt(np.mean(np.sum((baseline-actual)**2,axis=1)))),
                 yaw_rmse_rad=float(np.sqrt(np.mean(np.array([r[3] for r in results])**2))),
                 constant_rate_yaw_rmse_rad=float(np.sqrt(np.mean(np.array([r[4] for r in results])**2))),
                 inputs='Time-weighted measured future commands; this is prediction validation, not closed-loop tracking')
    output.mkdir(parents=True,exist_ok=True)
    (output/'forecast_validation.json').write_text(json.dumps(metrics,indent=2))
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    axes[0].plot(np.linalg.norm(learned-actual,axis=1),label='Learned dynamics')
    axes[0].plot(np.linalg.norm(baseline-actual,axis=1),label='Constant velocity')
    axes[0].set_ylabel('1-second position error [m]')
    axes[1].plot([abs(r[3]) for r in results],label='Learned dynamics')
    axes[1].plot([abs(r[4]) for r in results],label='Constant yaw rate')
    axes[1].set_ylabel('1-second yaw error [rad]')
    axes[1].set_xlabel('Forecast window (vehicle 1, then vehicle 2)')
    for ax in axes: ax.legend(); ax.grid(alpha=.25)
    fig.suptitle('Held-out SITL flight: independent one-second forecasts')
    fig.tight_layout()
    fig.savefig(output/'forecast_validation.png',dpi=150)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=ROOT/'models/ardupilot_dq_dynamics.npz')
    p.add_argument('--data',type=Path,default=ROOT/'results/data_driven/training_seed_3.npz')
    p.add_argument('--output',type=Path,default=ROOT/'results/data_driven')
    a=p.parse_args()
    validate(a.model,a.data,a.output)
