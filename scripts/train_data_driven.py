"""Collect simulator transitions, identify dynamics, and evaluate DQ predictive control."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dq_control import get_gains, LeaderTrajectory, LemniscateParams, PotatoChipTrajectory, PotatoChipParams, Quaternion
from dq_control.course_trajectories import BSplineTrajectory
from dq_control.learned_control import fit_response, file_digest
from simulators import SimConfig, LeaderFollowerSimulation


def metrics(log):
    result = {name: {
        "position_rmse_m": float(np.sqrt(np.mean(np.sum((log[name+"_pos"]-log[name+"_pos_d"])**2, axis=1)))),
        "altitude_min_m": float(log[name+"_pos"][:, 2].min()),
        "altitude_max_m": float(log[name+"_pos"][:, 2].max()),
    } for name in ("leader", "follower")}
    for name in result:
        angles = []
        for actual, desired in zip(log[name+"_rpy"], log[name+"_rpy_d"]):
            relative = Quaternion.from_rpy(actual).conj()*Quaternion.from_rpy(desired)
            angles.append(2*np.arctan2(np.linalg.norm(relative.vec), abs(relative.scalar)))
        result[name]["attitude_rmse_rad"] = float(np.sqrt(np.mean(np.square(angles))))
    return result


def collect(backend, duration, seed, trajectory, excitation, controller="analytic", model=None):
    cfg = SimConfig(simulator=backend, duration_sec=duration, seed=seed, trails=False,
                    excitation=excitation, controller=controller, learned_model=str(model) if model else None)
    sim = LeaderFollowerSimulation(get_gains("real_eig"), cfg, trajectory)
    log = sim.run()
    if not all(np.all(np.isfinite(a)) for a in log.values()):
        raise RuntimeError("Non-finite simulation samples; refusing training")
    return log, sim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator", choices=["pybullet", "mujoco", "gazebo", "kinematic"], default="pybullet")
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT/"data"/"identification")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--reuse-data", action="store_true", help="Refit from previously collected episodes")
    args = parser.parse_args()
    if args.duration < 5 or args.episodes < 2:
        parser.error("Use duration >= 5 and episodes >= 2")
    folder = args.output/args.simulator
    folder.mkdir(parents=True, exist_ok=True)
    model_path = args.model or ROOT/"models"/f"{args.simulator}_response.npz"
    training, validation, provenance = [], [], []
    for episode in range(args.episodes+1):
        split = "train" if episode < args.episodes else "validation"
        seed = 1200+episode
        rng = np.random.default_rng(seed)
        trajectory = LeaderTrajectory(LemniscateParams(r_x=rng.uniform(.5, 1.), r_y=rng.uniform(.35, .7),
                   w_d=rng.uniform(.12, .23), z0=rng.uniform(.9, 1.4))) if episode % 2 == 0 else PotatoChipTrajectory(
                   PotatoChipParams(r=rng.uniform(.5, .9), w=rng.uniform(.12, .23), z_amp=.15, z0=1.2))
        path = folder/f"{split}_{episode:02d}.npz"
        if args.reuse_data:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["episode_metadata"]))
                expected = {"split": split, "seed": seed, "backend": args.simulator,
                            "duration": args.duration, "dt": 1/48, "excitation": .18}
                if any(metadata.get(k) != v for k, v in expected.items()):
                    raise ValueError(f"Collection settings differ for {path}; recollect without --reuse-data")
                log = {k: data[k] for k in data.files if k != "episode_metadata"}
        else:
            print(f"[training] Collecting {split} episode {episode+1}/{args.episodes+1}", flush=True)
            log, _ = collect(args.simulator, args.duration, seed, trajectory, .18)
            np.savez_compressed(path, **log, episode_metadata=np.asarray(json.dumps({
                "split": split, "seed": seed, "backend": args.simulator,
                "duration": args.duration, "dt": 1/48, "excitation": .18,
                "trajectory": type(trajectory).__name__, "parameters": vars(trajectory.p)})))
        (training if split == "train" else validation).append(log)
        provenance.append({"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                           "sha256": file_digest(path), "split": split})
    def rows(logs, key):
        return np.concatenate([d[key].reshape(-1, d[key].shape[-1]) for d in logs])
    x, u, y = [rows(training, key) for key in ("response_x", "response_u", "response_y")]
    model = fit_response(x, u, y, {"backend": args.simulator, "dt": 1/48,
                                   "provenance": provenance, "training_episodes": args.episodes})
    if args.simulator == "gazebo":
        # Gazebo's low-level loop already chooses tilt from translation.
        # Large independent roll/pitch rates let the optimizer exploit weak
        # learned cross-couplings and destabilize tracking. Keep these small;
        # yaw retains its independently learned bound.
        model.limit[3:5] = np.minimum(model.limit[3:5], .15)
        model.metadata["roll_pitch_rate_cap_rad_s"] = .15
        model.metadata["constraint_selection"] = "Tuned on simulator tracking evaluations, not an untouched test set"
    vx, vu, vy = [rows(validation, key) for key in ("response_x", "response_u", "response_y")]
    model.metadata["validation_rmse_per_state"] = np.sqrt(np.mean((model.predict(vx, vu)-vy)**2, axis=0)).tolist()
    model.metadata["persistence_rmse_per_state"] = np.sqrt(np.mean((vx-vy)**2, axis=0)).tolist()
    model.save(model_path)
    print(f"[training] Model saved: {model_path}", flush=True)
    report = {"model": str(model_path), "model_sha256": file_digest(model_path), "identification": model.metadata,
              "evaluation": {}, "scope": "Local simulator-trained closed-loop response, not real-flight validation"}
    # Whole new trajectories, not shuffled samples from training flights.
    for name in ("unseen_lemniscate", "unseen_bspline"):
        report["evaluation"][name] = {}
        for controller in ("analytic", "learned"):
            trajectory = LeaderTrajectory(LemniscateParams(r_x=.72, r_y=.48, w_d=.18, z0=1.1)) if name == "unseen_lemniscate" else BSplineTrajectory(
                [[0, 0, 1], [.7, 0, 1.1], [1.5, .7, 1.3], [2., -.3, 1.4], [2.8, 0, 1.1]], args.duration)
            print(f"[evaluation] {name}: {controller}", flush=True)
            log, sim = collect(args.simulator, args.duration, 9001, trajectory, 0., controller, model_path)
            np.savez_compressed(folder/f"eval_{name}_{controller}.npz", **log)
            result = metrics(log)
            if controller == "learned":
                result["optimization"] = {"leader_solves": sim.leader_ctrl.solve_count,
                    "follower_solves": sim.follower_ctrl.solve_count,
                    "leader_saturated_steps": sim.leader_ctrl.saturated_steps,
                    "follower_saturated_steps": sim.follower_ctrl.saturated_steps}
            report["evaluation"][name][controller] = result
            (folder/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["evaluation"], indent=2))
    print(f"[training] Validation report: {folder/'report.json'}")


if __name__ == "__main__":
    main()
