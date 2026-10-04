"""External simulator dependency locations and sibling-repo helpers."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


EXTERNAL_REPOS = {
    "gym-pybullet-drones": {
        "url": "https://github.com/utiasDSL/gym-pybullet-drones.git",
        "directory": "gym-pybullet-drones",
    },
    "ardupilot": {
        "url": "https://github.com/ArduPilot/ardupilot.git",
        "directory": "ardupilot",
        "ref": "Copter-4.5.7",
    },
    "ardupilot_gazebo": {
        "url": "https://github.com/ArduPilot/ardupilot_gazebo.git",
        "directory": "ardupilot_gazebo",
        "commit": "082a0fe231f6e63bc8d1598f1cba461d9e2ea7f5",
    },
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sibling_repo(name: str) -> Path:
    """Return a repository path immediately beside this project folder."""
    return project_root().parent / name


def ensure_external_repo(name: str, *, update: bool = False) -> Path:
    """Clone an external simulator repo as a project sibling when missing."""
    if name not in EXTERNAL_REPOS:
        raise KeyError(f"Unknown external repository {name!r}")

    meta = EXTERNAL_REPOS[name]
    destination = sibling_repo(meta["directory"])

    if destination.exists():
        if not (destination / ".git").exists():
            raise RuntimeError(
                f"Expected sibling repository at {destination}, but it is not a Git repository."
            )
        if update:
            subprocess.run(["git", "-C", str(destination), "pull", "--ff-only"], check=True)
        return destination

    command = ["git", "clone", "--depth", "1"]
    if meta.get("ref"):
        command += ["--branch", meta["ref"]]
    subprocess.run(command + [meta["url"], str(destination)], check=True)
    if meta.get("commit"):
        subprocess.run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", meta["commit"]], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", meta["commit"]], check=True)
    return destination


def dependency_status() -> dict:
    return {
        name: {"path": str(sibling_repo(meta["directory"])), "present": sibling_repo(meta["directory"]).exists()}
        for name, meta in EXTERNAL_REPOS.items()
    }


def print_dependency_status() -> None:
    for name, status in dependency_status().items():
        state = "present" if status["present"] else "missing"
        print(f"{name:24} {state:8} {status['path']}")
