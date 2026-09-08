"""Simulator discovery and construction."""

from __future__ import annotations

from typing import Callable

from .base import SimConfig, SimulationBackend


_BACKENDS: dict[str, Callable] = {}


def register_backend(name: str, factory: Callable) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("Simulator name cannot be empty")
    _BACKENDS[key] = factory


def available_simulators() -> list[str]:
    return sorted(_BACKENDS)


def create_simulator(name: str, cfg: SimConfig) -> SimulationBackend:
    key = name.strip().lower()
    try:
        factory = _BACKENDS[key]
    except KeyError as exc:
        available = ", ".join(available_simulators())
        raise KeyError(
            f"Unknown simulator '{name}'. Available simulators: {available}"
        ) from exc
    return factory(cfg)
