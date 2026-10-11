"""Baseline policies to benchmark learned agents against.

Each agent is a callable ``agent(obs, info) -> action`` that works on the raw
(unwrapped) exponent-board observation and uses ``info["action_mask"]``.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from game2048.game import ACTIONS, DOWN, LEFT, RIGHT, UP, move


class Agent(Protocol):
    def __call__(self, obs: np.ndarray, info: dict[str, Any]) -> int: ...


def _legal(info: dict[str, Any]) -> list[int]:
    return [a for a in ACTIONS if info["action_mask"][a]] or list(ACTIONS)


class RandomAgent:
    """Uniformly random legal move. The floor any learner must beat."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        return int(self.rng.choice(_legal(info)))


class GreedyAgent:
    """Legal move with the highest immediate merge score (ties broken randomly)."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        legal = _legal(info)
        scores = np.array([move(obs, a).score for a in legal])
        best = np.flatnonzero(scores == scores.max())
        return legal[int(self.rng.choice(best))]


class CornerAgent:
    """Classic human heuristic: prefer down, then left, then right, and up only if forced."""

    PRIORITY = (DOWN, LEFT, RIGHT, UP)

    def __init__(self, seed: int | None = None) -> None:
        pass  # deterministic; accepts a seed for a uniform constructor

    def __call__(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        mask = info["action_mask"]
        return next((a for a in self.PRIORITY if mask[a]), DOWN)


AGENTS: dict[str, type] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "corner": CornerAgent,
}
