"""Baseline policies to benchmark learned agents against.

Each agent is a callable ``agent(obs, info) -> action`` that works on the raw
(unwrapped) exponent-board observation and uses ``info["action_mask"]``.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from game2048.env import Game2048Env
from game2048.game import ACTIONS, DOWN, LEFT, RIGHT, UP, move
from game2048.wrappers import OneHotBoard


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


class PPOAgent:
    """A trained ``MaskablePPO`` checkpoint (one-hot, flattened observations)."""

    def __init__(
        self,
        path: str,
        seed: int | None = None,
        size: int = 4,
        deterministic: bool = True,
    ) -> None:
        # Imported here so the baselines don't pull in torch.
        from sb3_contrib import MaskablePPO

        self.model = MaskablePPO.load(path, device="cpu")
        if seed is not None:
            self.model.set_random_seed(seed)
        self.deterministic = deterministic
        self.encode = OneHotBoard(Game2048Env(size=size))  # reuse its .observation()
        expected = (int(np.prod(self.encode.observation_space.shape)),)
        if self.model.observation_space.shape != expected:
            raise ValueError(
                f"model expects observations of shape {self.model.observation_space.shape}, "
                f"but a one-hot {size}x{size} board flattens to {expected}"
            )

    def __call__(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        x = self.encode.observation(obs).ravel()
        action, _ = self.model.predict(
            x, action_masks=info["action_mask"], deterministic=self.deterministic
        )
        return int(action)


BASELINES: dict[str, type] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "corner": CornerAgent,
}

AGENTS: dict[str, type] = {**BASELINES, "ppo": PPOAgent}
