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
    """A trained ``MaskablePPO`` checkpoint on one-hot observations.

    Works with both MLP checkpoints (flattened ``(channels * size * size,)`` input)
    and CNN checkpoints (``(channels, size, size)`` input); the layout is read from
    the checkpoint's observation space.
    """

    def __init__(
        self,
        path: str,
        seed: int | None = None,
        size: int = 4,
        deterministic: bool = True,
    ) -> None:
        # Imported here so the baselines don't pull in torch.
        from pathlib import Path

        from sb3_contrib import MaskablePPO

        if not (Path(path).is_file() or Path(f"{path}.zip").is_file()):
            raise FileNotFoundError(f"no checkpoint at {path!r}")
        self.model = MaskablePPO.load(path, device="cpu")
        if seed is not None:
            self.model.set_random_seed(seed)
        self.deterministic = deterministic
        self.encode = OneHotBoard(Game2048Env(size=size))  # reuse its .observation()
        grid = self.encode.observation_space.shape
        flat = (int(np.prod(grid)),)
        model_shape = self.model.observation_space.shape
        if model_shape not in (grid, flat):
            raise ValueError(
                f"model expects observations of shape {model_shape}, but a one-hot "
                f"{size}x{size} board is {grid} (or {flat} flattened)"
            )
        self.flatten = model_shape == flat

    def __call__(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        return int(self.act_batch(obs[None], info["action_mask"][None])[0])

    def act_batch(self, boards: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Pick moves for ``(N, size, size)`` boards with ``(N, 4)`` masks in one forward pass."""
        x = self.encode.observation(boards)
        if self.flatten:
            x = x.reshape(len(boards), -1)
        actions, _ = self.model.predict(x, action_masks=masks, deterministic=self.deterministic)
        return actions

    def action_probs(self, board: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """The policy's probability for each of the 4 moves (0 for illegal ones)."""
        import torch

        x = self.encode.observation(board[None])
        if self.flatten:
            x = x.reshape(1, -1)
        obs, _ = self.model.policy.obs_to_tensor(x)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs, action_masks=mask[None])
        return dist.distribution.probs[0].cpu().numpy()


BASELINES: dict[str, type] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "corner": CornerAgent,
}

AGENTS: dict[str, type] = {**BASELINES, "ppo": PPOAgent}
