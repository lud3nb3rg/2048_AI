"""Observation wrappers that re-encode the exponent board for neural networks."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class OneHotBoard(gym.ObservationWrapper):
    """One-hot encode tile exponents into a ``(channels, size, size)`` ``float32`` tensor.

    Channel ``k`` is 1 where the tile has exponent ``k`` (channel 0 marks empty cells).
    Channels-first, so it plugs straight into a ``Conv2d``. Flatten it with
    ``gymnasium.wrappers.FlattenObservation`` for an MLP.
    """

    def __init__(self, env: gym.Env, channels: int | None = None) -> None:
        super().__init__(env)
        board_space = env.observation_space
        assert isinstance(board_space, spaces.Box)
        size = board_space.shape[0]
        self.channels = channels if channels is not None else int(board_space.high.max()) + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.channels, size, size), dtype=np.float32
        )
        self._eye = np.eye(self.channels, dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        clipped = np.minimum(observation, self.channels - 1)
        return np.ascontiguousarray(np.moveaxis(self._eye[clipped], -1, 0))


class NormalizedBoard(gym.ObservationWrapper):
    """Scale exponents into ``[0, 1]`` as ``float32`` (exponent / max exponent)."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        board_space = env.observation_space
        assert isinstance(board_space, spaces.Box)
        self._scale = float(board_space.high.max())
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=board_space.shape, dtype=np.float32
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation.astype(np.float32) / self._scale
