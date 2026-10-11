"""Gymnasium environment for 2048."""

from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from game2048.game import ACTION_NAMES, Game2048, MoveResult, board_to_str, max_exponent
from game2048.render import render_rgb

RewardMode = Literal["score", "log", "merges"]


class Game2048Env(gym.Env):
    """2048 as a Gymnasium environment.

    Observation
        ``(size, size)`` ``uint8`` array of tile exponents (0 = empty, 1 = 2, 2 = 4, ...).
        Use :class:`game2048.wrappers.OneHotBoard` for a one-hot ``(channels, size, size)``
        encoding, which usually trains much better.

    Actions
        ``Discrete(4)``: 0 = up, 1 = right, 2 = down, 3 = left.

    Reward (``reward_mode``)
        ``"score"``  -- value of every tile created by a merge (the in-game score).
        ``"log"``    -- log2 of every tile created by a merge; keeps rewards in a small range.
        ``"merges"`` -- number of merges in the move.
        Moves that don't change the board earn ``invalid_move_reward`` instead.

    Episode end
        ``terminated`` when no legal move remains, when ``win_tile`` is reached (if set),
        or after ``max_invalid_moves`` consecutive illegal moves (if set).
        Use ``gym.make(..., max_episode_steps=N)`` for truncation.

    ``info`` carries ``action_mask`` (legal moves for the *new* state), so masked
    policies never need to waste steps on illegal moves. :meth:`action_masks` exposes
    the same mask for ``sb3-contrib``'s ``MaskablePPO``.
    """

    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = None,
        size: int = 4,
        reward_mode: RewardMode = "score",
        invalid_move_reward: float = 0.0,
        max_invalid_moves: int | None = 16,
        win_tile: int | None = None,
        four_prob: float = 0.1,
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        if reward_mode not in ("score", "log", "merges"):
            raise ValueError(f"unsupported reward_mode {reward_mode!r}")
        if win_tile is not None and (win_tile < 4 or win_tile & (win_tile - 1)):
            raise ValueError("win_tile must be a power of two >= 4")

        self.render_mode = render_mode
        self.size = size
        self.reward_mode = reward_mode
        self.invalid_move_reward = float(invalid_move_reward)
        self.max_invalid_moves = max_invalid_moves
        self.win_tile = win_tile

        self.observation_space = spaces.Box(
            low=0, high=max_exponent(size), shape=(size, size), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(4)

        self.game = Game2048(size=size, four_prob=four_prob)
        self._invalid_streak = 0
        self._steps = 0
        self._last_action: int | None = None

    # ------------------------------------------------------------------ API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.game.reset(rng=self.np_random)
        if options and "board" in options:
            board = np.asarray(options["board"], dtype=np.uint8)
            if board.shape != (self.size, self.size):
                raise ValueError(f"options['board'] must have shape {(self.size, self.size)}")
            self.game.board = board.copy()
            self.game._next = None
        self._invalid_streak = 0
        self._steps = 0
        self._last_action = None
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info(valid_move=True, merges=0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}")
        action = int(action)
        result = self.game.step(action)
        self._steps += 1
        self._last_action = action

        if result.moved:
            self._invalid_streak = 0
            reward = self._reward(result)
        else:
            self._invalid_streak += 1
            reward = self.invalid_move_reward

        won = self.win_tile is not None and self.game.max_tile >= self.win_tile
        too_many_invalid = (
            self.max_invalid_moves is not None and self._invalid_streak >= self.max_invalid_moves
        )
        terminated = won or too_many_invalid or self.game.is_over()

        info = self._info(valid_move=result.moved, merges=result.merges)
        info["won"] = won
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        """Legal-action mask for the current state (``sb3-contrib`` convention)."""
        return self.game.legal_actions_mask()

    def render(self) -> str | np.ndarray | None:
        if self.render_mode is None:
            gym.logger.warn("render() called without a render_mode; pass render_mode to make().")
            return None
        if self.render_mode == "rgb_array":
            return render_rgb(self.game.board)
        text = self._ansi()
        if self.render_mode == "ansi":
            return text
        print(text + "\n")
        return None

    # -------------------------------------------------------------- helpers

    def _reward(self, result: MoveResult) -> float:
        if self.reward_mode == "score":
            return float(result.score)
        if self.reward_mode == "log":
            return float(result.log_score)
        return float(result.merges)

    def _obs(self) -> np.ndarray:
        return self.game.board.copy()

    def _info(self, *, valid_move: bool, merges: int) -> dict[str, Any]:
        return {
            "score": self.game.score,
            "max_tile": self.game.max_tile,
            "moves": self.game.moves,
            "steps": self._steps,
            "valid_move": valid_move,
            "merges": merges,
            "action_mask": self.game.legal_actions_mask(),
        }

    def _ansi(self) -> str:
        last = ACTION_NAMES[self._last_action] if self._last_action is not None else "-"
        header = (
            f"score: {self.game.score}  max: {self.game.max_tile}  "
            f"moves: {self.game.moves}  last: {last}"
        )
        return header + "\n" + board_to_str(self.game.board)
