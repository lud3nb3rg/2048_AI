"""Core 2048 game logic, independent of any RL framework.

The board is a square ``numpy`` array of tile *exponents*: ``0`` is an empty
cell, ``1`` is a 2-tile, ``2`` is a 4-tile, ``11`` is the 2048-tile, and so on.
Exponents keep values small and bounded, which is what neural networks want.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTIONS = (UP, RIGHT, DOWN, LEFT)
ACTION_NAMES = ("up", "right", "down", "left")


@dataclass(frozen=True, slots=True)
class MoveResult:
    """Outcome of sliding the board in one direction (before a tile spawns)."""

    board: np.ndarray
    score: int
    """Sum of the values of all tiles created by merges (the classic 2048 score)."""
    log_score: int
    """Sum of the exponents of all tiles created by merges."""
    merges: int
    moved: bool


@lru_cache(maxsize=None)
def _slide_row_left(row: tuple[int, ...]) -> tuple[tuple[int, ...], int, int, int]:
    """Slide and merge one row of exponents to the left.

    Each tile merges at most once per move, so ``[2, 2, 2, 2]`` becomes
    ``[4, 4, 0, 0]`` and ``[2, 2, 4, 0]`` becomes ``[4, 4, 0, 0]``.
    """
    tiles = [v for v in row if v]
    out: list[int] = []
    score = log_score = merges = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            exp = tiles[i] + 1
            out.append(exp)
            score += 1 << exp
            log_score += exp
            merges += 1
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    out.extend([0] * (len(row) - len(out)))
    return tuple(out), score, log_score, merges


def _to_left(board: np.ndarray, action: int) -> np.ndarray:
    """View the board so that ``action`` becomes a left slide."""
    if action == LEFT:
        return board
    if action == RIGHT:
        return board[:, ::-1]
    if action == UP:
        return board.T
    if action == DOWN:
        return board.T[:, ::-1]
    raise ValueError(f"invalid action {action!r}; expected one of {ACTIONS}")


def _from_left(board: np.ndarray, action: int) -> np.ndarray:
    """Inverse of :func:`_to_left`."""
    if action == LEFT:
        return board
    if action == RIGHT:
        return board[:, ::-1]
    if action == UP:
        return board.T
    return board[:, ::-1].T  # DOWN


def move(board: np.ndarray, action: int) -> MoveResult:
    """Slide ``board`` in direction ``action`` without spawning a new tile.

    Pure function: ``board`` is not modified.
    """
    rows = _to_left(board, action)
    new_rows = np.empty_like(rows)
    score = log_score = merges = 0
    for i, row in enumerate(rows):
        new_row, s, ls, m = _slide_row_left(tuple(row.tolist()))
        new_rows[i] = new_row
        score += s
        log_score += ls
        merges += m
    new_board = np.ascontiguousarray(_from_left(new_rows, action))
    return MoveResult(
        board=new_board,
        score=score,
        log_score=log_score,
        merges=merges,
        moved=not np.array_equal(new_board, board),
    )


def legal_actions_mask(board: np.ndarray) -> np.ndarray:
    """Boolean mask of shape ``(4,)``: ``True`` where the action changes the board."""
    return np.array([move(board, a).moved for a in ACTIONS], dtype=bool)


def spawn_tile(board: np.ndarray, rng: np.random.Generator, four_prob: float = 0.1) -> None:
    """Place a 2 (or a 4 with probability ``four_prob``) on a random empty cell, in place."""
    empty = np.flatnonzero(board == 0)
    if empty.size == 0:
        raise ValueError("cannot spawn a tile on a full board")
    board.flat[rng.choice(empty)] = 2 if rng.random() < four_prob else 1


def max_exponent(size: int) -> int:
    """Largest exponent reachable on a ``size x size`` board (17, i.e. 131072, for 4x4)."""
    return size * size + 1


class Game2048:
    """A stateful game of 2048.

    Moves for the current board are computed once and cached, so querying the
    legal-action mask and then stepping costs no more than stepping alone.
    """

    def __init__(
        self,
        size: int = 4,
        four_prob: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        if size < 2:
            raise ValueError("board size must be at least 2")
        self.size = size
        self.four_prob = four_prob
        self.rng = rng if rng is not None else np.random.default_rng()
        self.board = np.zeros((size, size), dtype=np.uint8)
        self.score = 0
        self.moves = 0
        self._next: list[MoveResult] | None = None
        self.reset()

    def reset(self, rng: np.random.Generator | None = None) -> None:
        if rng is not None:
            self.rng = rng
        self.board = np.zeros((self.size, self.size), dtype=np.uint8)
        self.score = 0
        self.moves = 0
        spawn_tile(self.board, self.rng, self.four_prob)
        spawn_tile(self.board, self.rng, self.four_prob)
        self._next = None

    def _results(self) -> list[MoveResult]:
        if self._next is None:
            self._next = [move(self.board, a) for a in ACTIONS]
        return self._next

    def peek(self, action: int) -> MoveResult:
        """What ``action`` would do to the current board, without applying it."""
        return self._results()[action]

    def legal_actions_mask(self) -> np.ndarray:
        return np.array([r.moved for r in self._results()], dtype=bool)

    def is_over(self) -> bool:
        return not any(r.moved for r in self._results())

    def step(self, action: int) -> MoveResult:
        """Apply ``action``. If it moves the board, a new tile spawns.

        An illegal move (one that changes nothing) leaves the game untouched.
        """
        result = self.peek(action)
        if result.moved:
            self.board = result.board.copy()
            spawn_tile(self.board, self.rng, self.four_prob)
            self.score += result.score
            self.moves += 1
            self._next = None
        return result

    @property
    def max_tile(self) -> int:
        top = int(self.board.max())
        return 1 << top if top else 0

    def __str__(self) -> str:
        return board_to_str(self.board)


def board_to_str(board: np.ndarray) -> str:
    values = [[(1 << int(e)) if e else 0 for e in row] for row in board]
    width = max(4, max(len(str(v)) for row in values for v in row))
    sep = "+" + "+".join("-" * (width + 2) for _ in board[0]) + "+"
    lines = [sep]
    for row in values:
        cells = (f" {str(v) if v else '.':>{width}} " for v in row)
        lines.append("|" + "|".join(cells) + "|")
        lines.append(sep)
    return "\n".join(lines)
