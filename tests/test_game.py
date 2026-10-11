import numpy as np
import pytest

from game2048.game import DOWN, LEFT, RIGHT, UP, Game2048, legal_actions_mask, move, spawn_tile


def b(values):
    """Build an exponent board from tile values (0 = empty)."""
    arr = np.array(values)
    out = np.zeros(arr.shape, dtype=np.uint8)
    nz = arr > 0
    out[nz] = np.log2(arr[nz]).astype(np.uint8)
    return out


def row(values):
    return b([values])


@pytest.mark.parametrize(
    ("before", "after", "score"),
    [
        ([2, 2, 0, 0], [4, 0, 0, 0], 4),
        ([2, 0, 0, 2], [4, 0, 0, 0], 4),
        ([2, 2, 2, 2], [4, 4, 0, 0], 8),
        ([2, 2, 4, 0], [4, 4, 0, 0], 4),  # a merged tile can't merge again this move
        ([4, 2, 2, 0], [4, 4, 0, 0], 4),
        ([2, 2, 2, 0], [4, 2, 0, 0], 4),  # leading pair merges first
        ([8, 8, 4, 4], [16, 8, 0, 0], 24),
        ([2, 4, 8, 16], [2, 4, 8, 16], 0),
        ([0, 0, 0, 2], [2, 0, 0, 0], 0),
    ],
)
def test_left_slide(before, after, score):
    result = move(row(before), LEFT)
    np.testing.assert_array_equal(result.board, row(after))
    assert result.score == score
    assert result.moved == (before != after)


def test_directions():
    board = b([
        [2, 0, 0, 2],
        [0, 0, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 4],
    ])
    np.testing.assert_array_equal(move(board, LEFT).board, b([[4, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0]]))
    np.testing.assert_array_equal(move(board, RIGHT).board, b([[0, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 4]]))
    np.testing.assert_array_equal(move(board, UP).board, b([[4, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0]]))
    np.testing.assert_array_equal(move(board, DOWN).board, b([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2], [4, 0, 0, 4]]))


def test_move_is_pure():
    board = b([[2, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4])
    copy = board.copy()
    move(board, LEFT)
    np.testing.assert_array_equal(board, copy)


def test_log_score_and_merges():
    result = move(row([8, 8, 4, 4]), LEFT)
    assert result.log_score == 4 + 3  # 16 = 2^4, 8 = 2^3
    assert result.merges == 2


def test_legal_mask_on_stuck_board():
    stuck = b([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    assert not legal_actions_mask(stuck).any()
    corner = b([[2, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4])
    np.testing.assert_array_equal(legal_actions_mask(corner), [False, True, True, False])


def test_spawn_distribution():
    rng = np.random.default_rng(0)
    fours = 0
    for _ in range(5000):
        board = np.zeros((4, 4), dtype=np.uint8)
        spawn_tile(board, rng, four_prob=0.1)
        assert np.count_nonzero(board) == 1
        fours += int(board.max() == 2)
    assert 0.08 < fours / 5000 < 0.12


def test_game_starts_with_two_tiles_and_tracks_score():
    game = Game2048(rng=np.random.default_rng(1))
    assert np.count_nonzero(game.board) == 2
    game.board = b([[2, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4])
    game._next = None
    result = game.step(LEFT)
    assert result.moved and game.score == 4 and game.moves == 1
    assert np.count_nonzero(game.board) == 2  # merged tile + spawned tile


def test_illegal_step_changes_nothing():
    game = Game2048(rng=np.random.default_rng(1))
    game.board = b([[2, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4])
    game._next = None
    before = game.board.copy()
    result = game.step(LEFT)
    assert not result.moved
    np.testing.assert_array_equal(game.board, before)
    assert game.moves == 0


@pytest.mark.parametrize("size", [3, 5])
def test_other_board_sizes(size):
    game = Game2048(size=size, rng=np.random.default_rng(0))
    while not game.is_over():
        game.step(int(np.flatnonzero(game.legal_actions_mask())[0]))
    assert game.board.shape == (size, size)
