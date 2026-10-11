import json
import re

import numpy as np
import pytest

from game2048.agents import CornerAgent
from game2048.evaluate import evaluate
from game2048.env import Game2048Env
from game2048.game import move
from game2048.render import render_rgb
from game2048.replay import record_game, write_gif, write_html


def test_record_game_matches_evaluation():
    frames = record_game(CornerAgent(), seed=3)
    result = evaluate(Game2048Env(), CornerAgent(), episodes=1, seed=3)
    assert frames[-1]["s"] == result.scores[0]
    assert frames[-1]["a"] is None  # final frame is the game-over board
    assert all(f["a"] is not None for f in frames[:-1])
    assert sum(f["m"][f["a"]] for f in frames[:-1]) == result.moves[0]


def test_frames_are_consistent():
    frames = record_game(CornerAgent(), seed=0)
    for prev, cur in zip(frames, frames[1:]):
        assert cur["s"] == prev["s"] + cur["g"]
        afterstate = move(np.array(prev["b"], dtype=np.uint8).reshape(4, 4), prev["a"]).board.ravel()
        if cur["sp"] is not None:
            assert afterstate[cur["sp"]] == 0  # spawned on a cell the move left free
            assert cur["b"][cur["sp"]] in (1, 2)
            assert np.count_nonzero(np.array(cur["b"]) != afterstate) == 1


def test_write_html_embeds_data(tmp_path):
    frames = record_game(CornerAgent(), seed=0)
    out = tmp_path / "r.html"
    write_html(frames, out, {"agent": "corner", "seed": 0, "score": frames[-1]["s"]}, size=4)
    html = out.read_text(encoding="utf-8")
    assert "__REPLAY_DATA__" not in html
    data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1))
    assert len(data["frames"]) == len(frames)


def test_write_gif(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    frames = record_game(CornerAgent(), seed=0)[:20]
    out = tmp_path / "r.gif"
    write_gif(frames, out, size=4, tile=32)
    with PIL.open(out) as img:
        assert img.n_frames == 20
        for i in (0, 19):
            img.seek(i)
            board = np.array(frames[i]["b"], dtype=np.uint8).reshape(4, 4)
            np.testing.assert_array_equal(np.array(img.convert("RGB")), render_rgb(board, 32, 3))
