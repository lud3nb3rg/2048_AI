"""A Gymnasium reinforcement-learning environment for 2048."""

from gymnasium.envs.registration import register

from game2048.env import Game2048Env
from game2048.game import ACTION_NAMES, DOWN, LEFT, RIGHT, UP, Game2048
from game2048.wrappers import NormalizedBoard, OneHotBoard

register(id="Game2048-v0", entry_point="game2048.env:Game2048Env")

__all__ = [
    "ACTION_NAMES",
    "DOWN",
    "LEFT",
    "RIGHT",
    "UP",
    "Game2048",
    "Game2048Env",
    "NormalizedBoard",
    "OneHotBoard",
]
