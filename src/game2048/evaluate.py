"""Benchmark an agent over many episodes.

    uv run eval-2048 --agent greedy --episodes 200
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from game2048.agents import AGENTS, Agent


@dataclass
class EvalResult:
    scores: list[int]
    max_tiles: list[int]
    moves: list[int]
    steps: int
    seconds: float

    def summary(self) -> str:
        scores = np.array(self.scores)
        tiles = Counter(self.max_tiles)
        n = len(self.scores)
        lines = [
            f"episodes: {n}   steps/sec: {self.steps / max(self.seconds, 1e-9):,.0f}",
            f"score: mean {scores.mean():,.0f}  median {np.median(scores):,.0f}  "
            f"min {scores.min():,}  max {scores.max():,}",
            f"moves: mean {np.mean(self.moves):,.0f}",
            "max tile reached:",
        ]
        cumulative = 0
        for tile in sorted(tiles, reverse=True):
            cumulative += tiles[tile]
            lines.append(
                f"  {tile:>6}: {tiles[tile]:>5} ({tiles[tile] / n:6.1%})   >= {tile}: {cumulative / n:6.1%}"
            )
        return "\n".join(lines)


def evaluate(env: gym.Env, agent: Agent, episodes: int, seed: int | None = None) -> EvalResult:
    scores, max_tiles, moves = [], [], []
    steps = 0
    start = time.perf_counter()
    for ep in range(episodes):
        obs, info = env.reset(seed=None if seed is None else seed + ep)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(agent(obs, info))
            steps += 1
            done = terminated or truncated
        scores.append(info["score"])
        max_tiles.append(info["max_tile"])
        moves.append(info["moves"])
    return EvalResult(scores, max_tiles, moves, steps, time.perf_counter() - start)


def main(argv: list[str] | None = None) -> None:
    import game2048  # noqa: F401  (registers the environment)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent", choices=sorted(AGENTS), default="random")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--render", action="store_true", help="print every move")
    args = parser.parse_args(argv)

    env = gym.make(
        "Game2048-v0", size=args.size, render_mode="human" if args.render else None
    )
    agent = AGENTS[args.agent](seed=args.seed)
    print(f"agent: {args.agent}")
    print(evaluate(env, agent, args.episodes, seed=args.seed).summary())
    env.close()


if __name__ == "__main__":
    main()
