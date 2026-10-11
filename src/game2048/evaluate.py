"""Benchmark an agent over many episodes.

    uv run eval-2048 --agent greedy --episodes 200
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from game2048.agents import AGENTS, Agent, PPOAgent
from game2048.env import Game2048Env


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


class BatchAgent(Protocol):
    def act_batch(self, boards: np.ndarray, masks: np.ndarray) -> np.ndarray: ...


def evaluate_batched(
    agent: BatchAgent,
    episodes: int,
    seed: int | None = None,
    batch_size: int = 256,
    **env_kwargs: Any,
) -> EvalResult:
    """Like :func:`evaluate`, but plays up to ``batch_size`` games in lockstep.

    The agent picks moves for every live game in one ``act_batch`` call, which is
    far cheaper than one call per move for neural-network agents. Episode ``i``
    always uses seed ``seed + i``, so a deterministic agent gets exactly the
    results :func:`evaluate` would give.
    """
    envs = [Game2048Env(**env_kwargs) for _ in range(min(batch_size, episodes))]
    scores, max_tiles, moves = [0] * episodes, [0] * episodes, [0] * episodes
    slot_episode: list[int | None] = [None] * len(envs)
    boards = np.zeros((len(envs), *envs[0].observation_space.shape), dtype=np.uint8)
    masks = np.zeros((len(envs), 4), dtype=bool)
    next_episode = 0
    steps = 0
    start = time.perf_counter()

    def start_episode(slot: int) -> None:
        nonlocal next_episode
        if next_episode >= episodes:
            slot_episode[slot] = None
            return
        ep = next_episode
        next_episode += 1
        boards[slot], info = envs[slot].reset(seed=None if seed is None else seed + ep)
        masks[slot] = info["action_mask"]
        slot_episode[slot] = ep

    for slot in range(len(envs)):
        start_episode(slot)

    while live := [s for s, ep in enumerate(slot_episode) if ep is not None]:
        actions = agent.act_batch(boards[live], masks[live])
        for slot, action in zip(live, actions):
            boards[slot], _, terminated, truncated, info = envs[slot].step(int(action))
            masks[slot] = info["action_mask"]
            steps += 1
            if terminated or truncated:
                ep = slot_episode[slot]
                scores[ep], max_tiles[ep], moves[ep] = info["score"], info["max_tile"], info["moves"]
                start_episode(slot)
    return EvalResult(scores, max_tiles, moves, steps, time.perf_counter() - start)


def main(argv: list[str] | None = None) -> None:
    import game2048  # noqa: F401  (registers the environment)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent", choices=sorted(AGENTS), default="random")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--render", action="store_true", help="print every move")
    parser.add_argument("--model", help="path to a trained checkpoint (required for --agent ppo)")
    parser.add_argument(
        "--stochastic", action="store_true", help="sample actions instead of taking the argmax"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="games played in lockstep (ppo only)"
    )
    parser.add_argument("--threads", type=int, default=4, help="torch CPU threads (ppo only)")
    args = parser.parse_args(argv)

    print(f"agent: {args.agent}" + (f" ({args.model})" if args.model else ""))
    if args.agent == "ppo":
        if not args.model:
            parser.error("--agent ppo requires --model PATH")
        import torch

        torch.set_num_threads(args.threads)
        agent = PPOAgent(
            args.model, seed=args.seed, size=args.size, deterministic=not args.stochastic
        )
        if not args.render:
            result = evaluate_batched(
                agent, args.episodes, seed=args.seed, batch_size=args.batch_size, size=args.size
            )
            print(result.summary())
            return
    else:
        agent = AGENTS[args.agent](seed=args.seed)

    env = gym.make(
        "Game2048-v0", size=args.size, render_mode="human" if args.render else None
    )
    print(evaluate(env, agent, args.episodes, seed=args.seed).summary())
    env.close()


if __name__ == "__main__":
    main()
