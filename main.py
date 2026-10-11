"""Minimal tour of the 2048 environment: one episode with a random legal-move policy."""

import gymnasium as gym
import numpy as np

import game2048  # noqa: F401  (registers "Game2048-v0")


def main():
    env = gym.make("Game2048-v0", render_mode="ansi", reward_mode="log")
    rng = np.random.default_rng(0)

    obs, info = env.reset(seed=0)
    print(env.render())
    total_reward = 0.0
    done = False
    while not done:
        legal = np.flatnonzero(info["action_mask"])
        obs, reward, terminated, truncated, info = env.step(int(rng.choice(legal)))
        total_reward += reward
        done = terminated or truncated

    print(env.render())
    print(f"return: {total_reward:g}   score: {info['score']}   max tile: {info['max_tile']}")
    env.close()


if __name__ == "__main__":
    main()
