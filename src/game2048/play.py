"""Play 2048 yourself in the terminal, through the same environment the agent uses.

    uv run play-2048
"""

from __future__ import annotations

import argparse

import gymnasium as gym

from game2048.game import DOWN, LEFT, RIGHT, UP

KEYS = {"w": UP, "d": RIGHT, "s": DOWN, "a": LEFT}


def main(argv: list[str] | None = None) -> None:
    import game2048  # noqa: F401  (registers the environment)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    env = gym.make("Game2048-v0", size=args.size, render_mode="ansi", max_invalid_moves=None)
    _, info = env.reset(seed=args.seed)
    print(env.render())
    while True:
        key = input("move [w/a/s/d, q to quit]: ").strip().lower()
        if key == "q":
            break
        if key not in KEYS:
            continue
        _, reward, terminated, _, info = env.step(KEYS[key])
        print(env.render())
        if not info["valid_move"]:
            print("(that move doesn't change the board)")
        elif reward:
            print(f"+{reward:g}")
        if terminated:
            print(f"game over! score {info['score']}, max tile {info['max_tile']}")
            break
    env.close()


if __name__ == "__main__":
    main()
