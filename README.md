# 2048-ai

A [Gymnasium](https://gymnasium.farama.org/) environment for training reinforcement-learning agents to play 2048.

```bash
uv sync
uv run pytest            # run the test suite
uv run main.py           # one random episode, printed
uv run play-2048         # play it yourself (w/a/s/d)
uv run eval-2048 --agent corner --episodes 500   # benchmark a baseline
```

## Using the environment

```python
import gymnasium as gym
import game2048  # registers "Game2048-v0"

env = gym.make("Game2048-v0", reward_mode="log", max_episode_steps=5000)
obs, info = env.reset(seed=42)
while True:
    action = env.action_space.sample(mask=info["action_mask"].astype("int8"))
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
print(info["score"], info["max_tile"])
```

| | |
|---|---|
| **Observation** | `(4, 4)` `uint8` tile exponents: `0` empty, `1` = 2, `2` = 4, … `11` = 2048 |
| **Actions** | `Discrete(4)`: `0` up, `1` right, `2` down, `3` left |
| **Reward** | depends on `reward_mode` (below) |
| **Terminated** | no legal move left, `win_tile` reached, or too many consecutive illegal moves |
| **Info** | `score`, `max_tile`, `moves`, `steps`, `valid_move`, `merges`, `won`, `action_mask` |

### Options (`gym.make("Game2048-v0", **kwargs)`)

| kwarg | default | meaning |
|---|---|---|
| `reward_mode` | `"score"` | `"score"`: value of each merged tile (the game score). `"log"`: log2 of each merged tile (smaller range, usually more stable). `"merges"`: number of merges |
| `invalid_move_reward` | `0.0` | reward for a move that doesn't change the board (no tile spawns) |
| `max_invalid_moves` | `16` | end the episode after this many illegal moves in a row (`None` = never) |
| `win_tile` | `None` | end the episode once this tile is made, e.g. `2048` |
| `size` | `4` | board side length (3x3 boards are handy for quick experiments) |
| `four_prob` | `0.1` | probability that a spawned tile is a 4 |
| `render_mode` | `None` | `"human"` (prints), `"ansi"` (returns a string), `"rgb_array"` (image frame, works with `RecordVideo`) |

You can also start from a fixed position: `env.reset(options={"board": exponent_array})`.

### Action masking

Illegal moves (ones that change nothing) are allowed but wasteful. `info["action_mask"]` gives the legal moves for the next step, and `env.unwrapped.action_masks()` returns the same mask in the format `sb3-contrib`'s `MaskablePPO` expects. With DQN-style methods, set the Q-values of illegal actions to `-inf` before taking the argmax.

### Observation wrappers

```python
from gymnasium.wrappers import FlattenObservation
from game2048 import OneHotBoard, NormalizedBoard

env = OneHotBoard(gym.make("Game2048-v0"))                       # (18, 4, 4) float32, for CNNs
env = FlattenObservation(OneHotBoard(gym.make("Game2048-v0")))   # (288,) float32, for MLPs
env = NormalizedBoard(gym.make("Game2048-v0"))                   # (4, 4) float32 in [0, 1]
```

One-hot encoding generally trains much better than raw exponents, because a 1024 tile isn't "twice" a 512 tile in any way a network can use.

### Vectorised training

```python
envs = gym.make_vec("Game2048-v0", num_envs=16, vectorization_mode="async")
```

## Baselines

200 episodes each (`uv run eval-2048 --agent <name> --episodes 200`):

| agent | mean score | reaches 256 | reaches 512 |
|---|---|---|---|
| `random` (random legal move) | ~1,060 | 7.5% | 0% |
| `greedy` (best immediate merge) | ~2,070 | 50% | 2.5% |
| `corner` (prefer down > left > right > up) | ~2,560 | 49% | 5% |

A learned agent should beat `corner` comfortably. Strong agents reach 2048 in most games.

## Layout

```
src/game2048/
  game.py      # pure game logic (no Gym dependency), MoveResult, Game2048
  env.py       # Game2048Env (Gymnasium)
  wrappers.py  # OneHotBoard, NormalizedBoard
  render.py    # dependency-free RGB renderer
  agents.py    # random / greedy / corner baselines
  evaluate.py  # benchmark CLI (eval-2048)
  play.py      # human play CLI (play-2048)
tests/         # game rules, Gymnasium API compliance, wrappers, agents
```
