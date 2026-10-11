# 2048-ai

A reinforcement-learning agent that learns to play 2048, built on a custom [Gymnasium](https://gymnasium.farama.org/)
environment and [MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html) from sb3-contrib.

<p align="center">
  <img src="docs/replay.gif" width="320" alt="The trained agent playing 2048 and reaching the 4096 tile">
</p>

*The best model (128-filter CNN, 30M training steps) playing its best of 200 games: 74,612 points, reaching 4096 in
3,551 moves. Shown at 50 moves per second.*

| Agent (500 games, seeds 10,000–10,499) | Mean score | Reaches 512 | Reaches 1024 | Reaches 2048 | Reaches 4096 |
|---|---|---|---|---|---|
| **PPO, CNN (128 filters), γ=0.995, 30M steps (best)** | **32,580** | **99.6%** | **96.2%** | **69.8%** | **10.0%** |
| PPO, CNN (64 filters), γ=0.99, 10M steps | 17,405 | 97.4% | 79.6% | 18.8% | 0% |
| PPO, MLP, 1M steps | 3,733 | 27.4% | 0.8% | 0% | 0% |
| `corner` heuristic (down > left > right > up) | 2,600 | 5.6% | 0% | 0% | 0% |

On 2,000 fresh games (seeds 20,000–21,999), the best model averages 34,447 points, reaches 2048 in 75.2% of games
and 4096 in 12.6%, and made an 8192 tile three times.

## Quick start

```bash
uv sync
uv run pytest                                  # 42 tests: game rules, Gymnasium API, agents, replays
uv run play-2048                               # play it yourself (w/a/s/d)
uv run eval-2048 --agent corner --episodes 500 --seed 10000

# Train the best model (~70 min on a single GPU; drop --device cuda to train on the CPU, which is far slower)
uv run python -u scripts/train_ppo.py --run-name ppo_cnn128_norm_g0995_30M --device cuda \
  --filters 128 --gamma 0.995 --total-timesteps 30000000 --lr-decay-from 0.6666667 --checkpoint-every 1000000

uv run eval-2048 --agent ppo --model checkpoints/ppo_cnn128_norm_g0995_30M/final_30M.zip --episodes 500 --seed 10000
uv run replay-2048 --model checkpoints/ppo_cnn128_norm_g0995_30M/final_30M.zip --best-of 200 --open
```

Checkpoints aren't committed (`checkpoints/` is git-ignored), so train a model before evaluating or replaying it.

---

## How the best model was reached

All numbers below come from the same evaluation: 500 games on seeds 10,000–10,499, which no model trained on, with
deterministic play (the most likely legal move) unless noted.

### 1. The environment

`Game2048-v0` (in [src/game2048/](src/game2048/)) is a Gymnasium environment on top of a framework-free game engine:

- **Observation:** the 4×4 board as tile exponents (`0` empty, `1` = 2, … `11` = 2048).
- **Actions:** up, right, down, left. `info["action_mask"]` and `env.action_masks()` give the legal moves, so the
  agent never wastes a step on a move that doesn't change the board.
- **Reward:** configurable (see [Reward design](#reward-design)). All the agents below use `"log"`.

### 2. Baselines

Simple policies give a floor any learner has to clear:

| Baseline | Mean score | Reaches 512 |
|---|---|---|
| `random` (random legal move) | ~1,060 | 0% |
| `greedy` (largest immediate merge) | ~2,070 | 2.5% |
| `corner` (prefer down > left > right > up) | 2,600 | 5.6% |

### 3. MaskablePPO with an MLP

The first agent: MaskablePPO with action masking, a one-hot board (18 channels × 4 × 4, flattened to 288 inputs),
a 256×256 MLP, 16 parallel games and log rewards.

**After 1M steps: 3,733 mean, 27.4% reach 512.** It clearly learned something, but slowly.

### 4. A CNN, and why it failed at first

The next model replaced the MLP with a small CNN built for 2048 ([networks.py](src/game2048/networks.py)). It uses
1×2 and 2×1 filters, because merges only happen between horizontally or vertically adjacent cells. There is no
pooling, because tile positions matter.

**After 1M steps it was worse: 2,549 mean, 4.2% reach 512.** The training logs showed why. The value network's
explained variance stayed at about 0 for the whole run, and its loss stayed flat around 1,000. Returns in "log"
reward units are in the hundreds, too large for the value head to fit, so it predicted a constant. Without a
working value estimate, PPO's advantage estimates are mostly noise. The policy improved briefly, then drifted and
became overconfident (entropy fell to −0.32).

### 5. Making training and evaluation faster

Profiling showed where the time went, and four changes followed:

| Change | Why |
|---|---|
| CNN filters 128 → 64 | Network updates were ~70% of training time; 64 filters cut the update cost by ~3× |
| `torch.set_num_threads(4)` | 16 threads were *slower* than 4 at this size on a 6-performance-core CPU |
| `DummyVecEnv` instead of `SubprocVecEnv` | A game step (~86 µs) is cheaper than the inter-process round trips; 2× faster env stepping |
| Batched evaluation | Plays 256 games in lockstep with one network call per move: **18× faster**, identical results |

### 6. The fixes: reward normalization and separate networks (0–5M steps)

- **`VecNormalize(norm_reward=True)`** scales returns to roughly unit size, which the value head can fit.
- **`share_features_extractor=False`** gives the policy and the value function separate CNNs, so large value-loss
  gradients can't disrupt the policy's features.

Explained variance reached 0.68 within 100k steps, against about 0 for the whole first CNN run, and the agent kept
improving for all 5M steps:

| Checkpoint | Mean score | Reaches 512 | Reaches 1024 | Reaches 2048 |
|---|---|---|---|---|
| 600k | 5,039 | 45.4% | 7.4% | 0% |
| 1M | 7,196 | 72.6% | 20.4% | 0.4% |
| 2M | 10,425 | 88.2% | 40.8% | 3.2% |
| 3M | 10,986 | 89.2% | 48.0% | 3.6% |
| 4M | 12,734 | 92.2% | 61.4% | 6.8% |
| **5M** | **13,569** | **93.8%** | **63.0%** | **8.2%** |

At 1M steps this model roughly doubles the MLP's score at the same number of steps. Training ran at ~520 steps/s,
so 5M steps took about 2.6 hours on an Intel Core Ultra 7 155H (CPU only).

### 7. Training longer with a decaying learning rate

The model was still improving at 5M steps, but its progress was uneven: flat from 2.5M to 3.5M, then a jump. So
training resumed from the 5M checkpoint for another 5M steps, with the learning rate falling linearly from 3e-4 to
0 over the new segment. The reward-normalization statistics were carried over, so the value targets kept the same
scale.

| Checkpoint | Learning rate | Mean score | Reaches 512 | Reaches 1024 | Reaches 2048 |
|---|---|---|---|---|---|
| 5M (start) | 3e-4 | 13,569 | 93.8% | 63.0% | 8.2% |
| 6M | 2.4e-4 | 14,912 | 95.4% | 68.8% | 14.4% |
| 7M | 1.8e-4 | 15,269 | 95.6% | 68.8% | 13.4% |
| 8M | 1.2e-4 | 15,884 | 95.8% | 75.0% | 13.2% |
| 9M | 0.6e-4 | 16,430 | 97.2% | 74.6% | 18.2% |
| **10M** | 0 | **17,405** | **97.4%** | **79.6%** | **18.8%** |

Every checkpoint beat the one before on mean score, and the 2048 rate more than doubled. The 6M and 9M checkpoints
each made a 4096 tile in one of the 500 games.

### 8. Moving to a GPU

Everything so far ran on a laptop CPU at ~520 steps/s. On a single GPU (`--device cuda`), the same setup runs at
**~7,800 steps/s, 15× faster**: 10M steps take about 21 minutes instead of about 5 hours. That made the longer
experiments below practical. The game itself still runs on the CPU; what the GPU speeds up is the network updates,
which had been ~70% of the training time.

### 9. A longer horizon: `gamma` 0.99 → 0.995

With `gamma=0.99`, rewards more than ~100 moves away barely count, but a good game now lasts over 1,000 moves.
Raising `gamma` to 0.995 (an effective horizon of ~200 moves) was the only change in this run. The schedule was the
same: 3e-4 for 5M steps, then falling linearly to 0 at 10M.

| Model | Mean score | Reaches 512 | Reaches 1024 | Reaches 2048 | Reaches 4096 |
|---|---|---|---|---|---|
| γ=0.99, 5M | 13,569 | 93.8% | 63.0% | 8.2% | 0% |
| γ=0.995, 5M | 13,996 | 90.4% | 66.8% | 16.4% | 0% |
| γ=0.99, 10M | 17,405 | 97.4% | 79.6% | 18.8% | 0% |
| **γ=0.995, 10M** | **18,696** | 96.4% | **82.2%** | **27.4%** | **0.4%** |

Looking further ahead doubled the 2048 rate at 5M steps and raised it by ~9 points at 10M. The cost is a few more
early losses (a slightly lower 512 rate): the agent takes more risks for bigger tiles. All later runs use 0.995.

### 10. Much longer training

Two ways to spend more steps were compared:

- **A longer run:** 30M steps from scratch, at a constant 3e-4 for 20M steps, then falling linearly to 0.
- **A fine-tune:** the 10M γ=0.995 model trained for another 10M steps, at 1e-4 falling to 0.

| Model | Learning rate at that point | Mean score | Reaches 1024 | Reaches 2048 | Reaches 4096 |
|---|---|---|---|---|---|
| Longer run, 10M | 3e-4 | 17,389 | 79.6% | 23.6% | 0.6% |
| Longer run, 20M | 3e-4 | 24,193 | 90.2% | 51.0% | 3.0% |
| Longer run, 25M | 1.5e-4 | 27,473 | 93.0% | 62.0% | 5.2% |
| **Longer run, 30M** | 0 | **31,676** | **96.2%** | **71.6%** | **9.8%** |
| Fine-tune, 20M | 0 | 22,924 | 91.8% | 41.0% | 2.0% |

Steps at the full learning rate were the main limit. At 20M steps, still at a constant 3e-4, the longer run already
beat every earlier model, and the final 10M steps of decay took the 2048 rate from 51% to 72%. The fine-tune is
cheaper (+10M steps instead of 30M from scratch) but levels off much lower.

### 11. A bigger network: 128 filters

The CNN went down to 64 filters only to make laptop training affordable (step 5). On the GPU, the 30M-step run was
repeated with 128 filters (4.2M parameters instead of ~2M), at ~7,000 steps/s (72 minutes).

| Model | Mean score | Reaches 1024 | Reaches 2048 | Reaches 4096 |
|---|---|---|---|---|
| 128 filters, 20M | 23,766 | 90.2% | 47.4% | 2.2% |
| 128 filters, 25M | 26,225 | 93.0% | 54.6% | 4.8% |
| **128 filters, 30M** | **32,580** | **96.2%** | **69.8%** | **10.0%** |

The two network sizes were level at 20M and 30M steps on the usual 500 games. On 2,000 fresh games (seeds
20,000–21,999), the 128-filter model comes out ahead:

| 30M steps, 2,000 games | Mean score | Median | Reaches 2048 | Reaches 4096 | Reaches 8192 |
|---|---|---|---|---|---|
| **128 filters** | **34,447** | **35,174** | **75.2%** | **12.6%** | 3 games |
| 64 filters | 32,187 | 33,008 | 72.7% | 9.8% | 1 game |

The 4096 gap is about three standard errors, well outside evaluation noise; the 2048 gap is borderline. Each setting was trained once, so part
of the difference may come from the training seed.

### Final configuration

| Setting | Value |
|---|---|
| Algorithm | MaskablePPO (sb3-contrib) |
| Observation | One-hot board, 18 × 4 × 4 |
| Network | `Conv2048` (1×2 / 2×1 filters, **128 filters**, 256 features), separate for policy and value, 256-unit heads |
| Reward | `"log"`, normalized with `VecNormalize` |
| Environments | 16 × `DummyVecEnv` |
| Steps | **30M** |
| `n_steps` / `batch_size` / `n_epochs` | 512 / 2048 / 4 |
| `gamma` / `gae_lambda` | **0.995** / 0.95 |
| Learning rate | 3e-4 for 20M steps, then falling linearly to 0 at 30M |
| `ent_coef` / `clip_range` | 0.01 / 0.2 |
| Training time | ~72 minutes on a single GPU |

The full script is [scripts/train_ppo.py](scripts/train_ppo.py); the [quick start](#quick-start) command reproduces
this model. Other options: `--lr`, `--run-name`, `--checkpoint-every`, and `--resume CHECKPOINT` to start from an
existing model. `--time-limit SECONDS` stops cleanly and saves `latest.zip`; running the same command again resumes
from it, with the learning-rate schedule intact.

### What's next

- **Train even longer.** The 30M runs were still improving when their learning rate reached 0. A 100M-step run with
  128 filters is the obvious next experiment.
- **An even longer horizon.** Games now last ~1,800 moves, so `gamma=0.999` (an effective horizon of ~1,000 moves)
  may help again.
- **Several seeds per setting**, to separate real differences (like 64 vs 128 filters) from training noise.
- **N-tuple networks with TD learning.** The strongest published 2048 learners use lookup tables over groups of
  cells instead of a neural network, and reach 2048 in roughly 90% of games.

---

## Reward design

`reward_mode` controls what the agent is paid for each move:

| Mode | Reward per move | Making a 2048 tile pays | Making a 4096 tile pays |
|---|---|---|---|
| `"score"` | value of each tile created by a merge (the in-game score) | 2048 | 4096 |
| `"log"` | log2 of each tile created by a merge | 11 | 12 |
| `"merges"` | number of merges | 1 | 1 |

All the agents above use `"log"`: it keeps rewards in a small, stable range. Under it, a single big merge is worth
barely more than a small one. Bigger tiles still pay off overall, because building one takes many smaller merges,
each of which earns reward, and because a longer game earns more reward in total. Building a 4096 from 2s earns
about 6,100 in log reward along the way, versus about 3,100 for a 2048.

## The environment in detail

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
| **Observation** | `(4, 4)` `uint8` tile exponents |
| **Actions** | `Discrete(4)`: `0` up, `1` right, `2` down, `3` left |
| **Terminated** | no legal move left, `win_tile` reached, or too many consecutive illegal moves |
| **Info** | `score`, `max_tile`, `moves`, `steps`, `valid_move`, `merges`, `won`, `action_mask` |

| kwarg | default | meaning |
|---|---|---|
| `reward_mode` | `"score"` | see [Reward design](#reward-design) |
| `invalid_move_reward` | `0.0` | reward for a move that doesn't change the board (no tile spawns) |
| `max_invalid_moves` | `16` | end the episode after this many illegal moves in a row (`None` = never) |
| `win_tile` | `None` | end the episode once this tile is made, e.g. `2048` |
| `size` | `4` | board side length (3×3 boards are handy for quick experiments) |
| `four_prob` | `0.1` | probability that a spawned tile is a 4 |
| `render_mode` | `None` | `"human"` (prints), `"ansi"` (returns a string), `"rgb_array"` (image frames) |

`env.reset(options={"board": exponent_array})` starts from a fixed position.

**Observation wrappers** re-encode the board for neural networks:

```python
from gymnasium.wrappers import FlattenObservation
from game2048 import OneHotBoard, NormalizedBoard

env = OneHotBoard(gym.make("Game2048-v0"))                       # (18, 4, 4) float32, for CNNs
env = FlattenObservation(OneHotBoard(gym.make("Game2048-v0")))   # (288,) float32, for MLPs
env = NormalizedBoard(gym.make("Game2048-v0"))                   # (4, 4) float32 in [0, 1]
```

**With MaskablePPO,** put `ActionMasker` outermost:
`ActionMasker(env, lambda e: e.unwrapped.action_masks())`. Gymnasium 1.x wrappers no longer pass unknown methods
like `action_masks()` through to the inner environment.

## Tools

| Command | What it does |
|---|---|
| `play-2048` | Play in the terminal with w/a/s/d |
| `eval-2048 --agent {random,greedy,corner,ppo}` | Benchmark over many games: score, tile distribution, speed. `--model` for PPO checkpoints, `--stochastic` to sample moves |
| `replay-2048` | Record one game as an interactive HTML replay (play/pause, scrub, the policy's probability for each move, score chart). `--best-of N` keeps the best of N games; `--gif PATH` also writes an animated GIF |

## Layout

```
src/game2048/
  game.py                # pure game logic (no Gym dependency)
  env.py                 # Game2048Env (Gymnasium)
  wrappers.py            # OneHotBoard, NormalizedBoard
  networks.py            # Conv2048 feature extractor
  render.py              # dependency-free RGB renderer
  agents.py              # random / greedy / corner baselines, PPOAgent
  evaluate.py            # eval-2048 (sequential and batched)
  replay.py              # replay-2048 (HTML + GIF)
  replay_template.html   # replay page template
  play.py                # play-2048
scripts/train_ppo.py     # training / resuming
tests/                   # game rules, Gymnasium API compliance, wrappers, agents, replays
docs/replay.gif          # the GIF above
```
