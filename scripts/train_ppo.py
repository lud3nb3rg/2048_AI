"""Train a MaskablePPO agent on 2048, resuming automatically if interrupted.

    uv run python -u scripts/train_ppo.py --total-timesteps 10_000_000 --lr-decay-from 0.5

A run lives in ``checkpoints/<run-name>``. If it holds a ``latest.zip`` (saved when
``--time-limit`` stops training early), running the same command again continues
from there, with the same learning-rate schedule, until ``--total-timesteps`` is
reached. ``--resume CHECKPOINT`` starts a run from any other checkpoint instead.
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import game2048  # noqa: F401  (registers "Game2048-v0")
from game2048.wrappers import OneHotBoard
from game2048.networks import Conv2048

N_ENVS = 16
DEFAULT_GAMMA = 0.99
DEFAULT_LR = 3e-4
DEFAULT_FILTERS = 64


class EpisodeStatsCallback(BaseCallback):
    """Log game scores, tile values, and milestone rates at episode end."""

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue

            max_tile = info["max_tile"]
            self.logger.record_mean("episode/score", info["score"])
            self.logger.record_mean("episode/max_tile", max_tile)
            for threshold in (512, 1024, 2048, 4096):
                self.logger.record_mean(
                    f"episode/reached_{threshold}",
                    float(max_tile >= threshold),
                )

        return True


class TimeLimitCallback(BaseCallback):
    """Stop training cleanly once ``seconds`` have passed (e.g. before a job's time limit)."""

    def __init__(self, seconds: float) -> None:
        super().__init__()
        self.deadline = time.monotonic() + seconds

    def _on_step(self) -> bool:
        return time.monotonic() < self.deadline


def lr_schedule(initial: float, decay_from: float):
    """Constant ``initial`` until ``decay_from`` of the run is done, then linear to 0.

    SB3 calls this with ``progress_remaining``, which goes 1 -> 0 over the run.
    Training always targets the same total, so progress (and the rate) carries
    over correctly when a run is resumed.
    """
    decay_span = 1.0 - decay_from

    def schedule(progress_remaining: float) -> float:
        return initial * min(1.0, progress_remaining / decay_span)

    return schedule


def make_env():
    env = gym.make("Game2048-v0", reward_mode="log", max_episode_steps=10_000)
    env = OneHotBoard(env)
    env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
    return env


def new_model(env, gamma: float, filters: int, device: str):
    return MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=2048,
        n_epochs=4,
        gamma=gamma,
        gae_lambda=0.95,
        learning_rate=DEFAULT_LR,  # replaced by the schedule in main()
        ent_coef=0.01,
        policy_kwargs=dict(
            features_extractor_class=Conv2048,
            features_extractor_kwargs=dict(filters=filters, features_dim=256),
            net_arch=[256],
            # Separate CNNs for policy and value, so value-loss gradients
            # can't wreck the policy's features.
            share_features_extractor=False,
        ),
        tensorboard_log="runs/",
        device=device,
        verbose=1,
    )


def vecnormalize_path(checkpoint: Path) -> Path:
    """Where the reward-normalization stats for ``checkpoint`` were saved."""
    if checkpoint.stem.startswith("rl_model_"):  # written by CheckpointCallback
        return checkpoint.with_name(checkpoint.stem.replace("rl_model", "rl_model_vecnormalize") + ".pkl")
    return checkpoint.with_name(checkpoint.stem + "_vecnormalize.pkl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", default="ppo_cnn64_norm")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument(
        "--lr-decay-from", type=float, default=0.0,
        help="fraction of the run after which the learning rate decays linearly to 0",
    )
    parser.add_argument("--gamma", type=float, help=f"discount factor (default {DEFAULT_GAMMA})")
    parser.add_argument("--resume", type=Path, help="checkpoint to start this run from")
    parser.add_argument("--time-limit", type=float, help="seconds to train before saving and stopping")
    parser.add_argument("--device", default="cpu", help='"cpu", "cuda" or "auto"')
    # 4 threads beat the default on a 6-performance-core laptop CPU; see the benchmark.
    parser.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="learning rate before the decay")
    parser.add_argument(
        "--filters", type=int, help=f"Conv2048 filters for a new model (default {DEFAULT_FILTERS})"
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=100_000, help="steps between saved checkpoints"
    )
    args = parser.parse_args()
    if not 0.0 <= args.lr_decay_from < 1.0:
        parser.error("--lr-decay-from must be in [0, 1)")

    torch.set_num_threads(args.threads)
    out_dir = Path("checkpoints") / args.run_name
    latest = out_dir / "latest.zip"
    final = out_dir / f"final_{args.total_timesteps / 1e6:g}M.zip"
    if final.exists():
        print(f"{final} already exists; nothing to do")
        print("STATUS: done")
        return
    start_from = latest if latest.exists() else args.resume

    # Each game step costs ~86us, less than the IPC of SubprocVecEnv, so step in-process.
    env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=DummyVecEnv)
    if start_from:
        # Reuse the reward-normalization statistics saved with the checkpoint.
        env = VecNormalize.load(str(vecnormalize_path(start_from)), env)
        env.training = True
        model = MaskablePPO.load(start_from, env=env, device=args.device, tensorboard_log="runs/")
        if args.gamma is not None and args.gamma != model.gamma:
            parser.error(f"--gamma {args.gamma} differs from the checkpoint's {model.gamma}")
        filters = model.policy_kwargs["features_extractor_kwargs"]["filters"]
        if args.filters is not None and args.filters != filters:
            parser.error(f"--filters {args.filters} differs from the checkpoint's {filters}")
        print(f"resuming from {start_from} at {model.num_timesteps:,} steps")
    else:
        gamma = args.gamma if args.gamma is not None else DEFAULT_GAMMA
        # Scale rewards so value targets are ~unit size; the unscaled returns (in the
        # hundreds) kept the value head from learning at all in the first CNN run.
        # Observations are one-hot already, so they stay as they are.
        env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=gamma)
        model = new_model(env, gamma, args.filters or DEFAULT_FILTERS, args.device)

    model.learning_rate = lr_schedule(args.lr, args.lr_decay_from)
    model._setup_lr_schedule()
    remaining = args.total_timesteps - model.num_timesteps
    if remaining <= 0:
        parser.error(f"{start_from} is already at {model.num_timesteps:,} steps; raise --total-timesteps")

    callbacks = [
        EpisodeStatsCallback(),
        CheckpointCallback(
            save_freq=max(args.checkpoint_every // N_ENVS, 1),
            save_path=str(out_dir),
            # Reward-normalization stats, needed to resume training.
            save_vecnormalize=True,
        ),
    ]
    if args.time_limit:
        callbacks.append(TimeLimitCallback(args.time_limit))
    model.learn(
        total_timesteps=remaining,
        tb_log_name=args.run_name,
        callback=CallbackList(callbacks),
        # Continue the step count (and the TensorBoard curve) when resuming.
        reset_num_timesteps=start_from is None,
    )

    done = model.num_timesteps >= args.total_timesteps
    path = final if done else latest
    model.save(path)
    env.save(str(vecnormalize_path(path)))
    if done and latest.exists():
        latest.unlink()
        vecnormalize_path(latest).unlink(missing_ok=True)
    print(f"saved {path} at {model.num_timesteps:,} steps")
    print("STATUS:", "done" if done else "incomplete")


if __name__ == "__main__":
    main()
