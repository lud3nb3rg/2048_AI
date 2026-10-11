"""Train (or resume training) a MaskablePPO agent on 2048.

    uv run python -u scripts/train_ppo.py                     # fresh 5M-step run
    uv run python -u scripts/train_ppo.py --resume checkpoints/ppo_cnn64_norm/final.zip
"""

import argparse
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

RUN_NAME = "ppo_cnn64_norm"
TOTAL_TIMESTEPS = 5_000_000
N_ENVS = 16
GAMMA = 0.99
LEARNING_RATE = 3e-4
# 4 threads beat the default 16 on this CPU (6 performance cores); see the benchmark.
TORCH_THREADS = 4


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


def linear_schedule(initial: float, start_progress: float = 1.0):
    """Learning rate falling linearly from ``initial`` to 0 by the end of training.

    SB3 calls this with ``progress_remaining``, which goes 1 -> 0 over the whole
    run. When resuming, training starts part-way through (``start_progress < 1``),
    so the rate is rescaled to start at ``initial`` there rather than lower.
    """

    def schedule(progress_remaining: float) -> float:
        return initial * min(1.0, progress_remaining / start_progress)

    return schedule


def make_env():
    env = gym.make("Game2048-v0", reward_mode="log", max_episode_steps=10_000)
    env = OneHotBoard(env)
    env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
    return env


def new_model(env):
    return MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=2048,
        n_epochs=4,
        gamma=GAMMA,
        gae_lambda=0.95,
        learning_rate=linear_schedule(LEARNING_RATE),
        ent_coef=0.01,
        policy_kwargs=dict(
            features_extractor_class=Conv2048,
            features_extractor_kwargs=dict(filters=64, features_dim=256),
            net_arch=[256],
            # Separate CNNs for policy and value, so value-loss gradients
            # can't wreck the policy's features.
            share_features_extractor=False,
        ),
        tensorboard_log="runs/",
        device="cpu",
        verbose=1,
    )


def resume_model(path: Path, env, timesteps: int):
    model = MaskablePPO.load(path, env=env, device="cpu", tensorboard_log="runs/")
    # Fraction of the combined (old + new) run that is still to go when we restart.
    start = timesteps / (model.num_timesteps + timesteps)
    model.learning_rate = linear_schedule(LEARNING_RATE, start_progress=start)
    model._setup_lr_schedule()
    print(f"resuming from {path} at {model.num_timesteps:,} steps for {timesteps:,} more")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resume", type=Path, help="checkpoint to continue training from")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()

    torch.set_num_threads(TORCH_THREADS)
    # Each game step costs ~86us, less than the IPC of SubprocVecEnv, so step in-process.
    env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=DummyVecEnv)
    out_dir = Path("checkpoints") / RUN_NAME

    if args.resume:
        # Reuse the reward-normalization statistics saved with the checkpoint.
        stats = args.resume.with_name(args.resume.stem + "_vecnormalize.pkl")
        if not stats.exists():
            stats = args.resume.with_name(
                args.resume.stem.replace("rl_model", "rl_model_vecnormalize") + ".pkl"
            )
        env = VecNormalize.load(str(stats), env)
        env.training = True
        model = resume_model(args.resume, env, args.timesteps)
    else:
        # Scale rewards so value targets are ~unit size; the unscaled returns (in the
        # hundreds) kept the value head from learning at all in the first CNN run.
        # Observations are one-hot already, so they stay as they are.
        env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=GAMMA)
        model = new_model(env)

    callbacks = CallbackList(
        [
            EpisodeStatsCallback(),
            CheckpointCallback(
                save_freq=max(100_000 // N_ENVS, 1),
                save_path=str(out_dir),
                # Reward-normalization stats, needed to resume training.
                save_vecnormalize=True,
            ),
        ]
    )
    model.learn(
        total_timesteps=args.timesteps,
        tb_log_name=RUN_NAME,
        callback=callbacks,
        # Continue the step count (and the TensorBoard curve) when resuming.
        reset_num_timesteps=not args.resume,
    )
    name = "final" if not args.resume else f"final_{model.num_timesteps // 1_000_000}M"
    model.save(out_dir / name)
    env.save(str(out_dir / f"{name}_vecnormalize.pkl"))
    print(f"saved {out_dir / name}.zip")
