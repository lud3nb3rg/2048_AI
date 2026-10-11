import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import game2048  # noqa: F401  (registers "Game2048-v0")
from game2048.wrappers import OneHotBoard


class EpisodeStatsCallback(BaseCallback):
    """Log game scores, tile values, and milestone rates at episode end."""

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue

            max_tile = info["max_tile"]
            self.logger.record_mean("episode/score", info["score"])
            self.logger.record_mean("episode/max_tile", max_tile)
            for threshold in (512, 1024, 2048):
                self.logger.record_mean(
                    f"episode/reached_{threshold}",
                    float(max_tile >= threshold),
                )

        return True


def make_env():
    env = gym.make("Game2048-v0", reward_mode="log", max_episode_steps=10_000)
    env = FlattenObservation(OneHotBoard(env))
    env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
    return env


if __name__ == "__main__":
    n_envs = 16
    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=2048,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log="runs/",
        verbose=1,
    )
    callbacks = CallbackList(
        [
            EpisodeStatsCallback(),
            CheckpointCallback(
                save_freq=max(100_000 // n_envs, 1),
                save_path="checkpoints/ppo",
            ),
        ]
    )
    model.learn(
        total_timesteps=1_000_000,
        tb_log_name="PPO_2048",
        callback=callbacks,
    )

