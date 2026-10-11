import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from gymnasium.wrappers import FlattenObservation

import game2048  # noqa: F401
from game2048 import Game2048Env, NormalizedBoard, OneHotBoard
from game2048.agents import BASELINES, PPOAgent
from game2048.evaluate import evaluate, evaluate_batched
from game2048.game import LEFT, RIGHT

STUCK = np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]], dtype=np.uint8)


def test_passes_gymnasium_env_checker():
    check_env(gym.make("Game2048-v0").unwrapped, skip_render_check=False)


@pytest.mark.filterwarnings("ignore:.*different from the unwrapped")
@pytest.mark.filterwarnings("ignore:.*not having a spec")
def test_passes_env_checker_with_wrappers():
    check_env(OneHotBoard(Game2048Env()))
    check_env(NormalizedBoard(Game2048Env()))


def test_registered_make():
    env = gym.make("Game2048-v0")
    obs, info = env.reset(seed=0)
    assert obs.shape == (4, 4) and obs.dtype == np.uint8
    assert np.count_nonzero(obs) == 2
    assert info["action_mask"].shape == (4,)


def test_seeding_is_deterministic():
    def rollout(seed):
        env = Game2048Env()
        obs, info = env.reset(seed=seed)
        boards = [obs]
        for _ in range(50):
            obs, *_ , info = env.step(int(np.flatnonzero(info["action_mask"])[0]))
            boards.append(obs)
        return np.stack(boards)

    np.testing.assert_array_equal(rollout(7), rollout(7))
    assert not np.array_equal(rollout(7), rollout(8))


def test_score_reward_and_merge():
    env = Game2048Env()
    board = np.zeros((4, 4), dtype=np.uint8)
    board[0, :2] = 1  # [2, 2, 0, 0]
    env.reset(seed=0, options={"board": board})
    _, reward, terminated, truncated, info = env.step(LEFT)
    assert reward == 4.0 and info["score"] == 4 and info["merges"] == 1
    assert not terminated and not truncated


def test_log_reward():
    env = Game2048Env(reward_mode="log")
    board = np.zeros((4, 4), dtype=np.uint8)
    board[0, :2] = 3  # [8, 8, 0, 0] -> 16 = 2^4
    env.reset(seed=0, options={"board": board})
    _, reward, *_ = env.step(LEFT)
    assert reward == 4.0


def test_invalid_move_penalty_and_termination():
    env = Game2048Env(invalid_move_reward=-2.0, max_invalid_moves=3)
    board = np.zeros((4, 4), dtype=np.uint8)
    board[0, 0] = 1
    env.reset(seed=0, options={"board": board})
    for i in range(3):
        _, reward, terminated, _, info = env.step(LEFT)
        assert reward == -2.0 and not info["valid_move"]
        assert terminated == (i == 2)


def test_valid_move_resets_invalid_streak():
    env = Game2048Env(max_invalid_moves=2)
    board = np.zeros((4, 4), dtype=np.uint8)
    board[0, 0] = 1
    env.reset(seed=0, options={"board": board})
    env.step(LEFT)
    _, _, terminated, _, _ = env.step(RIGHT)
    assert not terminated
    assert env._invalid_streak == 0


def test_game_over_terminates():
    env = Game2048Env()
    almost = STUCK.copy()
    almost[0, 0] = 0
    almost[0, 1] = 1  # [., 2, 2, 4] ... one move left
    env.reset(seed=0, options={"board": almost})
    terminated = False
    for _ in range(1000):
        mask = env.action_masks()
        if not mask.any():
            break
        _, _, terminated, _, _ = env.step(int(np.flatnonzero(mask)[0]))
        if terminated:
            break
    assert terminated


def test_win_tile():
    env = Game2048Env(win_tile=2048)
    board = np.zeros((4, 4), dtype=np.uint8)
    board[0, :2] = 10  # two 1024s
    env.reset(seed=0, options={"board": board})
    _, _, terminated, _, info = env.step(LEFT)
    assert terminated and info["won"] and info["max_tile"] == 2048


def test_render_modes():
    text = Game2048Env(render_mode="ansi")
    text.reset(seed=0)
    assert "score: 0" in text.render()
    rgb = Game2048Env(render_mode="rgb_array")
    rgb.reset(seed=0)
    frame = rgb.render()
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8


def test_one_hot_encoding():
    env = OneHotBoard(Game2048Env())
    obs, _ = env.reset(seed=0)
    assert obs.shape == (18, 4, 4)
    np.testing.assert_array_equal(obs.sum(axis=0), np.ones((4, 4)))
    flat = FlattenObservation(OneHotBoard(Game2048Env()))
    assert flat.reset(seed=0)[0].shape == (18 * 16,)


def test_one_hot_batch_matches_single():
    env = OneHotBoard(Game2048Env())
    boards = np.stack([env.unwrapped.reset(seed=i)[0] for i in range(5)])
    batch = env.observation(boards)
    assert batch.shape == (5, 18, 4, 4)
    for board, encoded in zip(boards, batch):
        np.testing.assert_array_equal(env.observation(board), encoded)


def test_vector_env():
    envs = gym.make_vec("Game2048-v0", num_envs=4, vectorization_mode="sync")
    obs, info = envs.reset(seed=0)
    assert obs.shape == (4, 4, 4)
    obs, rewards, *_ = envs.step(np.array([0, 1, 2, 3]))
    assert rewards.shape == (4,)
    envs.close()


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_baseline_agents_finish_episodes(name):
    result = evaluate(Game2048Env(), BASELINES[name](seed=0), episodes=3, seed=0)
    assert len(result.scores) == 3 and all(s > 0 for s in result.scores)


def test_greedy_beats_random():
    env = Game2048Env()
    random_scores = evaluate(env, BASELINES["random"](seed=0), episodes=30, seed=0).scores
    greedy_scores = evaluate(env, BASELINES["greedy"](seed=0), episodes=30, seed=0).scores
    assert np.mean(greedy_scores) > np.mean(random_scores)


@pytest.mark.parametrize("arch", ["mlp", "cnn"])
def test_ppo_agent_round_trip(tmp_path, arch):
    sb3_contrib = pytest.importorskip("sb3_contrib")
    from sb3_contrib.common.wrappers import ActionMasker

    env = OneHotBoard(gym.make("Game2048-v0", reward_mode="log"))
    policy_kwargs = None
    if arch == "mlp":
        env = FlattenObservation(env)
    else:
        from game2048.networks import Conv2048

        policy_kwargs = dict(
            features_extractor_class=Conv2048,
            features_extractor_kwargs=dict(filters=8, features_dim=16),
        )
    env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
    model = sb3_contrib.MaskablePPO(
        "MlpPolicy", env, n_steps=64, batch_size=64, n_epochs=1, policy_kwargs=policy_kwargs
    )
    model.learn(total_timesteps=64)
    path = tmp_path / "ppo"
    model.save(path)

    agent = PPOAgent(str(path), seed=0)
    assert agent.flatten == (arch == "mlp")
    result = evaluate(Game2048Env(), agent, episodes=6, seed=0)
    assert len(result.scores) == 6

    # Batched play (fewer slots than episodes, so slots get reused) matches sequential play.
    batched = evaluate_batched(agent, episodes=6, seed=0, batch_size=4)
    assert batched.scores == result.scores
    assert batched.max_tiles == result.max_tiles
    assert batched.steps == result.steps

    with pytest.raises(ValueError, match="observations of shape"):
        PPOAgent(str(path), size=3)
