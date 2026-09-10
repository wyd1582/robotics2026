"""Tests for physical/API semantics that could invalidate a learning experiment."""
import argparse

import gymnasium as gym
import numpy as np
import torch

from course_utils import (
    audit_transitions,
    collect_with_planner,
    prediction_diagnostics,
    preserve_cpu_rng,
)
from frontier import COMMIT, build_commands
from loop01_imitation import Actor, collect, expert_action
from loop02_reinforcement import TransitionRecorder
from loop03_world_model import CEMPlanner, known_transition


def test_teacher_reaches_and_xml_contract():
    env = gym.make("Reacher-v5")
    assert env.observation_space.shape == (10,)
    assert np.allclose(env.unwrapped.model.actuator_gear[:, 0], 200)
    assert np.isclose(env.unwrapped.dt, .02)
    for seed in [4, 17, 29]:
        obs, _ = env.reset(seed=seed)
        for _ in range(50):
            a = expert_action(env)
            assert env.action_space.contains(a)
            obs, *_ = env.step(a)
        assert np.linalg.norm(obs[-2:]) < .025
    env.close()


def test_raw_episode_boundaries_and_label_semantics():
    data = collect([2, 3], noise=.02)
    assert len(data["obs"]) == 100
    assert set(data["episode"]) == {2, 3}
    assert np.allclose(data["sample_time"][[0, 50]], 0)
    assert np.all(data["truncated"][[49, 99]])
    assert not np.array_equal(data["expert_action"], data["executed_action"])
    assert np.allclose(data["next_obs"][:49], data["obs"][1:50])


def test_known_dynamics_matches_actual_environment():
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=8)
    rng = np.random.default_rng(19)
    for _ in range(40):
        a = rng.uniform(-2, 2, 1).astype(np.float32)
        pred = known_transition(torch.tensor(obs)[None], torch.tensor(a)[None])[0].numpy()
        obs, *_ = env.step(a)
        np.testing.assert_allclose(pred, obs, atol=2e-6)
    env.close()


def test_actor_checkpoint_preserves_action(tmp_path):
    a = Actor(np.zeros(10), np.ones(10), np.zeros(2), np.ones(2))
    x = np.linspace(-1, 1, 10)
    expected = a.action(x)
    path = tmp_path / "actor.pt"
    torch.save(a.state_dict(), path)
    b = Actor(np.zeros(10), np.ones(10), np.zeros(2), np.ones(2))
    b.load_state_dict(torch.load(path, weights_only=True))
    np.testing.assert_array_equal(expected, b.action(x))


def test_recorder_preserves_timeout_transition():
    env = TransitionRecorder(gym.make("Pendulum-v1", max_episode_steps=2))
    first, _ = env.reset(seed=1)
    env.step(np.zeros(1))
    final, _, terminated, truncated, _ = env.step(np.zeros(1))
    assert truncated and not terminated
    np.testing.assert_array_equal(env.rows[0][0], first)
    np.testing.assert_array_equal(env.rows[-1][2], final)
    env.reset(seed=2)
    assert env.episode == 1
    env.close()


def test_cem_action_within_bounds():
    planner = CEMPlanner(known_transition, horizon=3, candidates=8, iterations=1, seed=4)
    a = planner.action(np.array([1., 0., 0.]))
    assert a.shape == (1,) and np.isfinite(a).all() and np.abs(a).max() <= 2


def test_frontier_commands_disable_upload_and_pin_dataset(tmp_path):
    args = argparse.Namespace(track="diffusion", stage="train", device="cpu", steps=100,
                              batch_size=4, seed=7, checkpoint=None)
    command = build_commands(args, tmp_path / "repo", tmp_path / "output")[0]
    assert "--policy.push_to_hub=false" in command
    assert "--wandb.enable=false" in command
    assert any(s.startswith("--dataset.revision=") for s in command)
    assert "--env_eval_freq=0" in command
    assert "--dataset.eval_split=0.1" in command
    assert len(COMMIT) == 40


def test_planner_collection_uses_real_next_states_and_action_repeat():
    config = {"horizon": 3, "candidates": 8, "iterations": 1, "repeat": 3}
    data, scores = collect_with_planner(known_transition, "oracle", [30, 31], config, episode_steps=8)
    audit_transitions(data, require_closed=True)
    assert len(data["obs"]) == 16 and len(scores) == 2
    assert data["truncated"][[7, 15]].all() and not data["terminated"].any()
    for start in [0, 8]:
        np.testing.assert_array_equal(data["action"][start:start+3], np.repeat(data["action"][start:start+1], 3, axis=0))
    predicted = known_transition(torch.tensor(data["obs"]), torch.tensor(data["action"])).numpy()
    np.testing.assert_allclose(predicted, data["next_obs"], atol=2e-6)


def test_multistep_diagnostic_respects_episode_boundaries():
    from loop03_world_model import collect as collect_dynamics
    data = collect_dynamics([6, 21])
    errors = prediction_diagnostics(known_transition, data, horizons=(1, 5))
    assert len(errors) == 4
    assert errors.angle_rmse_rad.max() < 1e-5
    assert errors.velocity_rmse_rad_s.max() < 1e-4


def test_checkpoint_inspection_preserves_training_random_streams(tmp_path):
    from stable_baselines3 import SAC
    model = SAC("MlpPolicy", "Pendulum-v1", seed=13, buffer_size=10, policy_kwargs={"net_arch": [8, 8]})
    model.save(tmp_path / "policy")
    state = torch.get_rng_state()
    np_state = np.random.get_state()
    with preserve_cpu_rng():
        loaded = SAC.load(tmp_path / "policy", device="cpu")
    assert torch.equal(torch.get_rng_state(), state)
    current = np.random.get_state()
    assert current[0] == np_state[0] and np.array_equal(current[1], np_state[1]) and current[2:] == np_state[2:]
    np.testing.assert_array_equal(model.predict(np.array([1., 0., 0.]), deterministic=True)[0],
                                  loaded.predict(np.array([1., 0., 0.]), deterministic=True)[0])
    model.get_env().close()
