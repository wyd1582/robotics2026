"""Check pedagogically important reward timing, units and time-limit behavior."""
import gymnasium as gym
import numpy as np

from sac_challenges import WeightedPendulum, metrics, rollout, score, soft_target, zero_action


def test_reward_wrapper_changes_cost_not_transition_or_timeout():
    original = gym.make("Pendulum-v1")
    modified = WeightedPendulum((1., .1, 1.))
    try:
        np.testing.assert_array_equal(original.reset(seed=13)[0], modified.reset(seed=13)[0])
        for step in range(200):
            theta, omega = original.unwrapped.state.copy()
            action = np.array([.7], dtype=np.float32)
            nxt, reward, term, trunc, _ = original.step(action)
            nxt2, reward2, term2, trunc2, info = modified.step(action)
            np.testing.assert_allclose(reward, score(theta, omega, action[0]), atol=1e-7)
            np.testing.assert_allclose(reward2, score(theta, omega, action[0], (1., .1, 1.)), atol=1e-7)
            np.testing.assert_array_equal(nxt, nxt2)
            assert info["original_reward"] == reward
            assert (term, trunc) == (term2, trunc2) == (False, step == 199)
    finally:
        original.close()
        modified.close()


def test_zero_torque_does_not_depend_on_observation_delay():
    plain = rollout(zero_action, seed=42)
    delayed = rollout(zero_action, seed=42, delay=2)
    np.testing.assert_array_equal(plain.to_numpy(), delayed.to_numpy())
    np.testing.assert_allclose(plain.reward, score(plain.theta, plain.omega, plain.torque))
    assert metrics(plain)["steps"] == 200
    assert plain.truncated.sum() == 1 and not plain.terminated.any()


def test_action_clipping_and_pre_action_reward():
    assert score(np.pi, 0, 9) == score(-np.pi, 0, 2)
    trace = rollout(lambda obs: [9.], initial=(1., 0.), max_steps=1)
    assert trace.torque.iloc[0] == 2.
    np.testing.assert_allclose(trace.reward.iloc[0], -1.004)
    assert not np.isclose(trace.reward.iloc[0], score(trace.next_theta.iloc[0], trace.next_omega.iloc[0], 2))


def test_soft_target_uses_lower_q_and_only_true_termination_mask():
    # Synthetic critic values: these are not measurements from a trained policy.
    np.testing.assert_allclose(soft_target(-1., -3., -2., -.5, False), -1+.99*(-3+.05))
    assert soft_target(-1., -3., -2., -.5, True) == -1.
