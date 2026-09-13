"""Inspectable Pendulum reward and paired-rollout helpers for lesson 02A.

Rewards use the pre-action state. Physical metrics use the post-action state.
No training, network access, or checkpoint loading occurs on import.
"""
from collections import deque

import gymnasium as gym
import numpy as np
import pandas as pd


def reward_terms(theta, omega, torque, weights=(1., .1, .001)):
    """Return positive costs, with Pendulum's wrapping and actuator clipping."""
    angle = (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi
    control = np.clip(np.asarray(torque), -2., 2.)
    return {
        "angle_cost": weights[0] * angle**2,
        "speed_cost": weights[1] * np.asarray(omega)**2,
        "control_cost": weights[2] * control**2,
    }


def score(theta, omega, torque, weights=(1., .1, .001)):
    return -sum(reward_terms(theta, omega, torque, weights).values())


class WeightedPendulum(gym.Wrapper):
    """Change training reward only; preserve physics, bounds and timeout semantics."""
    def __init__(self, weights=(1., .1, .001)):
        super().__init__(gym.make("Pendulum-v1"))
        self.weights = np.asarray(weights, dtype=float)
        if self.weights.shape != (3,) or not np.isfinite(self.weights).all() or (self.weights < 0).any():
            self.env.close()
            raise ValueError("weights must contain three finite nonnegative values")

    def step(self, action):
        theta, omega = self.unwrapped.state.copy()
        control = float(np.asarray(action).reshape(-1)[0])
        obs, original, terminated, truncated, info = self.env.step(action)
        reward = float(score(theta, omega, control, self.weights))
        info = dict(info, original_reward=float(original))
        return obs, reward, terminated, truncated, info


def zero_action(obs):
    return np.array([0.], dtype=np.float32)


def pd_action(obs, kp=6., kd=2.):
    """Local angle/velocity feedback; not claimed to solve global swing-up."""
    theta = np.arctan2(obs[1], obs[0])
    return np.array([np.clip(-kp * theta - kd * obs[2], -2., 2.)], dtype=np.float32)


def rollout(policy, seed=10000, initial=None, delay=0, gravity=10., max_steps=200):
    """Use default environment reward even when policy was trained with another one.

    initial=(theta, omega) is an explicit simulator-only intervention after reset.
    delay counts observations in units of 50 ms, not actuator delay.
    """
    if not isinstance(delay, int) or delay < 0 or not 1 <= max_steps <= 200:
        raise ValueError("delay must be a nonnegative integer; max_steps must be 1..200")
    env = gym.make("Pendulum-v1", g=gravity)
    records = []
    try:
        obs, _ = env.reset(seed=int(seed))
        if initial is not None:
            state = np.asarray(initial, dtype=float)
            if state.shape != (2,) or not np.isfinite(state).all() or abs(state[1]) > 8:
                raise ValueError("initial must be (theta, omega) with abs(omega) <= 8")
            env.unwrapped.state = state.copy()
            obs = env.unwrapped._get_obs()
        history = deque([obs.copy()] * (delay + 1), maxlen=delay + 1)
        for step in range(max_steps):
            theta, omega = env.unwrapped.state.copy()
            action = np.asarray(policy(history[0].copy()), dtype=np.float32).reshape(-1)
            if action.shape != (1,) or not np.isfinite(action).all():
                raise ValueError("policy must return one finite torque")
            action = np.clip(action, -2., 2.)
            nxt, reward, terminated, truncated, _ = env.step(action)
            next_theta = float((env.unwrapped.state[0] + np.pi) % (2*np.pi) - np.pi)
            next_omega = float(env.unwrapped.state[1])
            records.append(dict(step=step, time=step*.05, next_time=(step+1)*.05,
                theta=float((theta+np.pi) % (2*np.pi)-np.pi), omega=float(omega),
                torque=float(action[0]), reward=float(reward), next_theta=next_theta,
                next_omega=next_omega, upright=abs(next_theta)<.2 and abs(next_omega)<1.,
                terminated=bool(terminated), truncated=bool(truncated)))
            history.append(nxt.copy())
            if terminated or truncated:
                break
    finally:
        env.close()
    return pd.DataFrame(records)


def metrics(trace):
    return dict(original_return=float(trace.reward.sum()),
        upright_fraction=float(trace.upright.mean()),
        mean_torque_squared=float(np.mean(trace.torque**2)),
        peak_speed=float(np.abs(trace.next_omega).max()), steps=len(trace))


def soft_target(reward, next_q1, next_q2, next_log_prob, terminated,
                gamma=.99, alpha=.1):
    """One SAC bootstrap target; truncation alone does not zero the continuation."""
    continuation = np.minimum(next_q1, next_q2) - alpha * np.asarray(next_log_prob)
    return np.asarray(reward) + gamma * (1-np.asarray(terminated, dtype=float)) * continuation
