"""Pendulum: collect experience -> SAC/PPO updates -> reload -> paired evaluation.

Uses the actual Stable-Baselines3 implementation, with small educational budgets.
An independent learning seed and repeat runs are required for algorithm comparisons.
"""
from __future__ import annotations

import argparse
from collections import deque

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from common import (
    pendulum_frame,
    save_gif,
    setup_torch,
    start_run,
    summarize,
    write_paired_delta,
)


class TransitionRecorder(gym.Wrapper):
    """Record pre-reset terminal next_obs and explicit terminated/truncated flags."""
    def __init__(self, env):
        super().__init__(env)
        self.rows = []
        self.episode = -1
        self.last = None
        self.step_id = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last = obs.copy()
        self.episode += 1
        self.step_id = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.rows.append((self.last.copy(), np.asarray(action).copy(), obs.copy(), reward,
                          terminated, truncated, self.episode, self.step_id))
        self.step_id += 1
        self.last = obs.copy()
        return obs, reward, terminated, truncated, info

    def save(self, path):
        keys = ["obs", "action", "next_obs", "reward", "terminated", "truncated", "episode", "step"]
        np.savez_compressed(path, **{k: np.asarray([r[i] for r in self.rows]) for i, k in enumerate(keys)})


class Progress(BaseCallback):
    def _on_step(self):
        if self.num_timesteps % 2000 == 0:
            print(f"Collected {self.num_timesteps} environment steps", flush=True)
        return True


def evaluate(model, name, seeds, delay, gravity, out):
    env = gym.make("Pendulum-v1", g=gravity)
    rows, frames = [], []
    for i, seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(seed))
        history = deque([obs.copy()]*(delay+1), maxlen=delay+1)
        rng = np.random.default_rng(int(seed))
        total, upright = 0., []
        for step in range(200):
            action = rng.uniform(-2, 2, 1).astype(np.float32) if model is None else model.predict(history[0], deterministic=True)[0]
            obs, reward, done, trunc, _ = env.step(action)
            history.append(obs.copy())
            total += reward
            upright.append(int(abs(np.arctan2(obs[1], obs[0])) < .2 and abs(obs[2]) < 1.))
            if i == 0 and delay == 0 and gravity == 10 and step % 2 == 0:
                frames.append(pendulum_frame(obs, f"{name} | step {step+1} | return {total:.0f}"))
            if done or trunc:
                break
        rows.append({"policy": name, "condition": f"delay_{delay}_g{gravity:g}", "seed": int(seed),
                     "return": total, "upright_fraction": float(np.mean(upright))})
    env.close()
    save_gif(frames, out / f"{name}.gif", duration=100)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.steps, args.eval_episodes = 1000, 3
    if min(args.steps, args.eval_episodes) < 1:
        p.error("Counts must be positive")
    setup_torch(args.seed)
    out = start_run(f"02-{args.algo}", args)
    env = TransitionRecorder(gym.make("Pendulum-v1"))
    alg = SAC if args.algo == "sac" else PPO
    kwargs = {"policy_kwargs": {"net_arch": [64, 64]}, "seed": args.seed, "device": "cpu", "verbose": 0}
    if args.algo == "sac":
        kwargs.update(buffer_size=max(5000, args.steps), learning_starts=500, batch_size=128, learning_rate=1e-3)
    else:
        kwargs.update(n_steps=1024, batch_size=64, learning_rate=3e-4)
    model = alg("MlpPolicy", env, **kwargs)
    model.set_logger(configure(str(out / "training"), ["csv"]))
    print(f"Training {args.algo}; CPU is deliberate for a small MLP.", flush=True)
    model.learn(total_timesteps=args.steps, callback=Progress(), log_interval=1)
    model.save(out / "policy")
    if args.algo == "sac":
        model.save_replay_buffer(out / "replay_buffer.pkl")
    env.save(out / "raw_experience.npz")
    probe = np.array([1., 0., 0.], dtype=np.float32)
    expected = model.predict(probe, deterministic=True)[0]
    loaded = alg.load(out / "policy", device="cpu")
    if not np.allclose(expected, loaded.predict(probe, deterministic=True)[0]):
        raise RuntimeError("Checkpoint reload mismatch")
    env.close()
    rows = []
    seeds = range(10000, 10000+args.eval_episodes)
    for delay, gravity in [(0, 10.), (2, 10.), (0, 12.)]:
        rows += evaluate(None, "random", seeds, delay, gravity, out)
        rows += evaluate(loaded, args.algo, seeds, delay, gravity, out)
    frame = summarize(rows, out, "return", "Pendulum: higher return is better")
    write_paired_delta(frame, "random", "return", out)
    print("DONE. raw_experience.npz -> training/progress.csv -> policy.zip -> episodes.csv", flush=True)


if __name__ == "__main__":
    main()
