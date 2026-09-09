"""Learn a low-dimensional dynamics model, then plan with CEM/MPC in Pendulum.

This is a transparent model-based-control exercise, not PETS or Dreamer reproduction.
The reward is known. Only the transition model is learned. No image/language latent model.
"""
from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from torch import nn

from common import (
    pendulum_frame,
    plt,
    save_gif,
    setup_torch,
    start_run,
    summarize,
    write_paired_delta,
)


def collect(seeds):
    env = gym.make("Pendulum-v1")
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        rng = np.random.default_rng(int(seed))
        for step in range(200):
            action = rng.uniform(-2, 2, 1).astype(np.float32)
            nxt, reward, done, trunc, _ = env.step(action)
            rows.append((obs.copy(), action, nxt.copy(), int(seed), step, reward, done, trunc))
            obs = nxt
            if done or trunc:
                break
    env.close()
    keys = ["obs", "action", "next_obs", "episode", "step", "reward", "terminated", "truncated"]
    return {k: np.asarray([r[i] for r in rows]) for i, k in enumerate(keys)}


def known_transition(obs, action, gravity=10.):
    """Gymnasium Pendulum-v1 equations, implemented independently for an oracle reference."""
    theta = torch.atan2(obs[..., 1], obs[..., 0])
    velocity = obs[..., 2]
    torque = action[..., 0].clamp(-2, 2)
    new_v = (velocity + (.5*3*gravity*theta.sin()+3*torque)*.05).clamp(-8, 8)
    new_theta = theta + new_v*.05
    return torch.stack([new_theta.cos(), new_theta.sin(), new_v], dim=-1)


def running_cost(obs, action):
    theta = torch.atan2(obs[..., 1], obs[..., 0])
    return theta.square() + .1*obs[..., 2].square() + .001*action[..., 0].square()


class Dynamics(nn.Module):
    def __init__(self, xm, xs, ym, ys):
        super().__init__()
        for name, value in [("xm", xm), ("xs", xs), ("ym", ym), ("ys", ys)]:
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(4, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 3))

    def normalized_delta(self, obs, action):
        return self.net((torch.cat([obs, action], dim=-1)-self.xm)/self.xs)

    def forward(self, obs, action):
        nxt = obs + self.normalized_delta(obs, action)*self.ys+self.ym
        # Physical observation constraints: unit circle and bounded angular velocity.
        circle = nxt[..., :2] / torch.linalg.vector_norm(nxt[..., :2], dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.cat([circle, nxt[..., 2:3].clamp(-8, 8)], dim=-1)


def fit_model(train, val, updates, seed):
    torch.manual_seed(seed)
    inputs = np.concatenate([train["obs"], train["action"]], axis=1)
    deltas = train["next_obs"] - train["obs"]
    model = Dynamics(inputs.mean(0), np.maximum(inputs.std(0), .01),
                     deltas.mean(0), np.maximum(deltas.std(0), .001))
    x, a, y = [torch.tensor(train[k], dtype=torch.float32) for k in ["obs", "action", "next_obs"]]
    vx, va, vy = [torch.tensor(val[k], dtype=torch.float32) for k in ["obs", "action", "next_obs"]]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    history = []
    for step in range(updates):
        idx = rng.integers(0, len(x), 256)
        pred = model.normalized_delta(x[idx], a[idx])
        loss = ((pred-(y[idx]-x[idx]-model.ym)/model.ys)**2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, updates//10) == 0 or step == updates-1:
            with torch.inference_mode():
                val_mse = ((model(vx, va)-vy)**2).mean().item()
            history.append({"step": step+1, "train_normalized_mse": loss.item(), "val_next_state_mse": val_mse})
    return model.eval(), history


class CEMPlanner:
    """Each plan contains H torque commands, held for repeat steps each.

    Evaluate candidate sequences in the model, execute only the first command,
    then replan from the NEW simulator observation. No target observations leak in.
    """
    def __init__(self, transition, horizon=12, candidates=96, iterations=3, repeat=3, seed=0):
        self.transition = transition
        self.horizon, self.candidates, self.iterations, self.repeat = horizon, candidates, iterations, repeat
        self.rng = torch.Generator().manual_seed(seed)
        self.mean = torch.zeros(horizon, 1)

    @torch.inference_mode()
    def action(self, obs):
        mean, std = self.mean.clone(), torch.ones(self.horizon, 1)*1.4
        initial = torch.tensor(obs, dtype=torch.float32).repeat(self.candidates, 1)
        for _ in range(self.iterations):
            seq = (mean + std*torch.randn(self.candidates, self.horizon, 1, generator=self.rng)).clamp(-2, 2)
            state, cost = initial.clone(), torch.zeros(self.candidates)
            for h in range(self.horizon):
                for _ in range(self.repeat):
                    cost += running_cost(state, seq[:, h])
                    state = self.transition(state, seq[:, h])
            elite = seq[torch.topk(cost, k=max(4, self.candidates//8), largest=False).indices]
            mean, std = elite.mean(0), elite.std(0).clamp_min(.1)
        self.mean = torch.cat([mean[1:], mean[-1:]], dim=0)
        return mean[0].numpy()


def rollout(model, name, seeds, args, out):
    env = gym.make("Pendulum-v1")
    rows, frames = [], []
    for i, seed in enumerate(seeds):
        print(f"Evaluating {name}, episode {i+1}/{len(seeds)}", flush=True)
        obs, _ = env.reset(seed=int(seed))
        rng = np.random.default_rng(int(seed))
        planner = None if model is None else CEMPlanner(model, args.horizon, args.candidates, args.iterations, args.repeat, int(seed))
        total, upright = 0., []
        for step in range(200):
            if step % args.repeat == 0:
                action = rng.uniform(-2, 2, 1).astype(np.float32) if planner is None else planner.action(obs)
            obs, reward, done, trunc, _ = env.step(action)
            total += reward
            upright.append(int(abs(np.arctan2(obs[1], obs[0])) < .2 and abs(obs[2]) < 1.))
            if i == 0 and step % 2 == 0:
                frames.append(pendulum_frame(obs, f"{name} | step {step+1} | return {total:.0f}"))
            if done or trunc:
                break
        rows.append({"policy": name, "condition": f"action_repeat_{args.repeat}", "seed": int(seed),
                     "return": total, "upright_fraction": float(np.mean(upright))})
    env.close()
    save_gif(frames, out / f"{name}.gif", duration=100)
    return rows


def multistep_error(model, val):
    records = []
    with torch.inference_mode():
        for episode in np.unique(val["episode"]):
            idx = np.flatnonzero(val["episode"] == episode)
            for horizon in [1, 5, 10, 25]:
                starts = np.arange(0, len(idx)-horizon+1, 25)
                state = torch.tensor(val["obs"][idx[starts]], dtype=torch.float32)
                for step in range(horizon):
                    action = torch.tensor(val["action"][idx[starts+step]], dtype=torch.float32)
                    state = model(state, action)
                truth = torch.tensor(val["next_obs"][idx[starts+horizon-1]], dtype=torch.float32)
                records.append({"episode": int(episode), "horizon": horizon, "state_mse": ((state-truth)**2).mean().item()})
    return pd.DataFrame(records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--updates", type=int, default=1500)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--candidates", type=int, default=96)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--repeat", type=int, default=3)
    args = p.parse_args()
    if args.quick:
        args.episodes, args.updates, args.eval_episodes = 12, 300, 2
        args.horizon, args.candidates, args.iterations = 8, 32, 2
    if min(args.episodes, args.updates, args.eval_episodes, args.horizon, args.iterations, args.repeat) < 1 or args.candidates < 4:
        p.error("Counts must be positive and candidates >=4")
    setup_torch(args.seed)
    out = start_run("03-world-model", args)
    train, val = collect(range(args.episodes)), collect(range(2000, 2010))
    if set(train["episode"]) & set(val["episode"]):
        raise ValueError("Episode leakage")
    np.savez_compressed(out / "raw_train.npz", **train)
    np.savez_compressed(out / "raw_validation.npz", **val)
    print("Training a dynamics predictor, NOT an action predictor...", flush=True)
    model, history = fit_model(train, val, args.updates, args.seed)
    torch.save(model.state_dict(), out / "dynamics.pt")
    model.load_state_dict(torch.load(out / "dynamics.pt", weights_only=True))
    pd.DataFrame(history).to_csv(out / "training.csv", index=False)
    error = multistep_error(model, val)
    error.to_csv(out / "multistep_error.csv", index=False)
    fig, ax = plt.subplots()
    error.groupby("horizon").state_mse.mean().plot(ax=ax, marker="o")
    ax.set(xlabel="prediction horizon (steps)", ylabel="state MSE", title="Learned dynamics: compounding prediction error")
    fig.tight_layout()
    fig.savefig(out / "model_error.png", dpi=150)
    plt.close(fig)
    seeds = list(range(10000, 10000+args.eval_episodes))
    rows = []
    for name, transition in [("random", None), ("known_dynamics_mpc", known_transition), ("learned_dynamics_mpc", model)]:
        rows += rollout(transition, name, seeds, args, out)
    frame = summarize(rows, out, "return", "Learned model -> plan -> execute -> observe")
    write_paired_delta(frame, "random", "return", out)
    print("DONE. Compare offline model_error.png with closed-loop comparison.png.", flush=True)


if __name__ == "__main__":
    main()
