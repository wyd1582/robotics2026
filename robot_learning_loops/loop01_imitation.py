"""Real MuJoCo Reacher loop: expert data -> BC -> rollout -> DAgger-style repair.

This is an original educational implementation, not a reproduction of a paper's scores.
Teacher uses simulator state/dynamics (privileged); student sees standard 10D observations.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import deque

import gymnasium as gym
import mujoco
import numpy as np
import pandas as pd
import torch
from torch import nn

from common import (
    arm_frame,
    plt,
    save_gif,
    setup_torch,
    start_run,
    summarize,
    write_paired_delta,
)


def expert_action(env):
    """Analytic IK + computed torque. Specific to the bundled two-link Reacher XML."""
    u = env.unwrapped
    model, data = u.model, u.data
    x, y = data.qpos[2:4]
    l1, l2 = .1, .11
    cosine = np.clip((x*x + y*y - l1*l1 - l2*l2) / (2*l1*l2), -1, 1)
    q2 = np.arccos(cosine)  # fixed elbow branch makes the teacher unambiguous
    q1 = np.arctan2(y, x) - np.arctan2(l2*np.sin(q2), l1+l2*np.cos(q2))
    error = (np.array([q1, q2])-data.qpos[:2]+np.pi) % (2*np.pi)-np.pi
    desired_acc = np.clip(100*error - 20*data.qvel[:2], -150, 150)
    inverse = mujoco.MjData(model)
    inverse.qpos[:] = data.qpos
    inverse.qvel[:] = data.qvel
    mujoco.mj_forward(model, inverse)
    inverse.qacc[:] = 0
    inverse.qacc[:2] = desired_acc
    mujoco.mj_inverse(model, inverse)
    # Actuator gear is 200 in the verified XML: ctrl is NOT raw joint torque.
    action = inverse.qfrc_inverse[:2] / model.actuator_gear[:2, 0]
    return np.clip(action, -1, 1).astype(np.float32)


class Actor(nn.Module):
    def __init__(self, x_mean, x_std, a_mean, a_std):
        super().__init__()
        for name, value in [("x_mean", x_mean), ("x_std", x_std), ("a_mean", a_mean), ("a_std", a_std)]:
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(10, 128), nn.Tanh(), nn.Linear(128, 128),
                                 nn.Tanh(), nn.Linear(128, 2))

    def standardized(self, x):
        return self.net((x-self.x_mean)/self.x_std)

    def forward(self, x):
        return torch.clamp(self.standardized(x)*self.a_std+self.a_mean, -1, 1)

    def action(self, obs):
        with torch.inference_mode():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.x_mean.device)
            return self(x).cpu().numpy()


def collect(seeds, learner=None, beta=.2, noise=0., rng_seed=0):
    """Labels always come from teacher; executed behavior can be a learner mixture."""
    env = gym.make("Reacher-v5")
    rng = np.random.default_rng(rng_seed)
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        for step in range(50):
            label = expert_action(env)
            act = label if learner is None or rng.random() < beta else learner.action(obs)
            executed = np.clip(act+rng.normal(0, noise, 2), -1, 1).astype(np.float32)
            nxt, reward, terminated, truncated, _ = env.step(executed)
            rows.append((obs.copy(), label, executed, nxt.copy(), int(seed), step,
                         step*env.unwrapped.dt, reward, terminated, truncated))
            obs = nxt
            if terminated or truncated:
                break
    env.close()
    keys = ["obs", "expert_action", "executed_action", "next_obs", "episode", "step",
            "sample_time", "reward", "terminated", "truncated"]
    return {k: np.asarray([row[i] for row in rows]) for i, k in enumerate(keys)}


def fit(actor, data, val, steps, seed, phase):
    device = actor.x_mean.device
    x = torch.tensor(data["obs"], dtype=torch.float32, device=device)
    y = torch.tensor(data["expert_action"], dtype=torch.float32, device=device)
    vx = torch.tensor(val["obs"], dtype=torch.float32, device=device)
    vy = torch.tensor(val["expert_action"], dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    history = []
    for step in range(steps):
        idx = torch.as_tensor(rng.integers(0, len(x), size=256), device=device)
        pred = actor.standardized(x[idx])
        loss = ((pred-(y[idx]-actor.a_mean)/actor.a_std)**2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 5.)
        optimizer.step()
        if step % max(1, steps//10) == 0 or step == steps-1:
            with torch.inference_mode():
                vl = ((actor(vx)-vy)**2).mean().item()
            history.append({"phase": phase, "step": step+1, "train_normalized_mse": loss.item(), "val_action_mse": vl})
    return history


def evaluate(actor, name, seeds, delay, out):
    env = gym.make("Reacher-v5")
    rows, frames = [], []
    for i, seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(seed))
        history = deque([obs.copy()]*(delay+1), maxlen=delay+1)
        rng = np.random.default_rng(int(seed))
        total, distances = 0., []
        for step in range(50):
            if name == "teacher":
                action = expert_action(env)
            elif name == "random":
                action = rng.uniform(-1, 1, 2).astype(np.float32)
            elif name == "zero":
                action = np.zeros(2, dtype=np.float32)
            else:
                action = actor.action(history[0])
            obs, reward, done, trunc, _ = env.step(action)
            history.append(obs.copy())
            total += reward
            distances.append(float(np.linalg.norm(obs[-2:])))
            if i == 0 and delay == 0:
                u = env.unwrapped
                frames.append(arm_frame(u.data.qpos[:2], u.data.qpos[2:4],
                                        f"{name} | step {step+1} | distance {distances[-1]:.3f} m"))
            if done or trunc:
                break
        rows.append({"policy": name, "condition": f"delay_{delay}", "seed": int(seed),
                     "return": total, "final_distance": distances[-1],
                     "last10_mean_distance": float(np.mean(distances[-10:])),
                     "success": int(np.mean(distances[-10:]) < .025)})
    env.close()
    if frames:
        save_gif(frames, out / f"{name}.gif", duration=40)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="Small pipeline verification, not convergence benchmark")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--updates", type=int, default=2000)
    p.add_argument("--eval-episodes", type=int, default=30)
    args = p.parse_args()
    if args.quick:
        args.episodes, args.updates, args.eval_episodes = 24, 400, 8
    if min(args.episodes, args.updates, args.eval_episodes) < 1:
        p.error("Counts must be positive")
    setup_torch(args.seed, args.device)
    out = start_run("01-imitation", args)
    train = collect(range(args.episodes), noise=.015, rng_seed=args.seed)
    val = collect(range(1000, 1016), rng_seed=args.seed+1)
    np.savez_compressed(out / "raw_train.npz", **train)
    np.savez_compressed(out / "raw_validation.npz", **val)
    if not all(np.isfinite(train[k]).all() for k in ["obs", "expert_action", "executed_action", "next_obs"]):
        raise ValueError("Nonfinite data; stop before training")
    if set(train["episode"]) & set(val["episode"]):
        raise ValueError("Episode leakage")
    audit = {"train_rows": len(train["obs"]), "train_episodes": args.episodes,
             "validation_episodes": 16, "action_label": "computed-torque teacher command",
             "sample_time_unit": "seconds, reset within episode", "observation_dimension": 10,
             "control_period_seconds": .02, "actuator_gear": 200,
             "success_definition": "mean fingertip-target distance over last 10 steps < 0.025m; custom metric",
             "eval_seed_range": [10000, 10000+args.eval_episodes-1]}
    (out / "data_audit.json").write_text(json.dumps(audit, indent=2))
    actor = Actor(train["obs"].mean(0), np.maximum(train["obs"].std(0), .01),
                  train["expert_action"].mean(0), np.maximum(train["expert_action"].std(0), .01)).to(args.device)
    print("Fitting behavior cloning...", flush=True)
    history = fit(actor, train, val, args.updates, args.seed, "bc")
    # Save and reload before deployment: checkpoint is a tensor-only state_dict.
    torch.save(actor.state_dict(), out / "bc.pt")
    actor.load_state_dict(torch.load(out / "bc.pt", weights_only=True, map_location=args.device))
    extra_count = max(8, args.episodes//2)
    print("Collecting learner-visited states and teacher corrections...", flush=True)
    extra = collect(range(2000, 2000+extra_count), learner=actor, beta=.2, rng_seed=args.seed)
    more_expert = collect(range(2000, 2000+extra_count), rng_seed=args.seed)
    np.savez_compressed(out / "raw_dagger.npz", **extra)
    np.savez_compressed(out / "raw_more_expert.npz", **more_expert)
    variants = {"bc": actor}
    for label, addition in [("more_updates", None), ("more_expert", more_expert), ("dagger", extra)]:
        model = copy.deepcopy(actor)
        data = train if addition is None else {k: np.concatenate([train[k], addition[k]]) for k in train}
        history += fit(model, data, val, args.updates, args.seed+1, label)
        torch.save(model.state_dict(), out / f"{label}.pt")
        model.load_state_dict(torch.load(out / f"{label}.pt", weights_only=True, map_location=args.device))
        variants[label] = model
    pd.DataFrame(history).to_csv(out / "training.csv", index=False)
    fig, ax = plt.subplots()
    for label, group in pd.DataFrame(history).groupby("phase", sort=False):
        ax.plot(group.step, group.val_action_mse, label=label)
    ax.set(xlabel="updates within phase", ylabel="held-out expert action MSE", yscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "offline_loss.png", dpi=150)
    plt.close(fig)
    print("Deploying reloaded policies into independent simulation episodes...", flush=True)
    rows = []
    seeds = range(10000, 10000+args.eval_episodes)
    for name in ["random", "zero", "teacher"]:
        rows += evaluate(None, name, seeds, 0, out)
    for name, model in variants.items():
        for delay in [0, 2]:
            rows += evaluate(model, name, seeds, delay, out)
    frame = summarize(rows, out, "last10_mean_distance", "Reacher: lower distance is better")
    write_paired_delta(frame, "bc", "last10_mean_distance", out)
    print("DONE. Open comparison.png, bc.gif, dagger.gif and data_audit.json.", flush=True)


if __name__ == "__main__":
    main()
