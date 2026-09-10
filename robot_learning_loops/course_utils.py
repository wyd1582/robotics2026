"""Small, inspectable data and presentation helpers for the follow-up notebooks."""
from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import io
import json
import random
import time
from argparse import Namespace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from common import ROOT, pendulum_frame, save_gif, setup_torch, start_run
from loop03_world_model import CEMPlanner


@contextlib.contextmanager
def preserve_cpu_rng():
    """Keep evaluation checkpoint loading from perturbing an ongoing CPU run."""
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])


def begin_notebook(name, filename, config):
    setup_torch(config["seed"], "cpu")
    with contextlib.redirect_stdout(io.StringIO()):
        out = start_run(name, Namespace(**config))
    notebook = json.loads((ROOT / "notebooks" / filename).read_text())
    sources = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
               for c in notebook["cells"]]
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["notebook_source_sha256"] = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False).encode()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Results:", out.relative_to(ROOT))
    print("CPU threads:", torch.get_num_threads())
    return out, manifest


def show_gif(path):
    from IPython.display import HTML, display
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    display(HTML(f'<img width="400" alt="Actual simulated trajectory" src="data:image/gif;base64,{encoded}">'))


def audit_transitions(data, observation_dim=3, require_closed=False):
    n = len(data["obs"])
    assert n > 0 and data["obs"].shape == (n, observation_dim)
    assert data["next_obs"].shape == (n, observation_dim)
    assert data["action"].shape == (n, 1)
    assert all(len(value) == n for value in data.values())
    for field in ["obs", "action", "next_obs", "reward"]:
        assert np.isfinite(data[field]).all(), f"Nonfinite {field}"
    assert np.abs(data["action"]).max() <= 2 + 1e-6
    assert np.allclose(np.linalg.norm(data["obs"][:, :2], axis=1), 1, atol=1e-5)
    for episode in np.unique(data["episode"]):
        indices = np.flatnonzero(data["episode"] == episode)
        assert np.array_equal(data["step"][indices], np.arange(len(indices)))
        np.testing.assert_allclose(data["next_obs"][indices[:-1]], data["obs"][indices[1:]])
        ended = data["terminated"][indices] | data["truncated"][indices]
        assert not ended[:-1].any(), "Transition after episode end"
        if require_closed:
            assert ended[-1], "Incomplete episode"
    return {"rows": n, "episodes": len(np.unique(data["episode"])), "audit": "PASS"}


def reload_dynamics(model, out, name, validation):
    path = out / f"{name}.pt"
    obs = torch.tensor(validation["obs"][:32], dtype=torch.float32)
    action = torch.tensor(validation["action"][:32], dtype=torch.float32)
    torch.save(model.state_dict(), path)
    loaded = copy.deepcopy(model)
    loaded.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    with torch.inference_mode():
        torch.testing.assert_close(model(obs, action), loaded(obs, action), rtol=0, atol=0)
    return loaded.eval()


def prediction_diagnostics(model, validation, horizons=(1, 5, 10, 25)):
    records = []
    with torch.inference_mode():
        for episode in np.unique(validation["episode"]):
            idx = np.flatnonzero(validation["episode"] == episode)
            for horizon in horizons:
                starts = np.arange(0, len(idx) - horizon + 1, 25)
                state = torch.tensor(validation["obs"][idx[starts]], dtype=torch.float32)
                for step in range(horizon):
                    action = torch.tensor(validation["action"][idx[starts + step]], dtype=torch.float32)
                    state = model(state, action)
                truth = torch.tensor(validation["next_obs"][idx[starts + horizon - 1]], dtype=torch.float32)
                angle = torch.atan2(state[:, 1], state[:, 0]) - torch.atan2(truth[:, 1], truth[:, 0])
                angle = torch.atan2(angle.sin(), angle.cos())
                records.append({"episode": int(episode), "horizon": horizon,
                                "angle_rmse_rad": angle.square().mean().sqrt().item(),
                                "velocity_rmse_rad_s": (state[:, 2] - truth[:, 2]).square().mean().sqrt().item(),
                                "state_mse": (state - truth).square().mean().item()})
    return pd.DataFrame(records)


def collect_with_planner(transition, name, seeds, config, out=None, episode_steps=200):
    """Same simulator, action repeat and seeds for random and planned collection.

    Returns actual transitions and episode metrics. Planner calls use current
    observed state; its imagined next states never become training labels.
    """
    env = gym.make("Pendulum-v1", max_episode_steps=episode_steps)
    records, scores, frames = [], [], []
    try:
        for i, seed in enumerate(seeds):
            obs, _ = env.reset(seed=int(seed))
            rng = np.random.default_rng(int(seed))
            planner = None if transition is None else CEMPlanner(
                transition, horizon=config["horizon"], candidates=config["candidates"],
                iterations=config["iterations"], repeat=config["repeat"], seed=int(seed))
            timings, upright, total = [], [], 0.
            for step in range(episode_steps):
                if step % config["repeat"] == 0:
                    started = time.perf_counter()
                    action = (rng.uniform(-2, 2, 1).astype(np.float32)
                              if planner is None else planner.action(obs))
                    timings.append(time.perf_counter() - started)
                nxt, reward, terminated, truncated, _ = env.step(action)
                records.append((obs.copy(), action.copy(), nxt.copy(), int(seed), step,
                                reward, terminated, truncated))
                obs = nxt
                total += reward
                upright.append(abs(np.arctan2(obs[1], obs[0])) < .2 and abs(obs[2]) < 1)
                if out is not None and i == 0 and step % 2 == 0:
                    frames.append(pendulum_frame(obs, f"{name} | step {step + 1} | return {total:.0f}"))
                if terminated or truncated:
                    break
            scores.append({"policy": name, "condition": f"action_repeat_{config['repeat']}",
                           "seed": int(seed), "return": total, "upright_fraction": np.mean(upright),
                           "planning_p50_ms": np.quantile(timings, .5) * 1000,
                           "planning_p95_ms": np.quantile(timings, .95) * 1000})
    finally:
        env.close()
    if out is not None:
        save_gif(frames, out / f"{name}.gif", duration=100)
    keys = ["obs", "action", "next_obs", "episode", "step", "reward", "terminated", "truncated"]
    data = {key: np.asarray([row[i] for row in records]) for i, key in enumerate(keys)}
    return data, scores


def paired_return(frame, candidate, reference):
    records = []
    rng = np.random.default_rng(123)
    for condition, group in frame.groupby("condition"):
        pivot = group.pivot(index="seed", columns="policy", values="return")
        if candidate not in pivot or reference not in pivot:
            continue
        delta = (pivot[candidate] - pivot[reference]).dropna().to_numpy()
        means = rng.choice(delta, size=(3000, len(delta)), replace=True).mean(axis=1)
        records.append({"candidate": candidate, "reference": reference, "condition": condition,
                        "mean_delta": delta.mean(), "ci_low": np.quantile(means, .025),
                        "ci_high": np.quantile(means, .975), "n_pairs": len(delta)})
    return pd.DataFrame(records)


def write_review(out, title, prediction, questions, frame, elapsed):
    summary = frame.groupby(["policy", "condition"])[["return", "upright_fraction"]].mean()
    report = f"# {title}\n\n最初预测：{prediction}\n\n实际结果：\n```text\n{summary.round(4).to_string()}\n```\n\n"
    report += "\n\n".join(f"## {question}\n\n{answer}" for question, answer in questions.items())
    report += "\n\n边界：一个训练 seed；开发评测；低维 Pendulum 仿真。\n"
    (out / "review.md").write_text(report, encoding="utf-8")
    (out / "notebook_summary.json").write_text(json.dumps(
        {"elapsed_seconds": elapsed, "review_answers": questions,
         "status": "reached final computational cell", "rows": len(frame)}, indent=2, ensure_ascii=False))
    print(f"Elapsed from setup: {elapsed:.1f} seconds (includes reading time in an interactive run).")
    print("Saved:", (out / "review.md").relative_to(ROOT))
