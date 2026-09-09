"""Shared experiment bookkeeping and plots. No network or hardware commands."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw


def start_run(name, args):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    folder = ROOT / "runs" / f"{name}-{stamp}-{time.time_ns() % 1000000:06d}"
    folder.mkdir(parents=True, exist_ok=False)
    versions = {}
    for package in ["numpy", "torch", "gymnasium", "mujoco", "stable-baselines3"]:
        versions[package] = importlib.metadata.version(package)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")}
    manifest = {"experiment": name, "args": vars(args), "versions": versions,
                "python": platform.python_version(), "architecture": platform.machine(),
                "source_sha256": hashes, "evaluation_status": "development benchmark, not paper reproduction"}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Run directory: {folder}", flush=True)
    return folder


def setup_torch(seed, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable to this process; use --device cpu or a native Terminal.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; Mac users should choose cpu or mps.")


def summarize(rows, out, metric, title):
    """Episode-level intervals are conditional on ONE trained seed, not training variability."""
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "episodes.csv", index=False)
    groups = frame.groupby(["policy", "condition"], sort=False)[metric]
    records = []
    rng = np.random.default_rng(12345)
    for (policy, condition), values in groups:
        a = values.to_numpy()
        means = rng.choice(a, size=(2000, len(a)), replace=True).mean(axis=1)
        records.append({"policy": policy, "condition": condition, "n": len(a),
                        "mean": float(a.mean()), "ci_low": float(np.quantile(means, .025)),
                        "ci_high": float(np.quantile(means, .975))})
    summary = pd.DataFrame(records)
    summary.to_csv(out / "summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary))
    ax.bar(x, summary["mean"], color="#31798b")
    ax.errorbar(x, summary["mean"],
                yerr=np.maximum(0, np.vstack([summary["mean"]-summary.ci_low,
                                              summary.ci_high-summary["mean"]])),
                fmt="none", ecolor="#172937", capsize=3)
    ax.set_xticks(x, [f"{r.policy}\n{r.condition}" for r in summary.itertuples()], rotation=25, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title + "\nEpisode bootstrap 95% CI; one training seed")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)
    print(summary.to_string(index=False), flush=True)
    return frame


def write_paired_delta(frame, reference, metric, out):
    base = frame[frame.policy == reference][["condition", "seed", metric]]
    rows = []
    rng = np.random.default_rng(321)
    for policy, group in frame.groupby("policy"):
        if policy == reference:
            continue
        merged = group.merge(base, on=["condition", "seed"], suffixes=("", "_base"), validate="one_to_one")
        for condition, sub in merged.groupby("condition"):
            delta = (sub[metric] - sub[metric + "_base"]).to_numpy()
            boot = rng.choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1)
            rows.append({"policy": policy, "reference": reference, "condition": condition,
                         "mean_delta": float(delta.mean()), "ci_low": float(np.quantile(boot, .025)),
                         "ci_high": float(np.quantile(boot, .975)), "n_pairs": len(delta)})
    pd.DataFrame(rows).to_csv(out / "paired_deltas.csv", index=False)


def arm_frame(q, target, label):
    """Schematic drawn from actual MuJoCo qpos; not photorealistic simulator rendering."""
    im = Image.new("RGB", (480, 400), "#f4f5ef")
    draw = ImageDraw.Draw(im)
    origin = np.array([240., 215.])
    def xy(p):
        return tuple(origin + np.asarray(p) * [780., -780.])
    joint = .1 * np.array([np.cos(q[0]), np.sin(q[0])])
    tip = joint + .11 * np.array([np.cos(q.sum()), np.sin(q.sum())])
    draw.line([xy([0, 0]), xy(joint), xy(tip)], fill="#24697d", width=12)
    for p in [[0, 0], joint, tip]:
        x, y = xy(p)
        draw.ellipse([x-7, y-7, x+7, y+7], fill="#172937")
    x, y = xy(target)
    draw.ellipse([x-9, y-9, x+9, y+9], outline="#d45345", width=3)
    draw.text((15, 15), label, fill="#172937")
    draw.text((15, 365), "State schematic | MuJoCo Reacher-v5 | 2 joints", fill="#172937")
    return im


def pendulum_frame(obs, label):
    im = Image.new("RGB", (400, 400), "#f4f5ef")
    draw = ImageDraw.Draw(im)
    c, s = float(obs[0]), float(obs[1])
    end = (200 + 130*s, 220 - 130*c)
    draw.line([(200, 220), end], fill="#24697d", width=12)
    draw.ellipse([end[0]-12, end[1]-12, end[0]+12, end[1]+12], fill="#d45345")
    draw.text((12, 15), label, fill="#172937")
    draw.text((12, 370), "State schematic | Gymnasium Pendulum-v1", fill="#172937")
    return im


def save_gif(frames, path, duration):
    if frames:
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
