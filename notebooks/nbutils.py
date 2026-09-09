"""Shared helpers for the robotics2026 notebook series.

Conventions:
- every notebook logs its headline numbers via log_result() into results/NBxx.json,
  so later notebooks (and the capstone) can read them back;
- results files are append-only history: each run adds a timestamped record.
"""
import json
import platform
import random
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True
        ).strip()
    except Exception:
        return "unknown"


def log_result(notebook: str, payload: dict) -> Path:
    """Append one timestamped record to results/<notebook>.json."""
    path = RESULTS / f"{notebook}.json"
    history = json.loads(path.read_text()) if path.exists() else []
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "git": git_rev(),
        "host": platform.node(),
        **payload,
    }
    history.append(record)
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False))
    print(f"logged -> {path.name} (run #{len(history)})")
    return path


def load_results(notebook: str) -> list:
    path = RESULTS / f"{notebook}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} 不存在——先跑完对应 notebook 并 log_result()"
        )
    return json.loads(path.read_text())


def latest(notebook: str) -> dict:
    return load_results(notebook)[-1]


@contextmanager
def timer(label: str):
    t0 = time.time()
    yield
    print(f"[{label}] {time.time() - t0:.1f}s")


def load_lerobot_dataset(repo_id: str, episodes=None):
    """Version-compatible LeRobotDataset loader (module path moved across releases)."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if episodes is not None:
        return LeRobotDataset(repo_id, episodes=list(episodes))
    return LeRobotDataset(repo_id)


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial success rate. Returns (low, high)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (max(0.0, center - half), min(1.0, center + half))
