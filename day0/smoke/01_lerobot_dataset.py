"""Smoke test layer 2: pull a LeRobot dataset and interrogate its structure.

This doubles as Exercise 0 scaffolding: every field printed here is a field
you should be able to explain (fps, action space, episode boundaries, ...).
"""
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "lerobot/pusht"  # small (~200MB); swap for aloha/droid subsets later

ds = LeRobotDataset(REPO_ID)
print(f"dataset: {REPO_ID}")
print(f"episodes: {ds.num_episodes}, frames: {ds.num_frames}, fps: {ds.fps}")
print(f"features: {list(ds.features.keys())}")

frame = ds[0]
for k, v in frame.items():
    shape = tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
    print(f"  {k}: {shape}")

# Questions to answer by hand (put answers in physical-ai-notes):
# 1. What exactly is one "action" here - target position or delta? In what frame?
# 2. How is episode success defined and stored?
# 3. What is the observation->action latency implied by the recording setup?
print("PASS")
