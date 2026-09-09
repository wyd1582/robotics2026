"""Version-grounded LeRobot recipes. Dry run by default; --execute runs locally.

No cloud launch, uploads, robot hardware access, or implicit training.
Training/evaluation recipes are source-checked, not runtime-verified in this kit.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMIT = "2774d9bddcbbda50e697e162e89e7eaada8d7105"
DATA = {"pusht": ("lerobot/pusht", "7628202a2180972f291ba1bc6723834921e72c19"),
        "libero": ("lerobot/libero", "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4")}


def build_commands(args, repo, output):
    uv = shutil.which("uv") or "uv"
    extras = ["training", "evaluation", "smolvla", "libero"] if args.track == "smolvla" else ["training", "evaluation", "pusht", "diffusion"]
    sync = [uv, "sync", "--project", str(repo), "--no-dev"]
    for extra in extras:
        sync += ["--extra", extra]
    if args.stage == "prepare":
        commands = []
        if not repo.exists():
            commands += [["git", "clone", "--filter=blob:none", "--no-checkout", "https://github.com/huggingface/lerobot.git", str(repo)],
                         ["git", "-C", str(repo), "checkout", "--detach", COMMIT]]
        commands.append(sync)
        return commands
    prefix = [uv, "run", "--project", str(repo), "--no-sync"]
    dataset, revision = DATA["libero" if args.track == "smolvla" else "pusht"]
    if args.stage == "train":
        command = prefix + ["lerobot-train", f"--policy.type={args.track}",
                            "--policy.push_to_hub=false", "--save_checkpoint_to_hub=false",
                            f"--dataset.repo_id={dataset}", f"--dataset.revision={revision}",
                            "--dataset.video_backend=pyav", f"--policy.device={args.device}",
                            "--dataset.eval_split=0.1", f"--eval_steps={args.steps}", "--max_eval_samples=128",
                            f"--steps={args.steps}", f"--batch_size={args.batch_size}",
                            "--num_workers=0", "--persistent_workers=false", "--env_eval_freq=0",
                            f"--save_freq={args.steps}", "--log_freq=10", "--wandb.enable=false",
                            f"--seed={args.seed}", f"--output_dir={output}"]
        if args.track == "smolvla":
            command += ["--policy.load_vlm_weights=true"]
        return [command]
    if not args.checkpoint:
        raise ValueError("eval requires --checkpoint pointing to a saved pretrained_model directory")
    command = prefix + ["lerobot-eval", f"--policy.path={Path(args.checkpoint).expanduser().resolve()}",
                        f"--policy.device={args.device}", f"--seed={args.seed}",
                        "--eval.batch_size=1", f"--eval.n_episodes={args.eval_episodes}",
                        f"--output_dir={output}"]
    if args.track == "smolvla":
        command += ["--env.type=libero", "--env.task=libero_spatial", "--env.task_ids=[0]",
                    "--env.max_parallel_tasks=1", "--env.init_states=true", "--env.hard_reset=true",
                    "--env.control_mode=relative"]
    else:
        command += ["--env.type=pusht"]
    return [command]


def verify_checkout(repo):
    if not (repo / ".git").exists():
        raise RuntimeError("Run --stage prepare --execute first")
    sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if sha != COMMIT:
        raise RuntimeError(f"Checkout is {sha}, expected {COMMIT}. Use a separate directory; no checkout changed.")
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty:
        raise RuntimeError("Tracked checkout changes detected. Keep reproduction checkout clean or record a reviewed patch separately.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--track", choices=["act", "diffusion", "smolvla"], required=True)
    p.add_argument("--stage", choices=["prepare", "inspect", "train", "eval"], required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--repo", type=Path, default=ROOT / "external" / "lerobot")
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--checkpoint")
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.eval_episodes) < 1:
        p.error("Counts must be positive")
    repo = args.repo.expanduser().resolve()
    suffix = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()%1000000:06d}"
    output = ROOT / "runs" / f"frontier-{args.track}-{args.stage}-{suffix}"
    print("LeRobot commit:", COMMIT)
    if args.stage == "inspect":
        dataset, revision = DATA["libero" if args.track == "smolvla" else "pusht"]
        url = f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/meta/info.json"
        print("Metadata URL:", url)
        if args.execute:
            with urllib.request.urlopen(url, timeout=30) as response:
                metadata = json.load(response)
            output.mkdir(parents=True, exist_ok=False)
            (output / "dataset_info.json").write_text(json.dumps(metadata, indent=2))
            (output / "source.json").write_text(json.dumps({"dataset": dataset, "revision": revision, "url": url}, indent=2))
            print(json.dumps(metadata, indent=2))
            print("Saved:", output)
        else:
            print("Dry run: no download. Add --execute for bounded metadata only.")
        return
    if args.execute and args.track == "smolvla" and sys.platform != "linux":
        p.error("This LIBERO track requires Linux. Use dry run on Mac; run on a Linux host when ready.")
    try:
        commands = build_commands(args, repo, output)
    except ValueError as e:
        p.error(str(e))
    for command in commands:
        print(shlex.join(command))
    if not args.execute:
        print("Dry run only. No checkout, install, model/data download, training, or evaluation occurred.")
        return
    if repo.exists() or args.stage != "prepare":
        verify_checkout(repo)
    environment = os.environ.copy()
    environment["WANDB_MODE"] = "disabled"
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if sys.platform == "linux":
        environment.setdefault("MUJOCO_GL", "egl")
    repo.parent.mkdir(parents=True, exist_ok=True)
    ROOT.joinpath("runs").mkdir(exist_ok=True)
    plan = {"track": args.track, "stage": args.stage, "commit": COMMIT,
            "commands": commands, "runtime_verified_before_delivery": False}
    (ROOT / "runs" / f"plan-{suffix}.json").write_text(json.dumps(plan, indent=2))
    if args.stage == "eval" and not (Path(args.checkpoint).expanduser() / "config.json").is_file():
        p.error("checkpoint must contain config.json; choose the actual saved pretrained_model directory")
    for command in commands:
        subprocess.run(command, check=True, env=environment)
    if args.stage == "prepare":
        verify_checkout(repo)
    print("Completed:", args.stage)


if __name__ == "__main__":
    main()
