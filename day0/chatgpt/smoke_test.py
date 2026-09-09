"""Small CPU/MPS/CUDA training check; optional headless MuJoCo physics check.

Run inside your own project: uv run python /path/to/smoke_test.py --device mps
Add --mujoco only after installing mujoco. No downloads or saved files.
This verifies basic operations, not compatibility with an entire robotics stack.
"""
import argparse
import platform
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--mujoco", action="store_true")
    args = parser.parse_args()
    import torch

    print("python", sys.version.split()[0], "arch", platform.machine())
    print("torch", torch.__version__, "device requested", args.device)
    print("mps built", torch.backends.mps.is_built(), "available", torch.backends.mps.is_available())
    print("cuda available", torch.cuda.is_available())
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable. Check native arm64 Python, macOS and torch wheel.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check NVIDIA driver and the torch build.")

    torch.manual_seed(7)
    torch.set_num_threads(2)
    x = torch.randn(128, 4, dtype=torch.float32)
    y = x @ torch.tensor([[1.0], [-2.0], [0.5], [0.25]]) + 0.1
    model = torch.nn.Linear(4, 1).to(args.device)
    x, y = x.to(args.device), y.to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = ((model(x) - y) ** 2).mean().item()
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = ((model(x) - y) ** 2).mean()
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite loss")
        loss.backward()
        optimizer.step()
    after = ((model(x) - y) ** 2).mean().item()
    if not after < before * 0.1:
        raise RuntimeError(f"Training did not converge: {before} -> {after}")
    print(f"PASS tiny regression: loss {before:.6f} -> {after:.6f}")

    if args.mujoco:
        import mujoco
        import numpy as np

        model_mj = mujoco.MjModel.from_xml_string('''
        <mujoco><option timestep="0.002"/>
          <worldbody><body pos="0 0 1"><freejoint/>
            <geom type="sphere" size="0.05" mass="1"/>
          </body></worldbody>
        </mujoco>''')
        data = mujoco.MjData(model_mj)
        start_z = float(data.qpos[2])
        for _ in range(100):
            mujoco.mj_step(model_mj, data)
        if not (data.time > 0 and np.isfinite(data.qpos).all() and data.qpos[2] < start_z):
            raise RuntimeError("MuJoCo physics check failed")
        print("PASS headless MuJoCo", mujoco.__version__, "time", data.time)


if __name__ == "__main__":
    main()
