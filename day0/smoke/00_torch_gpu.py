"""Smoke test layer 1: PyTorch install + device + matmul throughput."""
import time

import torch

print(f"torch {torch.__version__}")

if torch.cuda.is_available():
    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    dev = torch.device("mps")
    print("device: Apple MPS")
else:
    dev = torch.device("cpu")
    print("device: CPU (training exercises should run on cloud GPU)")

n = 2048
a = torch.randn(n, n, device=dev)
b = torch.randn(n, n, device=dev)
for _ in range(3):  # warmup
    a @ b
if dev.type == "cuda":
    torch.cuda.synchronize()
t0 = time.time()
iters = 20
for _ in range(iters):
    c = a @ b
if dev.type == "cuda":
    torch.cuda.synchronize()
dt = (time.time() - t0) / iters
gflops = 2 * n**3 / dt / 1e9
print(f"{n}x{n} matmul: {dt*1e3:.1f} ms/iter, ~{gflops:.0f} GFLOP/s")
print("PASS")
