#!/usr/bin/env bash
# Physical AI Day-0 environment self-check.
# Usage: bash check_env.sh   (run inside your activated venv)

pass=0; fail=0
ok()   { echo "  [PASS] $1"; pass=$((pass+1)); }
bad()  { echo "  [FAIL] $1"; fail=$((fail+1)); }

echo "== System =="
command -v git >/dev/null    && ok "git $(git --version | awk '{print $3}')"        || bad "git missing"
command -v ffmpeg >/dev/null && ok "ffmpeg (needed by LeRobot video decoding)"      || bad "ffmpeg missing: sudo apt install ffmpeg"
disk_free=$(df -h --output=avail / | tail -1 | tr -d ' ')
echo "  [INFO] free disk on /: ${disk_free} (datasets need 20GB+)"

echo "== GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    && ok "NVIDIA driver responding" || bad "nvidia-smi errored"
else
  echo "  [INFO] no nvidia-smi: CPU / Apple Silicon / cloud-GPU mode"
fi

echo "== Python stack =="
python - <<'EOF'
import importlib, sys
print(f"  [INFO] python {sys.version.split()[0]}")
checks = {
    "torch": "PyTorch",
    "lerobot": "LeRobot",
    "mujoco": "MuJoCo",
    "gymnasium": "Gymnasium",
}
failed = 0
for mod, name in checks.items():
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        print(f"  [PASS] {name} {v}")
    except Exception as e:
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        failed += 1
try:
    import torch
    if torch.cuda.is_available():
        print(f"  [PASS] CUDA available: {torch.cuda.get_device_name(0)}")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("  [PASS] Apple MPS available")
    else:
        print("  [INFO] no GPU acceleration - training goes to cloud")
except Exception:
    pass
sys.exit(1 if failed else 0)
EOF
py_status=$?

echo "== Summary =="
if [ $fail -eq 0 ] && [ $py_status -eq 0 ]; then
  echo "All checks passed. Run the smoke tests next: python smoke/00_torch_gpu.py"
else
  echo "Some checks failed - fix them top to bottom before proceeding."
  exit 1
fi
