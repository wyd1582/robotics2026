"""Smoke test layer 3: MuJoCo physics stepping without a display."""
import time

import mujoco

XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" rgba="0.8 0.3 0.3 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

t0 = time.time()
steps = 10_000
for _ in range(steps):
    mujoco.mj_step(model, data)
dt = time.time() - t0
print(f"mujoco {mujoco.__version__}")
print(f"{steps} steps in {dt:.2f}s -> {steps/dt:,.0f} steps/s (realtime x{steps*0.002/dt:,.0f})")
print(f"box settled at z={data.qpos[2]:.3f} (expect ~0.05)")
assert abs(data.qpos[2] - 0.05) < 0.02, "physics looks wrong"
print("PASS")
