# 实际验证记录

## 新增后续课，2026-09-10

- 第二课 `notebooks/02_强化学习与经验回放.ipynb`：21 个单元格，11 个代码格从新 kernel 顺序完成，HTML 已导出。SAC 2k → 10k，16 个开发初始条件，每个策略分别测试正常/延迟/重力变化。计算阶段约 14.1 秒。
- 第三课 `notebooks/03_世界模型与规划迭代.ipynb`：24 个单元格，13 个代码格从新 kernel 顺序完成，HTML 已导出。20 条初始训练轨迹、8 条追加轨迹、每阶段 1,000 更新、8 个开发初始条件，包含随机数据与规划器数据的等预算重训对照。计算阶段约 15.1 秒。
- 时间来自当前 Mac CPU、2 个 PyTorch 计算线程；不含依赖安装、kernel 启动或人工阅读，不保证其他机器相同。
- 新增检查：真实下一状态与动作保持周期、跨 episode 多步预测诊断、加载评测 checkpoint 时保持 CPU 随机数状态。总计 `10 passed`；脚本 Ruff 检查通过。
- 两课保留真实数据表、训练/评测图和嵌入式 GIF。参考讲评使用本次实际结果，包含小收益与无法清晰区分的结果。
- 新增课的已知方程基线、单模型 MPC 和 SAC 小预算均为教学实验；ACT、Diffusion、SmolVLA 仍未完成训练验证。

完整快照：`runs/notebook02-20260910-093347-645000` 与 `runs/notebook03-20260910-093403-308000`。源代码和 Notebook source hash 见各自 manifest。

## Notebook，2026-09-09

- 文件：`notebooks/01_机械臂模仿学习.ipynb`。
- 从新 kernel 顺序执行 30 个单元格，包含 16 个代码格，全部完成，无 error output。
- CPU / arm64 / Python 3.12.14；具体依赖见 `example_results/01-notebook/manifest.json` 与 `uv.lock`。
- 原始运行：`runs/notebook01-20260909-221649-436000`。计算阶段实际墙钟时间约 8.6 秒；不包含启动、依赖安装或人工阅读时间。
- 80 条训练、16 条验证轨迹；每阶段 2000 updates；40 条追加轨迹；30 个开发评测初始种子。
- 通过轨迹连续性、episode 切分、动作范围、NaN 拦截与保存加载一致性检查。
- Notebook 源码单元格 SHA256：`a2331821003da8f129195a05cf4362e9d2030da0fe437b081a54aeb5918967d1`。
- HTML 已验证嵌入曲线/表格和 3 个 GIF；不需要重新训练即可阅读。图表和示意回放已做视觉检查。
- 数值与参考解读见 `notebooks/参考复盘.md`，可审计表格见 `example_results/01-notebook/`。

## 脚本验证

- 首版 `pytest -q`：7 passed；当前扩展后 10 passed。测试覆盖专家动作/动力学合同、轨迹时间边界、checkpoint、时间截断、Pendulum 物理方程一致性与规划动作范围。
- 模仿学习脚本已实际完成默认训练与控制组评测。
- SAC 已运行 10,000 环境步，6 个评测 episodes；PPO 仅完成约 1,024 环境步的链路检查，不能声称收敛。
- 世界模型 + MPC 已运行 60 条训练轨迹、1,500 updates、5 个评测 episodes。
- SAC 与 MPC 的控制协议不同，不能用这些结果直接排列算法优劣。
- 前沿 LeRobot ACT / Diffusion / SmolVLA：提供锁定上游代码与数据版本的计划生成器，尚未安装其独立环境或完成其训练、评测。源码核查不等于运行验证。

## 复现与边界

交付包排除 `.venv`、缓存和大体积原始运行；保留源码、锁文件、已执行 Notebook、HTML、讲义与部分结果证据。此 GitHub 分支另外保留完整历史数据与 checkpoint 于 `runs/`；下载压缩包仍采用轻量内容，可运行 Notebook 重新生成数据。

所有结果是教学规模的本地仿真；区间是单一训练 seed 条件下的 episode bootstrap，不包含训练随机性。没有启动 AWS、连接真机、公开上传或提交外部 PR。
# 02A 补充练习验证（2026-09-13）

- `02A_SAC直觉与奖励设计挑战.ipynb` 从干净 kernel 完成默认执行；保存表格、曲线与真实仿真 GIF。默认未运行第六题训练。
- `test_sac_challenges.py`：4 项通过，检查奖励来自动作前状态、裁剪、奖励 wrapper 不改变轨迹或 timeout、零动作延迟对照，以及 SAC bootstrap 的终止掩码。
- 可选第六题的原 Notebook 代码另以每组 600 步执行，两组均完成训练、checkpoint 保存和 8 个固定初始 seed 的评测；这是分支运行检查，不是收敛或算法效果验证。
- 第二课原 Notebook 仅更新第 2/3 节说明；原代码单元与输出保持一致，并重新导出 HTML。新增辅助 Notebook 的源文件生成器为 `tools/build_sac_challenges.py`。
- 默认输出：`runs/notebook02A-20260913-171527-120000/`；可选分支检查：`runs/challenge02A-training-smoke-1789290969367514000/`。这些运行目录按现有 Git 忽略规则保留在本地。
