# 实际验证记录

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

- `pytest -q`：7 passed；`ruff check`：通过。测试覆盖专家动作/动力学合同、轨迹时间边界、checkpoint、时间截断、Pendulum 物理方程一致性与规划动作范围。
- 模仿学习脚本已实际完成默认训练与控制组评测。
- SAC 已运行 10,000 环境步，6 个评测 episodes；PPO 仅完成约 1,024 环境步的链路检查，不能声称收敛。
- 世界模型 + MPC 已运行 60 条训练轨迹、1,500 updates、5 个评测 episodes。
- SAC 与 MPC 的控制协议不同，不能用这些结果直接排列算法优劣。
- 前沿 LeRobot ACT / Diffusion / SmolVLA：提供锁定上游代码与数据版本的计划生成器，尚未安装其独立环境或完成其训练、评测。源码核查不等于运行验证。

## 复现与边界

交付包排除 `.venv`、缓存和大体积原始运行；保留源码、锁文件、已执行 Notebook、HTML、讲义与部分结果证据。此 GitHub 分支另外保留完整历史数据与 checkpoint 于 `runs/`；下载压缩包仍采用轻量内容，可运行 Notebook 重新生成数据。

所有结果是教学规模的本地仿真；区间是单一训练 seed 条件下的 episode bootstrap，不包含训练随机性。没有启动 AWS、连接真机、公开上传或提交外部 PR。
