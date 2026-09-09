# robotics2026 · Physical AI 训战留痕

对应《Physical AI 训战合一计划》的实操留痕区。目录规范：

```
robotics2026/
├── day0/          # 新机自检、分层安装、冒烟测试（先做这个）
├── notebooks/     # NB00–NB08 递进闭环训练系列（主线：跑→输出→分析→复盘）
│   ├── nbutils.py #   共享工具：结果账本、评测CI、数据集兼容加载
│   └── results/   #   每本 notebook 的落盘账本（json/png/csv，append-only）
├── week01/        # 每周一个目录：paper notes + 口试自答
├── week02/        #   （允许粗糙，禁止事后美化——留痕的价值在真实）
└── experiments/   # 溢出 notebooks 的大实验，成熟后并入 physical-ai-data-observatory
```

主线是 `notebooks/`（见其 README 的递进地图与运行纪律）；day0 的
`exercises/EXERCISES.md` 保留为速览版索引，具体执行以 notebook 为准。

约定：

- 每周目录里放 `NOTES.md`（学习笔记）、`oral-exam.md`（本周口试题书面自答）、代码/notebook。
- commit message 英文、动词开头、说清 why；每周 ≥4 天有 commit。
- 练习与实验产出可复用的部分，最终迁往独立仓库（observatory / notes / audits / so101-loop），
  这里是它们的孵化处。


## ChatGPT：可运行的机器人学习闭环

新增材料位于 [`robot_learning_loops/`](robot_learning_loops/README.md)，与现有 NB00–NB08 系列并列，使用独立的 `uv` 环境。

| 入口 | 内容 |
|---|---|
| [第一课 Notebook](robot_learning_loops/notebooks/01_机械臂模仿学习.ipynb) | 30 个单元格：采集、审计、BC 训练、执行、DAgger 式纠正、对照、复盘；保留已执行输出 |
| [HTML 预览](robot_learning_loops/notebooks/01_机械臂模仿学习.html) | 下载到本地用浏览器打开，查看完整表格、曲线与 GIF |
| [参考复盘](robot_learning_loops/notebooks/参考复盘.md) | 先写自己的判断，再对照讲评 |
| [学习路线](robot_learning_loops/lessons/00_闭环地图.md) | 模仿学习 → 强化学习 → 世界模型/MPC → ACT/Diffusion → VLA |
| [结果与验证](robot_learning_loops/VALIDATION.md) | 已跑通的范围、实际数据与尚未验证的前沿项目 |
| [Mac 起步工具](day0/chatgpt/START_HERE.md) | 新机自检、Torch smoke test、时间语义练习与安装指南 |
| [轻量下载包](downloads/) | 整套练习和起步工具的便携压缩包 |

### 开始第一课

在仓库根目录运行（需要已安装 `uv`）：

```bash
cd robot_learning_loops
uv sync --locked --group notebooks
uv run --locked --group notebooks jupyter lab notebooks/01_机械臂模仿学习.ipynb
```

默认 Mac CPU、小网络、2 个 PyTorch 计算线程，无需云 GPU。逐格按 Shift+Enter 运行；改配置后从新 kernel 重跑。结束时关闭 kernel 与 Jupyter 服务以释放内存。

本次固定 seed 的示例中，BC 正常反馈误差约 4.17 cm，纠正后约 1.37 cm；增加 40 ms 观测延迟后仍明显退化。它是教学仿真结果，不能外推真机或论文 benchmark。

发布范围及文件索引见 [EXPORT.md](EXPORT.md)。原有 `day0/` 与 `notebooks/` 内容保留。
