机器人学习闭环练习册

这套材料按“先完成一笔研究交易，再理解每个模块”组织：采集原始数据 → 定义语义与标签 → 学习或规划 → 保存模型 → 在独立仿真中执行 → 分析失败 → 改进 → 对照评测。

它面向有数据科学、量化、Python/PyTorch 经验，希望尽快获得机器人研究手感的人。前三个项目已经提供可独立运行的小型完整闭环；第四、第五项提供锁定上游代码/数据版本的前沿复现入口、训练与评测脚本，尚未完成大模型训练验证。具体本次验证结果见 `验证记录.md`。

前三课已补齐为逐格 Notebook，并保留实际执行输出和同名 HTML。学习顺序与每课完成标准见 [课程指南.md](课程指南.md)。

| Notebook | 单元格 / 代码格 | 主实验 |
|---|---:|---|
| [01 机械臂模仿学习](notebooks/01_机械臂模仿学习.ipynb) | 30 / 16 | BC → 执行 → DAgger 式纠正与预算对照 |
| [02 强化学习与经验回放](notebooks/02_强化学习与经验回放.ipynb) | 21 / 11 | SAC 0/2k/10k → 数据与 timeout 审计 → 压力评测 |
| [03 世界模型与规划迭代](notebooks/03_世界模型与规划迭代.ipynb) | 24 / 13 | 动力学 → MPC → 新数据 → 对照重训 → 复评 |

先打开第一课 HTML 预览，再逐格实践。写完自己的复盘后，再看同目录参考讲评。

第一课的核心问题是：为什么专家动作预测得不错，策略执行仍会走偏？先完成这一个问题，再读 `lessons/00_闭环地图.md` 选择下一项。

启动交付版 Notebook（从仓库根目录运行）：

```bash
cd robot_learning_loops
uv run --locked --group notebooks jupyter lab notebooks/01_机械臂模仿学习.ipynb
```

默认使用小网络和 CPU。关闭 Notebook 标签页不一定结束 kernel；结束学习时在 Jupyter 中关闭 kernel，再回到启动它的终端按 Ctrl+C 关闭服务并按提示确认，释放内存。首次运行会按锁文件安装依赖；以后复用本地环境。

| 项目 | 你在解决什么问题 | 方法/工具 | 第一遍运行地点 |
|---|---|---|---|
| 1 | 机械臂如何模仿专家？偏离演示后怎么修复？ | MuJoCo Reacher + BC + 单轮 DAgger 式纠正 | Mac CPU |
| 2 | 没有动作标签，怎样从奖励学控制？ | Gymnasium Pendulum + SB3 SAC/PPO | Mac CPU |
| 3 | 先学环境如何变化，再规划动作，能否控制系统？ | 神经动力学模型 + CEM/MPC | Mac CPU |
| 4 | 看图像预测一段动作，比逐步回归有什么不同？ | LeRobot PushT + ACT / Diffusion Policy | Mac 小试；完整训练按需 CUDA |
| 5 | 视觉、语言、状态如何共同驱动动作？ | LeRobot SmolVLA + LIBERO | Linux + NVIDIA GPU |

1. 文件位置与启动

整个文件夹一起使用；脚本会自动把结果写入本文件夹的 `runs/`，每次运行生成新目录，不覆盖旧实验。不要只复制单个 `.py`，因为脚本会导入 `common.py`。

若使用交付时的本地位置：

```bash
cd robot_learning_loops
uv sync --locked
uv run python loop01_imitation.py --quick
```

也可以把整个 `robot_learning_loops` 文件夹复制到自己的 Developer 目录，再在它里面打开 Terminal 执行后两条命令。`uv.lock` 已记录此次验证的依赖版本。已有本项目环境时，`uv sync --locked` 不会要求重新安装全部软件。

此环境独立于之前的 physical-ai-lab，不修改其依赖。默认无摄像头、无机器人硬件、无云资源、无公开上传；前 3 项除首次安装依赖之外均可离线执行。

2. 先做一项，不用同时启动

```bash
# 模仿学习：快速完整链路检查
uv run python loop01_imitation.py --quick
# 更多数据、更新步数与评测 episodes
uv run python loop01_imitation.py

# 强化学习：10k 是练习预算，不保证收敛
uv run python loop02_reinforcement.py --algo sac --steps 10000
# 对照算法：相同环境步数不等于相同计算量
uv run python loop02_reinforcement.py --algo ppo --steps 10000

# 学习动力学后用 MPC 执行动作
uv run python loop03_world_model.py --quick
uv run python loop03_world_model.py

# 列出并检查前沿复现计划；默认只打印，不下载或训练
uv run python frontier.py --track diffusion --stage train
uv run python frontier.py --track smolvla --stage train
```

`--quick` 是少数据/少更新的完整链路演示，不是论文 benchmark。默认小 MLP 在 CPU 上运行，避免 GPU 小算子开销。第 1 项提供 `--device mps`，有需要再比较。

3. 每次实验结束必须打开的文件

| 文件 | 类比你的 quant toolbox | 怎么读 |
|---|---|---|
| raw_*.npz | 原始行情与标签 | 先打印字段、维度、episode、时间，再建模 |
| manifest.json | 代码/数据/回测配置版本 | 确認 seed、依赖、参数，避免无法复现 |
| training.csv 或 training/progress.csv | 拟合与调参曲线 | 用于诊断，不能直接替代策略表现 |
| policy/checkpoint | 冻结的 signal 或模型 | 重新加载后再评测 |
| episodes.csv | 每一笔交易或每个独立回测窗口 | 不只看均值，要看失败分布 |
| comparison.png | 绩效对比图 | 与零动作/随机/专家等基线比较 |
| paired_deltas.csv | 同一初始条件下的相对收益 | 配对差值更利于识别改进 |
| *.gif | 执行回放 | 从真实模拟状态绘制的示意动画，不是写死的演示 |

4. 学习方式

每个项目采用三遍：第一遍不改算法，完成闭环并解释各文件；第二遍只改一个变量，预测结果后运行；第三遍提出失败假设，做有对照的修复。每遍都有一页实验记录，模板在 `lessons/实验记录模板.md`。

项目 1 的 `notebooks/01_机械臂模仿学习.ipynb` 是学习主入口，把数据、训练与评测拆成独立 cell。可用 VS Code 的 notebook 功能，并选择本目录 `.venv/bin/python`。如果提示缺少 ipykernel，执行 `uv sync --locked --group notebooks`。如果使用 JupyterLab，随后 `uv run --locked --group notebooks jupyter lab`。代码脚本便于学完后批量跑实验；Notebook 中保留可直接理解和修改的训练循环，复用脚本中的模拟器、专家与策略接口。

5. 研究诚实边界

- Reacher 是二维、无接触、低维状态控制；它不能证明视觉抓取、复杂接触或真机能力。
- 第 1、3 项是基于公开环境与经典方法的教学实现；第 2 项实际调用 SB3；它们均不声称复现论文分数。
- 第 4、5 项使用 LeRobot 原始实现，但缩小预算的配置仍是机制练习。严格复现需恢复原论文模型、数据、训练预算、种子与评测协议。
- 置信区间目前是一个训练 seed 内的 episode bootstrap。要比较学习算法，要再运行至少 3 个训练 seeds 并按训练 seed/episode 层级统计。
- 当前评测集是开发基准；看过结果再改算法之后，它不再是未触碰的最终测试集。最终报告必须另外冻结未使用的初始条件。
- 自行定义的成功阈值都标明，不与官方 leaderboard 混比。
- 练习脚本不会自动发起 PR、上传模型、公开数据或启动 AWS。外部贡献先查重、建立最小复现，再送维护者评审。


## GitHub 版本说明

本目录保留完整历史运行快照于 `runs/`，包括小型原始轨迹与 checkpoint；新增运行仍被 Git 忽略。只含 manifest 的目录是未完成运行，不能视为验证通过。`example_results/` 是更容易浏览的结果摘录。

Notebook 发布副本保留从新 kernel 顺序执行的 16 个代码格输出；本地交互工作副本不被改动。展示输出中的个人绝对路径改为相对路径，数值和图像保持不变。HTML 用同一份发布副本重新导出。源实验 manifest 保留原始校验值；发布转换记录见仓库根目录 `发布范围.md`。

历史 SAC replay buffer 是本练习生成的 pickle，仅作实验产物存档。Notebook 的模型权重使用 `weights_only=True` 加载；不需要加载 replay buffer 即可完成第一课。
