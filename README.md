# robotics2026 · Physical AI 训战留痕

《Physical AI 训战合一计划》的实操留痕区。目标身份：**Physical AI 数据与评测专家**——
能亲手理解/采集 episode、量化数据质量与边际价值、把 数据→训练→评测→部署 串成闭环。

本仓库有**两条互补的训练线**，共用一套留痕纪律。本 README 是总导航与整合学习路径。

## 仓库地图

```
robotics2026/
├── day0/                    # 新机自检与安装
│   ├── (Linux/通用版)       #   check_env.sh + 分层冒烟测试 + exercises 速览
│   └── chatgpt/             #   Mac (Apple Silicon) 版：新机起步.md + mac_selfcheck.sh + 时间语义练习
├── robot_learning_loops/    # 【A线·算法直觉】Mac CPU 可跑的完整闭环阶梯（已实测验证）
│   ├── loop01_imitation.py  #   BC + DAgger 式纠正（✅ 已跑通，见 验证记录.md）
│   ├── loop02_reinforcement.py  # SAC/PPO（✅ 链路已验证）
│   ├── loop03_world_model.py    # 学习动力学 + CEM/MPC（✅ 已跑通）
│   ├── frontier.py          #   项目4/5：ACT/Diffusion、SmolVLA 复现计划生成器（未训练验证）
│   ├── notebooks/01_机械臂模仿学习.ipynb   # 第一课主入口（30 cells，含已执行输出）
│   ├── lessons/             #   00_闭环地图 → 记录模板（先读 00）
│   └── runs/ example_results/  # 真实实验产出：manifest/episodes.csv/对照图/GIF
├── notebooks/               # 【B线·数据与评测】NB00–NB08 递进闭环系列
│   ├── nbutils.py           #   结果账本（append-only）、Wilson CI、数据集兼容加载
│   └── README.md            #   递进地图、运行纪律、算力预算
├── weekNN/                  # 每周留痕：NOTES.md + oral-exam.md（允许粗糙，禁止事后美化）
└── downloads/               # 便携压缩包（发布范围见 发布范围.md）
```

## 两条线为什么互补（不要二选一，也不要并行乱跑）

| | A线 robot_learning_loops | B线 notebooks NB00–NB08 |
|---|---|---|
| 练什么 | **算法直觉**：BC→RL→世界模型→ACT/Diffusion→VLA，亲手训练与执行策略 | **数据与评测**：解剖、统计可信度、消融定价、失败归因、质量分校准 |
| 算力 | Mac CPU 即可（前 3 项，分钟级） | NB02/04/06/08 需 GPU（云 4090 ≈ ¥2/h） |
| 学法 | 三遍法：跑通→只改一个变量→假设+对照修复 | 账本法：跑→落盘→分析→复盘，`results/` 只增不改 |
| 角色 | 给你「我训过、我执行过」的资格 | 给你「我定标准、我会定价」的差异化 |

A线证明你能做出东西，B线证明你能衡量东西。评测专家没有 A 线会被一句「你自己训过吗」问穿；
只有 A 线则是又一个「学过机器人的 ML 人」。

**三个交汇点（设计好的，不是巧合）：**

1. **A线项目4 ≈ B线NB02**：都是 LeRobot PushT 上训 ACT/Diffusion。从 A 线爬完前三级再做，
   一次投入两线记账。
2. **NB03（评测统计学）反哺 A 线**：NB03 产出你的《评测协议 v1》（n、seed、Wilson CI、A/B 判定规则），
   立刻应用到 A 线所有 `episodes.csv`——A 线当前的区间是单 seed bootstrap，NB03 告诉你它能支持什么结论。
3. **《新机起步》的首个研究问题 ≈ NB06 的具体化**：「观测延迟/训练配对错位 vs 数据干预对成功率的影响」
   正是 预测→干预→实测 的题材，且直指你的差异化方向（时序、point-in-time、数据价值）。

## 整合学习路径（按阶段推进，同一时间只推进一个项目）

### Phase 0 · 环境（大部分已完成）

当前三门已执行的 A 线课程入口：

| 入口 | 内容 |
|---|---|
| [第一课 Notebook](robot_learning_loops/notebooks/01_机械臂模仿学习.ipynb) | 30 个单元格：采集、审计、BC 训练、执行、DAgger 式纠正、对照、复盘；保留已执行输出 |
| [第二课：强化学习](robot_learning_loops/notebooks/02_强化学习与经验回放.ipynb) | SAC 预算对照、经验回放、动作缩放、时间截断和压力评测；已执行 |
| [第三课：世界模型与规划](robot_learning_loops/notebooks/03_世界模型与规划迭代.ipynb) | 动力学学习、MPC 执行、新数据采集和对照重训；已执行 |
| [三课学习顺序](robot_learning_loops/课程指南.md) | 每课应交付的复盘、第二遍练习与原有 NB00–NB08 的衔接 |
| [HTML 预览](robot_learning_loops/notebooks/01_机械臂模仿学习.html) | 下载到本地用浏览器打开，查看完整表格、曲线与 GIF |
| [参考复盘](robot_learning_loops/notebooks/参考复盘.md) | 先写自己的判断，再对照讲评 |
| [学习路线](robot_learning_loops/lessons/00_闭环地图.md) | 模仿学习 → 强化学习 → 世界模型/MPC → ACT/Diffusion → VLA |
| [结果与验证](robot_learning_loops/验证记录.md) | 已跑通的范围、实际数据与尚未验证的前沿项目 |
| [Mac 起步工具](day0/chatgpt/新机起步.md) | 新机自检、Torch smoke test、时间语义练习与安装指南 |
| [轻量下载包](downloads/) | 整套练习和起步工具的便携压缩包 |

- Mac：`day0/chatgpt/新机起步.md` + `mac_selfcheck.sh`；Linux/云：`day0/check_env.sh`。
- ✅ A线 loop01（BC+DAgger）已于 2026-09 跑通并验证（正常反馈误差 ~4.17cm → 纠正后 ~1.37cm）。

### Phase 1 · 本地筑基（约 2 周，全部 Mac 本地，零 GPU 成本）

| 顺序 | 做什么 | 完成标准 |
|---|---|---|
| 1 | A线 loop02（SAC/PPO）三遍法 | 三页实验记录；能解释 SAC 10k 步为何不保证收敛、PPO/SAC 同步数≠同算力 |
| 2 | A线 loop03（动力学+MPC）三遍法 | 三页实验记录；能解释 multistep error 与控制表现为何不必然相关 |
| 3 | B线 NB03（评测统计学） | 《评测协议 v1》落盘；**回头用它重新审视 A 线全部结论**，站不住的写进复盘 |
| 4 | B线 NB01（数据集解剖） | pusht 五问书面回答；这是 Dataset Profiler 的需求文档 |

Phase 1 出口检验（口试自测）：BC 为什么会 compounding error？没有动作标签怎么学？
预测准 ≠ 控制好，为什么？n=10 的对比表能说明什么？

### Phase 2 · 交汇与定价（约 3 周，开始用云 GPU）

| 顺序 | 做什么 | 完成标准 |
|---|---|---|
| 5 | A线项目4 / B线NB02（同一件事）：LeRobot PushT 训 ACT 或 Diffusion | success rate 达官方参考 ±10%，或有完整排查记录；成本账入账本 |
| 6 | B线 NB04（数据消融） | Success=f(N) 曲线（每点≥2 seeds + CI）；说出「每 25 条 episode 值几个百分点」 |
| 7 | 首个研究问题（《新机起步》§1 = NB06 方法论）：延迟/配对错位 vs 数据干预 | 预测先落盘→干预→实测→差值解释；一页双语报告 |

Phase 2 的产出直接是《数据资产审计》的实证脚注和 networking 弹药（Drive 计划 Week 2–4 的「战」线）。

### Phase 3 · 立标准（约 3 周+）

| 顺序 | 做什么 | 完成标准 |
|---|---|---|
| 8 | B线 NB05（失败归因）+ NB07（benchmark 迁移） | 失败分布 + 「20 条新数据采什么」；两 benchmark 含金量换算表 |
| 9 | B线 NB08（EQS Capstone） | 质量分 v0 + top/random/bottom 校准实验；写成 `EQS_v0_spec.md` 公开 |
| 10 | A线项目5（SmolVLA + LIBERO，Linux+GPU） | 达到「能对话、能评测」深度即可，不追训练 SOTA |

Phase 3 出口 = Observatory v0 种子齐备：Profiler(NB01)、协议(NB03)、缩放实验(NB04)、
失败税则(NB05)、EQS(NB08) 全部有亲手数据。

## 快速开始

```bash
# A线第一课（Mac 本地，需已安装 uv）
cd robot_learning_loops
uv sync --locked --group notebooks
uv run --locked --group notebooks jupyter lab notebooks/01_机械臂模仿学习.ipynb

# B线（先自检，再按 notebooks/README.md 的顺序）
cd day0 && bash check_env.sh        # Mac 用 day0/chatgpt/mac_selfcheck.sh
```

## 统一纪律（两线通用）

1. **一坐一环**：一次只推进一个项目的一遍；跑到一半不复盘就离开 = 作废重跑。
2. **账本只增不改**：B线 `notebooks/results/` 与 A线 `runs/` 同理——失败记录是留痕的一部分。
3. **先预测后运行**：任何对照实验，预测先落盘（带时间戳）再动手。
4. **区间不缺席**：所有公开数字带 CI 与 n；协议以 NB03 为准，改协议须书面说明并注明生效时间。
5. **诚实边界**：教学仿真结果不外推真机与论文 benchmark（A线 `验证记录.md` 的写法是模板，保持这个标准）。
6. **每周五一页 thesis**：从当周账本提炼，同步 GitHub + 对外渠道（对应《02 开源蓝图》节拍）。

## 与《Physical AI 训战合一计划》12 周课表的映射

| Drive 课表 | 本仓库对应 |
|---|---|
| W1 物理素养 | lessons/00_闭环地图 + loop01；坐标变换 notebook 进 weekNN/ |
| W2 Robot Data | NB01 + Profiler 需求（→ 独立仓库孵化） |
| W3 模仿学习 | loop01 三遍法 + NB02 + NB04 首条曲线 |
| W4 VLA 史 | frontier.py 的 ACT/Diffusion 计划 + 论文精读（weekNN/ 留痕） |
| W5 失败归因 | NB05 |
| W6 数据质量（专设） | NB08 EQS |
| W7 世界模型 | loop03 三遍法 + WALL-WM 精读 |
| W8 仿真与真机 | NB07 + SO-101 闭环（so101-loop 独立仓库） |
| W9–12 Capstone | NB08 校准实验 + Observatory v0 迁出 |

成熟产出的去处：`physical-ai-data-observatory`（旗舰）、`physical-ai-notes`（周留痕）、
`robot-data-audits`（审计）、`so101-loop`（真机）。本仓库是全部内容的孵化处。

## 约定

- commit message 英文、动词开头、说清 why；每周 ≥4 天有 commit。
- weekNN/ 放 `NOTES.md`、`oral-exam.md`、当周代码；发布范围见 `发布范围.md`。
- 不提交：`.venv`、大体积新 runs（历史验证快照除外）、任何凭证与 `.env`。
