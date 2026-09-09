# 练习闭环：六个可验证的小闭环（对应课表 Week 2–8）

原则：每个练习都必须走完「输入 → 操作 → 可量化产出 → 一段书面结论」，
缺任何一环不算完成。全部产出进 `physical-ai-notes`，可复用部分并入 observatory。

## 练习 0 · 数据集解剖（半天）→ 对应 Week 2

- 做什么：跑 `smoke/01_lerobot_dataset.py`，然后不借助文档，仅靠代码和数据本身回答：
  action 的物理含义与坐标系、success 如何定义、fps 与控制频率是否一致、有无缺帧。
- 产出：一页 markdown《一条 pusht episode 的完整解剖》。
- 通过标准：每个字段都能说清"是什么、谁写入的、错了会怎样"。
- 这就是 Robot Dataset Profiler v0 的需求文档来源。

## 练习 1 · 第一次训练闭环（1 天，需 GPU 或云）→ 对应 Week 3

- 做什么：用 LeRobot 官方脚本在 pusht 上训练 Diffusion Policy（或 ACT），
  然后在仿真里 rollout 50 次，报告 success rate。
- 参考命令（以 LeRobot 当前版本文档为准）：
  ```bash
  python -m lerobot.scripts.train policy=diffusion env=pusht
  python -m lerobot.scripts.eval  policy_path=outputs/... eval.n_episodes=50
  ```
- 产出：训练曲线截图 + success rate 数字 + 训练用时/显存记录。
- 通过标准：success rate 达到官方 README 报告值的 ±10% 以内。
  达不到就排查（数据、超参、评测协议），排查过程本身写下来——这是评测专家的核心肌肉。

## 练习 2 · 数据消融：第一条 Success = f(N) 曲线（1–2 天）→ 对应 Week 3/6

- 做什么：把训练集截为 25 / 50 / 100 / 全量 episodes，同一 policy、同一评测协议各训一次。
- 产出：一张 Success = f(N) 曲线图 + 每个点的方差（每个设置至少 2 个 seed）。
- 通过标准：能回答"边际 25 条 episode 值多少个百分点的成功率"。
- 这是 MDV（边际数据价值）的第一个实测数据点，直接进 observatory/experiments。

## 练习 3 · 失败归因（1 天）→ 对应 Week 5

- 做什么：把练习 1 中失败的 rollout 逐条看视频，按 taxonomy 手工标注：
  perception / policy / execution / env / 评测协议本身的问题。
- 产出：失败分布表 + 3 条最典型失败的逐帧描述。
- 通过标准：能回答"如果只能采 20 条新数据来修复，采什么场景"。

## 练习 4 · 一次干预迭代（1 天）→ 对应 Week 5/6

- 做什么：根据练习 3 的归因结论，选**一个**干预（补数据 / 改 action chunking /
  改增强 / 改评测初始分布），重训重评。
- 产出：一页《干预 → 预测 → 实测》报告：干预前先写下预测的 Δsuccess，再对比实测。
- 通过标准：不在乎涨没涨，在乎"预测与实测的差"有没有解释。这是从调参侠到科学家的分界线。

## 练习 5 · 陌生 benchmark 迁移（1 天）→ 对应 Week 8

- 做什么：在 LIBERO 或 Meta-World 上重复练习 1 的最小版本（一个任务即可）。
- 产出：两个 benchmark 评测协议的差异清单（初始状态分布、success 判定、最大步数、随机性来源）。
- 通过标准：能说清"同一个 80% success，在两个 benchmark 上含金量差多少"。

---

## 完成后的状态自检

六个练习全绿 = 你已经单人闭环过一遍 数据 → 训练 → 评测 → 归因 → 迭代。
此时（也只有此时）开始真机 SO-101 闭环和对外 networking 才有弹药。
