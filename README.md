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
