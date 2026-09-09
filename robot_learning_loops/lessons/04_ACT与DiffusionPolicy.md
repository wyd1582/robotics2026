项目 4：从低维单步回归，进入视觉动作序列模型

前沿连接点：ACT、Diffusion Policy 都是具身学习中有代表性的动作序列建模方法，适合作为理解更新VLA方法的基础。本项目复现其开源实现与训练/执行工作流，不声称它们是2026最新方法，也不承诺小预算复现论文分数。

任务用PushT：通过移动一个圆形推手，将T形物体推到目标姿态。输入包括图像和推手位置，动作是二维目标位置；不是关节力矩。这一步首次引入视觉与接触。

量化类比：把结构化特征策略升级为含图像/文本编码器的模型，再从单步信号升级为连续执行计划。需要注意这里的动作会改变物体姿态，离线预测准确并不保证连续推到目标。

两种建模方式：ACT一次产生一段动作，学习演示中的协调时序；Diffusion Policy通过条件去噪生成动作序列，能够表达多种可行行为。训练动作块可以包含未来动作标签，输入观测不能包含决策时尚不可用的未来信息。

一份动作块要区分三种长度：观察窗口、预测长度、实际执行长度。预测16步不意味着必须连续执行16步而不看新图像。改变执行长度通常也改变延迟/反应能力，不能只归因为模型架构。

`frontier.py` 已按以下实际读取的版本生成命令：

- LeRobot commit：2774d9bddcbbda50e697e162e89e7eaada8d7105
- lerobot/pusht revision：7628202a2180972f291ba1bc6723834921e72c19
- 软件依赖与算法代码来自上游；训练/评测尚未在这次交付中执行。

分阶段操作：

```bash
# 默认只是打印计划，不安装、不训练
uv run python frontier.py --track diffusion --stage prepare
# 真正准备时再执行；在external/lerobot建立独立上游环境
uv run python frontier.py --track diffusion --stage prepare --execute
# 只读取固定版本的metadata，不下载全部视频或模型权重
uv run python frontier.py --track diffusion --stage inspect --execute
# 先100步/4个batch样本验证数据、前向、反向、保存
uv run python frontier.py --track diffusion --stage train --device mps --steps 100 --batch-size 4 --execute
```

第一次训练可能下载视频、模型视觉骨干及其依赖；先读打印出的命令和metadata。选择pyav作为视频读取后端，减少独立系统ffmpeg链接问题，但是否兼容仍需实际解码验收。Mac遇到具体MPS算子失败，先用CPU做单batch排查，再决定转Linux/CUDA，不隐藏CPU fallback。

训练命令固定数据revision，按task保留10%episodes做离线验证；实际任务只有一个，所以不是跨任务泛化。确认该版本的拆分清单及归一化统计是否仅来自训练集：若上游使用全数据元统计，需标明局限，并在严格数据实验前改用训练集统计。

脚本输出独立run目录。找到其中实际存在的 `pretrained_model` 目录，它应包含config.json。将下面路径替换为它，不要把文字占位符直接复制执行：

```bash
uv run python frontier.py --track diffusion --stage eval --device mps --checkpoint /实际路径/pretrained_model --eval-episodes 3 --execute
```

3次rollout只验收链路。通过后再做较长训练和更多配对rollout。用于论文复现的steps、batch、归一化、EMA权重、图像resize与环境版本必须来自对应论文/上游实验配置；本脚本的默认设置只用于小实验。

ACT对照使用相同数据revision与episode划分：

```bash
uv run python frontier.py --track act --stage train --device mps --steps 100 --batch-size 4 --execute
```

不能仅以相同steps说公平：还需要报告有效训练样本数、动作chunk长度、骨干初始化、计算量和训练时长。前两遍先各自跑通，不急着做排行榜。

必须产出：数据卡摘要、一个batch的图像与动作窗口、训练/验证曲线、保存的policy与normalizer、同一初始条件的执行视频、成功/覆盖指标、推理延迟。检查评测时输入的图像key、尺寸、单位是否与训练一致，不能凭shape相同就判定接口匹配。

改动题：固定执行长度，改变预测长度；固定模型改变演示episode数量；给图像/状态分别加延迟；保留失败恢复片段和删掉恢复片段的对照。一次只选一个轴。

贡献候选：episode边界的动作chunk/padding回归测试；可复现的时延曲线；训练/评测图像键或归一化错配的最小复现；改进dataset profiler对这些语义的说明。先读现有tests/issues，避免重复已有功能。

阅读：[ACT官方文档](https://huggingface.co/docs/lerobot/act)、[ACT论文](https://arxiv.org/abs/2304.13705)、[Diffusion Policy论文与代码](https://diffusion-policy.cs.columbia.edu/)、[LeRobot训练例子](https://github.com/huggingface/lerobot/blob/2774d9bddcbbda50e697e162e89e7eaada8d7105/examples/training/train_policy.py)、[gym-pusht](https://github.com/huggingface/gym-pusht)。只需先看模型输入输出、训练目标、rollout执行策略，再按实验需要补读细节。
