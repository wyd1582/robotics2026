项目 5：视觉、语言、状态共同生成动作

问题：同样一张桌面图像，“把杯子放到盘子上”和“把杯子放到盘子旁”应产生不同动作。只有识别物体或生成语言答案不算完成，需要动作真的改变场景并满足任务。

量化类比：使用预训练语言/视觉模型提取非结构化信息，再接到实际决策。但VLA可以把条件表示和动作生成联合训练，不能把“LLM理解了指令”当作机器人已执行成功。

这一阶段采用LeRobot SmolVLA与LIBERO。具体路线遵循已核对官方示例：载入预训练VLM，训练SmolVLA动作专家；这不等同于直接对完整smolvla_base动作checkpoint做微调。完整checkpoint微调是后续对照，需要核对相机key、状态维度、动作语义与normalization。

当前LeRobot的LIBERO支持要求Linux，因此这项放Linux/NVIDIA主机。Mac仍可读材料、检查metadata和生成计划。本交付不启动云实例，不下载VLA权重或执行此训练。

版本：LeRobot 2774d9bddcbbda50e697e162e89e7eaada8d7105；lerobot/libero a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4。上游VLM权重由其配置解析，本脚本尚未独立锁定全部基础权重revision，严格复现需记录实际下载的快照哈希。

闭环：公开机器人演示 → 检查图像、语言、状态和动作 → 训练/验证按episode分离 → 训练动作专家 → 保存配置与预处理器 → 在相应LIBERO任务中执行 → 改指令/物体/空间布局 → 统计成功率与失效类型。

在Linux主机上的顺序：

```bash
uv run python frontier.py --track smolvla --stage prepare
uv run python frontier.py --track smolvla --stage prepare --execute
uv run python frontier.py --track smolvla --stage inspect --execute
uv run python frontier.py --track smolvla --stage train --device cuda --steps 100 --batch-size 4 --execute
```

100步用于检查，不代表训练好。先检查显存、视频解码、一个batch和checkpoint，再按预算增长训练。完整训练预算与预训练/微调方式需事先选定，不在第一次实验中默默启动100k步。

评价从一个task开始，脚本默认LIBERO-Spatial的task 0、固定initial states、hard reset、relative控制模式。训练数据包括更多任务，单task评价只能支持该task上的结论。动作是6D末端增量+夹爪，不能拿SO101关节位置动作直接接入。

```bash
uv run python frontier.py --track smolvla --stage eval --device cuda --checkpoint /实际路径/pretrained_model --eval-episodes 3 --execute
```

这仍是链路验收。扩到多个tasks、多个训练seeds、更多初始状态前，先核查相机key/尺寸、训练动作与执行动作的坐标系/单位/控制模式、normalization、语言字符串，以及环境帧率。基于训练中出现过的任务评测，不应写成zero-shot新任务泛化。

第一遍只学三个点：预训练VLM提供什么、action expert学什么、动作chunk如何进入机器人控制器。VLA不一定经过WBC；机械臂通常通过相应的关节或末端控制器执行。

第二遍试题：

- 在同一场景中改目标物体词，检查动作是否随指令改变；不能只观察文本embedding。
- 固定动作expert，冻结/开放部分视觉表征，比较数据预算与计算成本。
- 使用相同任务与数据划分比较ACT和SmolVLA；分别报告参数量、初始化、训练时长与rollout成功。
- 保留原始失败视频，区分语言理解、视觉定位、动作序列、接触执行、接口错配；不要只有一个总成功率。

第三遍贡献题：让失败taxonomy与代码日志对应；建立语言反事实测试；重现一类相机/动作语义错配；在等标签预算下比较失败驱动补数据与随机补数据。

以后接真机时，单独完成相机/手眼或本体标定、动作范围、时间与坐标一致性、低速空载执行和人工停止流程。先在一个桌面任务闭合teleop→数据→训练→部署→再采集。没有同本体/同接口验证，不把仿真成功外推到真机。

阅读：[SmolVLA](https://huggingface.co/docs/lerobot/smolvla)、[SmolVLA论文](https://arxiv.org/abs/2506.01844)、[LIBERO训练/评测](https://huggingface.co/docs/lerobot/libero)、[LIBERO项目](https://libero-project.github.io/)、[SO101接入](https://huggingface.co/docs/lerobot/so101)。官方数据卡/论文的具体性能不是本练习已复现的结果。
