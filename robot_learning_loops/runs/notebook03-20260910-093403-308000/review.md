# 世界模型与规划迭代复盘

最初预测：待填写：我的预测是……因为……

实际结果：
```text
                                 return  upright_fraction
policy       condition
base_mpc     action_repeat_3  -139.1591            0.8425
known_mpc    action_repeat_3  -139.1576            0.8419
more_random  action_repeat_3  -139.2924            0.8481
more_updates action_repeat_3  -139.3071            0.8456
planner_data action_repeat_3  -138.9644            0.8469
random       action_repeat_3 -1203.0332            0.0106
```

## 本课学习的函数是什么？

待填写：明确输入、标签、输出，以及已知的成本函数。

## CEM 与 MPC 分别负责什么？

待填写：推演、搜索、执行和重规划各在哪里？

## 新增数据有没有比多训练更好？

待填写：引用配对差值与区间。

## 离线模型误差与控制回报是否同序？

待填写：用本次结果解释，不假设必然相关。

## 下一轮只改什么？

待填写：时域/搜索预算/模型集成/数据之一；写出判定标准。

边界：一个训练 seed；开发评测；低维 Pendulum 仿真。
