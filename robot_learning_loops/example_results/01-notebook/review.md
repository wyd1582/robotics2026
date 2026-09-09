# 我的第一次机器人学习闭环

配置：{'seed': 7, 'device': 'cpu', 'quick': False, 'episodes': 80, 'updates': 2000, 'eval_episodes': 30, 'extra_episodes': 40}

最初预测：待填写：我预测……因为……

实际执行指标：
```text
                        episodes  success_rate  successes  mean_distance_cm
policy       condition
bc           delay_0          30        0.6000         18            4.1674
             delay_2          30        0.0000          0           16.9981
dagger       delay_0          30        0.8667         26            1.3679
             delay_2          30        0.1333          4           12.6074
more_expert  delay_0          30        0.7333         22            3.3940
             delay_2          30        0.0333          1           15.4914
more_updates delay_0          30        0.6667         20            2.5076
             delay_2          30        0.0667          2           15.8692
random       delay_0          30        0.0333          1           17.8388
teacher      delay_0          30        1.0000         30            0.0534
zero         delay_0          30        0.0000          0           24.9292
```

配对比较（cm）：
```text
candidate    reference condition  delta_cm  ci_low_cm  ci_high_cm  paired_episodes
   dagger           bc   delay_0   -2.7996    -4.6703     -1.2684               30
   dagger more_updates   delay_0   -1.1397    -1.9886     -0.3502               30
   dagger  more_expert   delay_0   -2.0261    -4.1043     -0.3945               30
```

## 离线误差与执行表现

待填写：用本次两项数值解释它们是否一致。

## 公平对照的结论

待填写：dagger 相对 more_updates / more_expert 的差值与区间说明什么？

## 最坏案例观察

待填写：我看到的动作现象是……还不能确定的原因是……

## 延迟测试

待填写：40 ms 延迟带来了什么变化？

## 下一轮假设

待填写：我只改……预期……用……判断；其余参数冻结。

边界：一个训练 seed；开发评测；MuJoCo 低维无接触任务；无真机验证。
