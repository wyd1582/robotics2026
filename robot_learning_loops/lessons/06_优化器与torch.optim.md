# 06 · 优化器与 torch.optim：为什么是 Adam，以及怎么换着跑

> 用法：第 1–4 节复制进 notebook 当一个 Markdown 格；第 5 节的代码复制成一个 Code 格，
> 放在 BC 训练格（定义了 `Actor` / `train` / `val` / `UPDATES` / `SEED`）之后运行。

## 1. 优化器在循环里的位置

```
loss.backward()    # autograd 算出每个参数的梯度 p.grad
optimizer.step()   # 优化器决定"拿梯度怎么更新参数"——它只是一条更新规则
```

模型、损失、优化器三者解耦：换优化器不改模型和损失，这就是可以做对照实验的原因。

## 2. 一条演化链记住整个家族

每一代都在修上一代的一个具体毛病：

| 优化器 | 更新规则（直觉版） | 修了什么毛病 |
|---|---|---|
| SGD | `p -= lr * grad` | —（基线） |
| +Momentum | 用**梯度的 EMA** 代替当期梯度 | 单批梯度噪声大、峡谷地形来回震荡 |
| RMSprop | 步长除以**梯度平方的 EMA 开根** | 各参数尺度不同，统一 lr 顾此失彼 |
| **Adam** | Momentum + RMSprop 二合一（带偏差修正） | 同时要方向平滑和逐参数自适应步长 |
| AdamW | Adam + **解耦的** weight decay | Adam 里 L2 正则被自适应分母扭曲 |

量化翻译：momentum 是对梯度做 EMA 平滑（信号去噪）；自适应分母是按各参数梯度的
"波动率"缩放步长——**Adam ≈ 对每个参数做 vol-targeting 的动量策略**。
记住这个类比，整个家族就不用背了。

## 3. 选用逻辑（不是玄学，是场景匹配）

- **小 MLP + 回归/BC（本课场景）→ Adam, lr=1e-3**：万金油，对 lr 不敏感，不用调参就能到位。
  数据小、训几分钟，优化器带来的"泛化差异"根本轮不到登场，快速稳定收敛就是全部需求。
- **大型视觉网络长训 → SGD+Momentum（+lr 调度）**：调好后终点泛化常略优于 Adam，代价是 lr 很敏感。
- **Transformer / VLA 微调 → AdamW**：事实标准（ACT、Diffusion Policy、OpenVLA 的默认都是它）。
- **小规模、全批量、确定性问题 → L-BFGS**：二阶拟牛顿，几十步内收敛（你在 HBS 用过）；
  数据一大或有噪声 minibatch 就不适用。
- 经验法则：**先 Adam(W) 跑通拿到基线，只有当优化器本身成为怀疑对象时才换**。
  优化器几乎不修"数据错了"和"表示错了"——那两类问题永远优先排查。

## 4. torch.optim 模块速览（够用的最小集）

```python
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
opt.zero_grad(set_to_none=True)   # 清梯度（set_to_none 更快，PyTorch 2.x 已是默认）
loss.backward(); opt.step()
opt.param_groups[0]["lr"]         # 运行中读/改超参的入口（分组可对不同层用不同 lr）
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000)  # 调度器包在外层，每步 sched.step()
```

特例：`LBFGS.step(closure)` 需要传入一个"重算 loss 并 backward"的闭包，因为它一步内要多次求值。

## 5. 对比实验（复制成一个 Code 格直接跑）

纪律：先把你对排名的预测写进 `PREDICTION`，跑完对照——错了才有信息量。

```python
import copy, time
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt

PREDICTION = "我预测收敛速度排名: adam ≈ adamw > rmsprop > sgd_mom > sgd; 最终val差距: 很小"

OPTIMIZERS = {
    "sgd":      lambda p: torch.optim.SGD(p, lr=1e-2),                    # 注意 lr 比 Adam 大一个量级
    "sgd_mom":  lambda p: torch.optim.SGD(p, lr=1e-2, momentum=0.9),
    "rmsprop":  lambda p: torch.optim.RMSprop(p, lr=1e-3),
    "adam":     lambda p: torch.optim.Adam(p, lr=1e-3),
    "adamw":    lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=1e-2),
}

def train_with(opt_factory, updates=UPDATES, seed=SEED):
    model = Actor(train["obs"].mean(0), np.maximum(train["obs"].std(0), .01),
                  train["expert_action"].mean(0), np.maximum(train["expert_action"].std(0), .01))
    x  = torch.tensor(train["obs"], dtype=torch.float32)
    y  = torch.tensor(train["expert_action"], dtype=torch.float32)
    vx = torch.tensor(val["obs"], dtype=torch.float32)
    vy = torch.tensor(val["expert_action"], dtype=torch.float32)
    opt = opt_factory(model.parameters())
    rng = np.random.default_rng(seed)          # 同一 seed → 同一批次序列，公平对照
    curve = []
    t0 = time.time()
    for step in range(updates):
        idx = torch.as_tensor(rng.integers(0, len(x), size=256))
        pred = model.standardized(x[idx])
        target = (y[idx] - model.a_mean) / model.a_std
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        opt.step()
        if step % max(1, updates // 20) == 0 or step == updates - 1:
            with torch.inference_mode():
                vmse = ((model(vx) - vy) ** 2).mean().item()
            curve.append(dict(step=step + 1, val_mse=vmse))
    return pd.DataFrame(curve).assign(seconds=round(time.time() - t0, 1)), model

results = {name: train_with(f) for name, f in OPTIMIZERS.items()}

fig, ax = plt.subplots(figsize=(8, 4))
for name, (curve, _) in results.items():
    ax.plot(curve.step, curve.val_mse, label=name)
ax.set(xlabel="updates", ylabel="val action MSE", yscale="log", title="Optimizer comparison (same seed, same batches)")
ax.legend(); plt.tight_layout(); plt.show()

summary = pd.DataFrame({n: {"final_val_mse": c.val_mse.iloc[-1], "seconds": c.seconds.iloc[0]}
                        for n, (c, _) in results.items()}).T.sort_values("final_val_mse")
display(summary)
print(PREDICTION)
```

可选加餐——L-BFGS（全批量 + closure，体验二阶方法在小问题上的暴力收敛）：

```python
model = Actor(train["obs"].mean(0), np.maximum(train["obs"].std(0), .01),
              train["expert_action"].mean(0), np.maximum(train["expert_action"].std(0), .01))
x = torch.tensor(train["obs"], dtype=torch.float32)
t = (torch.tensor(train["expert_action"], dtype=torch.float32) - model.a_mean) / model.a_std
opt = torch.optim.LBFGS(model.parameters(), max_iter=100, line_search_fn="strong_wolfe")

def closure():
    opt.zero_grad(set_to_none=True)
    loss = ((model.standardized(x) - t) ** 2).mean()
    loss.backward()
    return loss

opt.step(closure)   # 一次 step = 内部最多 100 次迭代
with torch.inference_mode():
    vx = torch.tensor(val["obs"], dtype=torch.float32)
    vy = torch.tensor(val["expert_action"], dtype=torch.float32)
    print("LBFGS val MSE:", ((model(vx) - vy) ** 2).mean().item())
```

## 6. 判读要点（跑完回答，写进实验记录）

1. 曲线的差异主要在**前 20% 步数**还是终点？——小问题上优化器决定的是速度，不是天花板。
2. 把 sgd 的 lr 改成 1e-3 再跑一次：它慢多少？——体会"自适应步长"到底省了什么。
3. adamw 和 adam 终点几乎一样？——正则在 1.8 万参数 + 充足数据下无事可做；想看出差别，把训练集截到 1/10 再比。
4. 任何一个优化器的 val MSE 优势，去 rollout 里验证过之前，都只是离线故事——第一课的主教训在这里同样成立。
