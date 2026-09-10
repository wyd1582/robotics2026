"""Rebuild lesson sources. Execution and HTML export are separate explicit steps."""
import argparse
import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--lessons", nargs="+", choices=["02", "03"], default=["02", "03"])
selected = parser.parse_args().lessons


def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))


def save(name):
    if name[:2] not in selected:
        cells.clear()
        return
    nb = nbf.v4.new_notebook(cells=list(cells), metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3 (robot learning)", "language": "python"},
        "language_info": {"name": "python", "version": "3.12"}})
    nbf.validate(nb)
    target = ROOT / "notebooks" / name
    nbf.write(nb, target)
    print(name, len(cells), "cells")
    cells.clear()


md(r'''
# 02 · 没有动作标签，如何从奖励学会控制？

第一课你有专家动作 $a^*$。这一课没有人告诉你每一步怎么做，环境只反馈动作之后的结果和奖励。

**研究问题：在固定交互预算下，策略是否学会把摆摆起并保持向上？训练过程中的“更好”怎样变成可核验的执行结果？**

闭环：`探索收集 transition → replay buffer → SAC 更新 → checkpoint → 独立 rollout → 失效测试 → 下一轮预算实验`。
对应 quant：先定义目标与约束，在可交互的模拟市场里优化行为，再用独立初始条件和压力场景检验。此处学习的是动作策略，不是一个静态 return 预测器。

本课交付：原始交互记录、0/2k/10k 步策略、经验回放检查、配对评测、动画和一页复盘。
默认 Mac CPU；读取低维状态；使用 Stable-Baselines3 的实际 SAC 实现。这是机制练习，不是论文分数复现。

## 使用方式

从 `robot_learning_loops/` 启动：
```bash
uv run --locked --group notebooks jupyter lab notebooks/02_强化学习与经验回放.ipynb
```
第一次不改参数，Shift+Enter 逐格运行，预留 60–90 分钟理解与记笔记。机器运行时间会记录。
改配置后重启 kernel 并从头运行；不要跳过创建模型的格子再追加训练，否则实验预算会变化。
''')
md(r'''
## 1 · 先冻结预算和问题

训练 seed 控制初始化与探索；开发评测使用另一批初始状态。评测结果看过后可用于开发，但不再叫未触碰的最终测试。
本课两个训练阶段相加是总预算：先 2,000 步，再追加 8,000 步。
**先写预测**：10k 策略会在哪种条件下最容易失效——正常、观测延迟、重力变化？
''')
code('''
SEED = 7
FIRST_STEPS = 2000
TOTAL_STEPS = 10000
N_EVAL = 16
MY_PREDICTION = "待填写：我预测……理由是……"
assert 500 < FIRST_STEPS < TOTAL_STEPS
''')
code('''
from pathlib import Path
import sys, time
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "course_utils.py").exists()), None)
if ROOT is None:
    raise RuntimeError("请保留整个 robot_learning_loops 文件夹，并在其中启动。")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from IPython.display import display, Image, Markdown
from common import plt, pendulum_frame, summarize
from loop02_reinforcement import TransitionRecorder, Progress, evaluate
from course_utils import begin_notebook, audit_transitions, show_gif, paired_return, write_review, preserve_cpu_rng
START = time.perf_counter()
OUT, manifest = begin_notebook("notebook02", "02_强化学习与经验回放.ipynb",
    dict(seed=SEED, first_steps=FIRST_STEPS, total_steps=TOTAL_STEPS, eval_episodes=N_EVAL))
display(pd.Series(manifest["versions"], name="installed_version").to_frame())
''')
md(r'''
## 2 · 环境合同：什么叫摆得好？

观测是 `[cos(theta), sin(theta), angular_velocity]`。角度 0 指向正上方；动作是 [-2,2] 的力矩，控制周期为 0.05 秒。
每次 episode 最多 200 步，即 10 秒。

环境奖励使用动作前的状态：$r_t=-(\theta_t^2+0.1\dot\theta_t^2+0.001u_t^2)$，角度先归一到 $[-\pi,\pi)$。
回报是整段奖励之和，**越高越好**，通常是负数。它没有内置的二元成功标签。

我们另记“保持向上的时间比例”：角度绝对值 < 0.2 rad 且角速度绝对值 < 1 rad/s。它是辅助、自定义指标。
''')
code('''
env = gym.make("Pendulum-v1")
obs, _ = env.reset(seed=42)
display(pd.DataFrame({"feature": ["cos(theta)", "sin(theta)", "angular_velocity"], "value": obs}))
print("dt:", env.unwrapped.dt, "action bounds:", env.action_space.low, env.action_space.high)
display(pendulum_frame(obs, "Initial state | goal: upright"))
action = np.array([.5], dtype=np.float32)
theta = np.arctan2(obs[1], obs[0])
predicted_reward = -(theta**2 + .1*obs[2]**2 + .001*action[0]**2)
nxt, reward, terminated, truncated, _ = env.step(action)
np.testing.assert_allclose(reward, predicted_reward, atol=2e-6)
print("Reward formula agrees with simulator:", round(float(reward), 5))
env.close()
''')
md(r'''
## 3 · 先建立 SAC，再理解它怎样得到训练信号

SAC 同时维护动作策略 actor、两套 Q 网络 critic 和慢更新的目标 Q 网络。Q 估计长期回报；动作策略追求较高 Q，并用熵项鼓励探索。两套 Q 与目标网络用于降低部分估计问题，不保证训练必然稳定。

这里使用 SB3 原实现，避免把实现 SAC 的工程量混进第一次问题定义。你可以直接查看实际网络、数据和更新次数。

Replay buffer 保存的是 $(o,a,r,o',terminated,truncated)$，**没有专家动作标签**。SAC 从历史交互中采样，反复更新模型；因此环境步数与梯度更新数是两个预算。
''')
code('''
training_env = TransitionRecorder(gym.make("Pendulum-v1"))
model = SAC("MlpPolicy", training_env, policy_kwargs={"net_arch": [64, 64]},
            seed=SEED, device="cpu", verbose=0, buffer_size=TOTAL_STEPS + 1000,
            learning_starts=500, batch_size=128, learning_rate=1e-3,
            train_freq=1, gradient_steps=1, gamma=.99, ent_coef="auto")
model.set_logger(configure(str(OUT / "training"), ["csv"]))
print(model.policy.actor)

def checkpoint(name):
    model.save(OUT / name)
    with preserve_cpu_rng():
        loaded = SAC.load(OUT / name, device="cpu")
    probe = np.array([1., 0., 0.], dtype=np.float32)
    np.testing.assert_allclose(model.predict(probe, deterministic=True)[0],
                               loaded.predict(probe, deterministic=True)[0], atol=0, rtol=0)
    print(name, "save/reload PASS")
    return loaded

policies = {"sac_0": checkpoint("sac_0")}
''')
md(r'''
## 4 · 第一阶段：交互数据从哪里来？

前 500 步用于随机探索，之后边采集边更新。训练时有探索，评测时使用确定性动作。
训练回报好坏还受探索策略和当时遇到的初始状态影响，不能直接当独立评测结果。
''')
code('''
stage_start = time.perf_counter()
model.learn(total_timesteps=FIRST_STEPS, callback=Progress(), log_interval=1)
print("Actual environment steps:", model.num_timesteps, "seconds:", round(time.perf_counter()-stage_start, 2))
policies[f"sac_{FIRST_STEPS}"] = checkpoint(f"sac_{FIRST_STEPS}")
training_env.save(OUT / "raw_first_stage.npz")
with np.load(OUT / "raw_first_stage.npz") as loaded:
    raw = {k: loaded[k] for k in loaded.files}
display(pd.DataFrame([audit_transitions(raw)]))
display(pd.DataFrame({"episode": raw["episode"][:6], "step": raw["step"][:6],
                      "torque": raw["action"][:6, 0], "reward": raw["reward"][:6]}))
print("Fields:", list(raw))
''')
md(r'''
## 5 · 找到一个非常实际的坑：动作尺度与时间截断

模拟器使用 [-2,2] 的动作；SB3 的 replay buffer 里动作缩放到 [-1,1]。字段同名不代表单位相同，导出数据或换框架时要显式转换。

`terminated` 指任务本身结束，`truncated` 指外部限制导致停止。Pendulum 的第 200 步是时间截断。
对于此处继续任务的价值估计，时间到了不等于下一状态价值为零。SB3 记录 timeout，在采样时使用 `done * (1-timeout)` 作为终止掩码。
下面检查真正写入 buffer 的记录，尤其是时间截断前的 next_obs 是否被错误换成 reset 后的初始观测。
''')
code('''
buffer = model.replay_buffer
n = len(raw["obs"])
assert n < buffer.buffer_size  # 当前 buffer 未环绕，便于逐行核对
np.testing.assert_allclose(buffer.observations[:n, 0], raw["obs"], atol=1e-6)
np.testing.assert_allclose(buffer.next_observations[:n, 0], raw["next_obs"], atol=1e-6)
np.testing.assert_allclose(buffer.actions[:n, 0], model.policy.scale_action(raw["action"]), atol=1e-6)
cut = int(np.flatnonzero(raw["truncated"])[0])
done = float(buffer.dones[cut, 0]); timeout = float(buffer.timeouts[cut, 0])
effective_done = done * (1 - timeout)
assert effective_done == 0 and not raw["terminated"][cut]
display(pd.DataFrame([dict(row=cut, terminated=bool(raw["terminated"][cut]),
    truncated=bool(raw["truncated"][cut]), buffer_done=done, timeout=timeout,
    effective_done=effective_done)]))

# 数值示例，只演示终止掩码；5.0 不是实际 critic 估计，省略了 SAC 熵项。
toy_reward, toy_next_value = -1., 5.
print("Time limit target example:", toy_reward + .99 * (1-effective_done) * toy_next_value)
print("Incorrectly treating time limit as terminal:", toy_reward)
''')
md(r'''
**停下来复述**：如果只导出一个 `done` 列，你会丢失什么信息？如果把 scaled_action 当原始力矩执行，会有什么变化？

## 6 · 继续训练，记录真实预算

第二阶段沿用同一 actor、critic、优化器和 replay buffer，追加 `TOTAL_STEPS-FIRST_STEPS` 次交互。没有重建模型。
这里比较的是一条训练轨迹的不同预算点，不是多个独立训练样本。
加载评测用 checkpoint 时会临时初始化网络；辅助函数保存并恢复 CPU 随机数状态，避免“读取 checkpoint”本身改变后续训练的随机序列。
''')
code('''
model.learn(total_timesteps=TOTAL_STEPS-FIRST_STEPS, reset_num_timesteps=False,
            callback=Progress(), log_interval=1)
assert model.num_timesteps == TOTAL_STEPS
policies[f"sac_{TOTAL_STEPS}"] = checkpoint(f"sac_{TOTAL_STEPS}")
training_env.save(OUT / "raw_experience.npz")
with np.load(OUT / "raw_experience.npz") as loaded:
    raw = {k: loaded[k] for k in loaded.files}
assert len(raw["obs"]) == TOTAL_STEPS
display(pd.DataFrame([audit_transitions(raw)]))
print("Raw arrays total:", round(sum(x.nbytes for x in raw.values())/1024**2, 2), "MiB")
training_env.close()
''')
code('''
progress = pd.read_csv(OUT / "training/progress.csv")
display(progress.tail(3))
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(progress["time/total_timesteps"], progress["rollout/ep_rew_mean"])
axes[0].set(xlabel="environment steps", ylabel="training return", title="Exploratory behavior")
axes[1].plot(progress["time/total_timesteps"], progress["train/critic_loss"])
axes[1].set(xlabel="environment steps", ylabel="critic loss", title="Changing bootstrapped targets")
fig.tight_layout(); fig.savefig(OUT / "training_curves.png", dpi=130); plt.close(fig)
display(Image(filename=str(OUT / "training_curves.png")))
''')
md(r'''
**读曲线**：critic 的目标由奖励、后续价值估计和策略共同决定，会随训练变化。它不像固定监督标签上的 MSE；“critic loss 不单调下降”本身不能判定失败。

## 7 · 冻结 checkpoint 后独立评测

随机策略、未训练策略、两个训练阶段使用相同的评测 seed，并记录每个 episode。
随机策略与未训练的确定性神经网络不是同一个基线。

三种场景分别评测：正常、100 ms 观测延迟、重力从 10 改到 12。改变重力时不同时加延迟，以免混淆原因。
随机动作不读取观测，因此单独加观测延迟不应改变它的结果，这也提供了评测器自检。
''')
code('''
EVAL_SEEDS = list(range(10000, 10000+N_EVAL))
rows = []  # 重跑此格不会累积重复结果
for delay, gravity in [(0, 10.), (2, 10.), (0, 12.)]:
    rows += evaluate(None, "random", EVAL_SEEDS, delay, gravity, OUT)
    for name, policy in policies.items():
        rows += evaluate(policy, name, EVAL_SEEDS, delay, gravity, OUT)
frame = summarize(rows, OUT, "return", "SAC snapshots and robustness: higher is better")
display(frame.groupby(["policy", "condition"])[["return", "upright_fraction"]].mean().round(3))
display(Image(filename=str(OUT / "comparison.png")))
random_results = frame[frame.policy == "random"].pivot(index="seed", columns="condition", values="return")
np.testing.assert_array_equal(random_results["delay_0_g10"], random_results["delay_2_g10"])
''')
code('''
paired = pd.concat([paired_return(frame, f"sac_{TOTAL_STEPS}", reference)
                    for reference in ["random", f"sac_{FIRST_STEPS}"]], ignore_index=True)
paired.to_csv(OUT / "controlled_paired_deltas.csv", index=False)
display(paired.round(3))
print("Delta > 0 means the final snapshot has higher return. These intervals condition on one training seed.")
for name in [f"sac_{FIRST_STEPS}", f"sac_{TOTAL_STEPS}"]:
    display(Markdown(f"**{name} · 同一个初始状态，真实仿真回放**"))
    show_gif(OUT / f"{name}.gif")
''')
md(r'''
## 8 · 复盘与下一轮

先回答，再读同目录 `02_参考复盘.md`。用实际表格引用数值，不写“模型应该变好了”。
回放只是一个初始状态，总体结论看完整分布；置信区间按 episode 重采样，但这里只有一个训练 seed。
''')
code('''
MY_REVIEW = {
    "没有动作标签，更新信号从哪里来？": "待填写：解释 replay 中的一行怎样参与价值与策略更新。",
    "训练曲线与独立评测是否一致？": "待填写：引用至少两个数值，解释口径差异。",
    "哪个压力条件损害最大？": "待填写：正常/延迟/重力变化分别怎样？",
    "保存策略等于能够原样续训吗？": "待填写：讨论 replay、随机数状态与模拟器状态。",
    "下一轮只改变什么？": "待填写：假设、固定变量、评测标准。",
}
write_review(OUT, "SAC 学习闭环复盘", MY_PREDICTION, MY_REVIEW, frame, time.perf_counter()-START)
''')
md(r'''
## 9 · 第二遍作业与课件

1. 固定所有配置，换训练 seeds 7、11、19；按训练 seed 比较最终回报和压力测试，不把 16×200 帧当 3,200 个独立样本。
2. 冻结网络，选 5k/10k/20k 总预算，记录实际环境步数和墙钟耗时。多训练是否总是更好？
3. 再用 `loop02_reinforcement.py --algo ppo --steps 10000` 运行 PPO。它可能按完整 rollout 超出请求步数；对齐实际交互预算，也记录计算量，不能用一次小预算结果宣布算法优劣。
4. 假如要改奖励，必须在原始奖励和物理行为指标下复评，检查有没有“新分数更好、实际行为更差”。

下一课继续用同一个摆，但学习目标换成“预测下一状态”，动作交给规划器来选。

随用随读的一手资料：

- [Pendulum 环境合同](https://gymnasium.farama.org/environments/classic_control/pendulum/)：对应第 2 节。
- [SB3 SAC 文档](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)：对应第 3–6 节；安装版本由本项目锁文件决定。
- [SB3 时间截断处理源码](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.0/stable_baselines3/common/buffers.py)：搜索 `handle_timeout_termination`，对应第 5 节。
- [SAC 原论文](https://arxiv.org/abs/1801.01290)：先读算法目标，再研究 critic/actor update。

可交付的小研究任务：给一个实际数据导出流程增加动作单位或 timeout 回归检查，证明它能捕获具体错误，再考虑贡献到相应项目。
''')
save("02_强化学习与经验回放.ipynb")

md(r'''
# 03 · 学一个世界模型，用它规划，再用新数据修复它

**研究问题：从随机轨迹学到的动力学，能否支持控制？让规划器采集新数据后，重训会更好吗？**

继续使用第二课的 Pendulum，把任务保持简单，集中理解方法的变化。
第二课直接学 $\pi(o)=a$；本课学 $f_\theta(o,a)\approx o'$，再通过搜索动作序列来控制系统。

完整闭环：`随机交互 → 动力学拟合 → 一步/多步诊断 → MPC 执行 → 新轨迹 → 对照重训 → 再评测 → 复盘`。
量化类比：先建立可用于情景推演的模型，再优化决策；决策走到的新区域又会暴露模型缺口。

这里“世界模型”指低维状态转移模型。奖励函数已知，网络只学动力学；不是图像生成、Dreamer 或 PETS 完整复现。

## 使用方式与交付物

```bash
uv run --locked --group notebooks jupyter lab notebooks/03_世界模型与规划迭代.ipynb
```
从项目目录执行。第一遍逐格跑，先预测结果，再看输出，预留 90–120 分钟理解与复盘。
你会得到原始 transition、四个动力学模型、同预算对照表、实际控制回放和规划耗时记录。
CPU 默认配置以小实验为目标，机器计算时间单独记录。
''')
md(r'''
## 1 · 固定协议

控制器每 3 个环境步重新规划一次，即 150 ms；动作在这 3 步内保持不变。
预测 8 个动作块，就是 $8\times3\times0.05=1.2$ 秒的未来。每次只执行第一块，然后读新观测重新规划。

第一遍不追求最强分数。先确保你能解释这个过程，再改模型、时域或搜索预算。
**先预测**：一步预测误差最小的模型，闭环回报一定最高吗？新增规划器数据一定优于新增随机数据吗？
''')
code('''
SEED = 7
N_TRAIN = 20
UPDATES = 1000
N_EXTRA = 8
N_EVAL = 8
PLAN = dict(horizon=8, candidates=48, iterations=2, repeat=3)
MY_PREDICTION = "待填写：我的预测是……因为……"
''')
code('''
from pathlib import Path
import sys, time, copy
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "course_utils.py").exists()), None)
if ROOT is None:
    raise RuntimeError("请在完整 robot_learning_loops 目录内启动。")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from IPython.display import display, Image, Markdown
from common import plt, summarize
from loop03_world_model import collect, Dynamics, CEMPlanner, known_transition, running_cost
from course_utils import (begin_notebook, audit_transitions, prediction_diagnostics,
                          collect_with_planner, reload_dynamics, paired_return, show_gif, write_review)
START = time.perf_counter()
OUT, manifest = begin_notebook("notebook03", "03_世界模型与规划迭代.ipynb",
    dict(seed=SEED, train_episodes=N_TRAIN, updates=UPDATES, extra_episodes=N_EXTRA,
         eval_episodes=N_EVAL, **PLAN))
display(pd.Series(manifest["versions"], name="installed_version").to_frame())
print("Planning horizon:", PLAN["horizon"]*PLAN["repeat"]*.05, "seconds")
''')
md(r'''
## 2 · 数据标签换成“动作后实际发生了什么”

输入是 $(o_t,a_t)$，标签是 $o_{t+1}-o_t$。训练时不需要专家动作，也不需要通过奖励更新 actor。
训练集由随机动作采集；验证集按完整 episode 分开，避免相邻帧泄漏。

先检查字段、连续性和单位。cos/sin 是无量纲，角速度是 rad/s，动作是力矩。
网络拟合标准化状态增量；输入和标签的统计量只从训练集估计。
''')
code('''
train = collect(range(N_TRAIN))
val = collect(range(2000, 2010))
assert not set(train["episode"]) & set(val["episode"])
np.savez_compressed(OUT / "raw_train.npz", **train)
np.savez_compressed(OUT / "raw_validation.npz", **val)
display(pd.DataFrame([audit_transitions(train, require_closed=True),
                      audit_transitions(val, require_closed=True)], index=["train", "validation"]))
display(pd.DataFrame({"theta_velocity": train["obs"][:6, 2], "action": train["action"][:6, 0],
                      "next_velocity": train["next_obs"][:6, 2],
                      "delta_velocity": (train["next_obs"]-train["obs"])[:6, 2]}))
''')
md(r'''
## 3 · 写出动力学训练循环

网络结构为 `4 → 128 → 128 → 3`；4 维输入是观测加动作，3 维输出是标准化状态增量。
输出加回当前状态后，cos/sin 投影回单位圆，角速度限制在环境范围。这个物理先验减少无效观测，但不能保证动力学准确。

训练目标：标准化增量 MSE。验证目标：实际下一观测预测误差。两者尺度不同。
本课用固定更新预算，不按开发评测成绩选择 checkpoint。
''')
code('''
inputs = np.concatenate([train["obs"], train["action"]], axis=1)
deltas = train["next_obs"] - train["obs"]
torch.manual_seed(SEED)
base = Dynamics(inputs.mean(0), np.maximum(inputs.std(0), .01),
                deltas.mean(0), np.maximum(deltas.std(0), .001))
print(base.net)

def train_dynamics(model, data, validation, updates, seed, phase):
    x, a, y = [torch.tensor(data[k], dtype=torch.float32) for k in ["obs", "action", "next_obs"]]
    vx, va, vy = [torch.tensor(validation[k], dtype=torch.float32) for k in ["obs", "action", "next_obs"]]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    records = []
    for step in range(updates):
        idx = rng.integers(0, len(x), 256)
        prediction = model.normalized_delta(x[idx], a[idx])
        target = (y[idx]-x[idx]-model.ym)/model.ys
        loss = (prediction-target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % max(1, updates//10) == 0 or step == updates-1:
            with torch.inference_mode():
                validation_loss = (model(vx, va)-vy).square().mean().item()
            records.append(dict(phase=phase, step=step+1, train_normalized_mse=loss.item(),
                                val_next_state_mse=validation_loss))
    model.eval()
    return records

history = train_dynamics(base, train, val, UPDATES, SEED, "base")
base = reload_dynamics(base, OUT, "base", val)
display(pd.DataFrame(history).iloc[[0, -1]])
''')
code('''
curve = pd.DataFrame(history)
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(curve.step, curve.train_normalized_mse)
axes[0].set(xlabel="updates", ylabel="normalized delta MSE", title="Training minibatch")
axes[1].plot(curve.step, curve.val_next_state_mse)
axes[1].set(xlabel="updates", ylabel="next observation MSE", title="Validation transitions")
fig.tight_layout(); fig.savefig(OUT / "dynamics_training.png", dpi=130); plt.close(fig)
display(Image(filename=str(OUT / "dynamics_training.png")))
''')
md(r'''
## 4 · 一步准，不等于推演 25 步仍然准

把预测出的状态重新送入模型，使用验证轨迹中的同一串动作向前滚动；中途不喂入真实状态纠错。
每个预测窗口完全位于同一 episode 内。这个指标检验多步模型误差，不是闭环控制成绩。

总 state MSE 混合了不同单位，只作工程诊断；下面另外报告角度 RMSE（rad）和角速度 RMSE（rad/s）。
角度误差按周期 wrap，避免把接近 $-\pi$ 和 $\pi$ 的两个姿态当成相差约 $2\pi$。
''')
code('''
errors = prediction_diagnostics(base, val)
errors.to_csv(OUT / "base_multistep_error.csv", index=False)
display(errors.groupby("horizon")[["angle_rmse_rad", "velocity_rmse_rad_s"]].mean().round(4))
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for ax, metric in zip(axes, ["angle_rmse_rad", "velocity_rmse_rad_s"]):
    errors.groupby("horizon")[metric].mean().plot(ax=ax, marker="o")
    ax.set(xlabel="open-loop prediction steps", ylabel=metric)
fig.tight_layout(); fig.savefig(OUT / "multistep_error.png", dpi=130); plt.close(fig)
display(Image(filename=str(OUT / "multistep_error.png")))
''')
md(r'''
## 5 · 用模型选动作：CEM 和 MPC 各做什么？

CEM 在这里是一个动作序列搜索器：采样多条候选序列 → 用模型推演每条序列的成本 → 保留成本较低的候选 → 更新采样分布 → 再搜索。
MPC 是执行方式：取搜索结果的第一段动作去执行，然后根据**模拟器的新观测**重新规划。

本课成本函数与环境负奖励相同，是已知函数；我们只学状态转移。
下面用随机候选序列做一次可见的打分，随后使用完整 CEMPlanner 执行控制。
''')
code('''
env = gym.make("Pendulum-v1")
initial, _ = env.reset(seed=10000)
env.close()
generator = torch.Generator().manual_seed(SEED)
candidates = torch.rand(48, PLAN["horizon"], 1, generator=generator)*4-2
with torch.inference_mode():
    imagined = torch.tensor(initial, dtype=torch.float32).repeat(48, 1)
    costs = torch.zeros(48)
    for h in range(PLAN["horizon"]):
        for _ in range(PLAN["repeat"]):
            costs += running_cost(imagined, candidates[:, h])
            imagined = base(imagined, candidates[:, h])
display(pd.DataFrame({"candidate": range(48), "predicted_cost": costs.numpy(),
                      "first_torque": candidates[:, 0, 0].numpy()}).sort_values("predicted_cost").head(6))
print("Actual CEM first action:", CEMPlanner(base, **PLAN, seed=SEED).action(initial))
''')
md(r'''
**停下来回答**：如果优化器找到了一条“模型以为很便宜，实际却很差”的序列，增加搜索候选一定有帮助吗？

## 6 · 第一轮部署：随机、已知方程、学到的模型

三者使用相同初始 seed 和动作保持周期。已知方程 MPC 仍受搜索预算与规划时域限制，不是全局最优上界。
所有数据标签取自实际 `env.step`；模型想象出的下一状态不会被伪装成真实训练标签。

这里与第二课 SAC 的默认动作频率不同；跨课回报不能直接用来给算法排榜。
''')
code('''
EVAL_SEEDS = list(range(10000, 10000+N_EVAL))
baseline_rows = []
for name, transition in [("random", None), ("known_mpc", known_transition), ("base_mpc", base)]:
    trajectory, scores = collect_with_planner(transition, name, EVAL_SEEDS, PLAN, OUT)
    audit_transitions(trajectory, require_closed=True)
    np.savez_compressed(OUT / f"evaluation_{name}.npz", **trajectory)
    baseline_rows += scores
display(pd.DataFrame(baseline_rows).groupby("policy")[["return", "upright_fraction"]].mean().round(3))
show_gif(OUT / "base_mpc.gif")
''')
md(r'''
## 7 · 完成第二个闭环：模型驱动新数据采集

现在冻结第一轮模型，让它在**新的采集初始条件**下用 MPC 控制摆。把实际访问的状态、动作、真实下一状态收回来。
再从相同初始条件收集等量随机数据，保持相同动作频率。

| 组 | 新标签来源 | 追加更新 | 起点 |
|---|---|---:|---|
| more_updates | 无新增标签 | S | 同一个 base |
| more_random | 随机动作采集 | S | 同一个 base |
| planner_data | base 模型驱动的 MPC 采集 | S | 同一个 base |

初始训练数据每步换随机动作；本轮两种新增采集都每 3 步换动作，因此新增数据对照的控制频率一致。
归一化统计保持初始训练集不变；追加阶段都重建 Adam、使用相同 batch 和更新预算。
''')
code('''
EXTRA_SEEDS = list(range(3000, 3000+N_EXTRA))
assert not set(EXTRA_SEEDS) & (set(EVAL_SEEDS) | set(train["episode"]) | set(val["episode"]))
planner_data, _ = collect_with_planner(base, "collection_mpc", EXTRA_SEEDS, PLAN)
random_data, _ = collect_with_planner(None, "collection_random", EXTRA_SEEDS, PLAN)
assert len(planner_data["obs"]) == len(random_data["obs"])
for name, data in [("planner", planner_data), ("random", random_data)]:
    audit_transitions(data, require_closed=True)
    np.savez_compressed(OUT / f"raw_extra_{name}.npz", **data)
coverage = []
for name, data in [("initial", train), ("extra_random", random_data), ("extra_planner", planner_data)]:
    angle = np.arctan2(data["obs"][:, 1], data["obs"][:, 0])
    coverage.append(dict(data=name, labels=len(angle),
        fraction_near_upright=float(np.mean((np.abs(angle)<.2) & (np.abs(data["obs"][:, 2])<1)))))
display(pd.DataFrame(coverage))
print("State coverage differences are descriptive; improvement still requires controlled evaluation.")
''')
code('''
models = {"base_mpc": base}
for name, addition in [("more_updates", None), ("more_random", random_data), ("planner_data", planner_data)]:
    data = train if addition is None else {k: np.concatenate([train[k], addition[k]]) for k in train}
    model = copy.deepcopy(base)
    history += train_dynamics(model, data, val, UPDATES, SEED+1, name)
    models[name] = reload_dynamics(model, OUT, name, val)
pd.DataFrame(history).to_csv(OUT / "training.csv", index=False)
print("All models reloaded from saved weights.")
''')
md(r'''
## 8 · 对照复评：模型更准、控制更好、计算更快是三件事

所有训练已经完成，下面评测各组。先看总体回报，再看“保持向上比例”和规划 p95 耗时。
记录的耗时只覆盖规划调用，不包含真实传感器、传输、驱动和调度；低于 150 ms 也不能证明真机实时性。

新增规划器数据未必取胜：小环境可能已经容易拟合；也可能数据集中在少数状态，削弱其他区域的精度。保留负结果，别为了漂亮结论改评测集合。
''')
code('''
rows = list(baseline_rows)
for name, model in models.items():
    if name != "base_mpc":
        trajectory, scores = collect_with_planner(model, name, EVAL_SEEDS, PLAN, OUT)
        np.savez_compressed(OUT / f"evaluation_{name}.npz", **trajectory)
        rows += scores
frame = summarize(rows, OUT, "return", "Dynamics -> planning -> new data -> retraining")
display(frame.groupby("policy")[["return", "upright_fraction", "planning_p95_ms"]].mean().round(3))
display(Image(filename=str(OUT / "comparison.png")))
paired = pd.concat([paired_return(frame, "planner_data", ref)
                    for ref in ["base_mpc", "more_updates", "more_random"]], ignore_index=True)
paired.to_csv(OUT / "controlled_paired_deltas.csv", index=False)
display(paired.round(3))
''')
code('''
diagnostics = []
for name, model in models.items():
    error = prediction_diagnostics(model, val)
    error["policy"] = name
    diagnostics.append(error)
diagnostics = pd.concat(diagnostics, ignore_index=True)
diagnostics.to_csv(OUT / "all_multistep_errors.csv", index=False)
offline = diagnostics[diagnostics.horizon == 25].groupby("policy")[["angle_rmse_rad", "velocity_rmse_rad_s"]].mean()
online = frame.groupby("policy")[["return", "upright_fraction"]].mean()
display(offline.join(online).sort_values("return", ascending=False).round(4))
display(Markdown("**补规划器数据后：与第一轮同一个初始状态**"))
show_gif(OUT / "planner_data.gif")
''')
md(r'''
## 9 · 复盘：把预测模型和决策系统分别讲清楚

先填写，再读 `03_参考复盘.md`。如果某组差异的区间跨 0，就如实写“这次评测不足以清晰区分”。
这里只做了单轮数据迭代、一个训练 seed 和少量开发 episodes；跨种子、未触碰测试集与跨任务泛化仍是下一步。
''')
code('''
MY_REVIEW = {
    "本课学习的函数是什么？": "待填写：明确输入、标签、输出，以及已知的成本函数。",
    "CEM 与 MPC 分别负责什么？": "待填写：推演、搜索、执行和重规划各在哪里？",
    "新增数据有没有比多训练更好？": "待填写：引用配对差值与区间。",
    "离线模型误差与控制回报是否同序？": "待填写：用本次结果解释，不假设必然相关。",
    "下一轮只改什么？": "待填写：时域/搜索预算/模型集成/数据之一；写出判定标准。",
}
write_review(OUT, "世界模型与规划迭代复盘", MY_PREDICTION, MY_REVIEW, frame, time.perf_counter()-START)
''')
md(r'''
## 10 · 第二遍作业：从小实验走向研究

1. 固定其他配置，只改 horizon=4/8/12，记录多步预测误差、闭环回报与规划耗时。长时域是否总是更好？
2. 固定模型，只改 candidates=24/48/96。搜索更充分能否弥补模型偏差？
3. 固定开发协议，运行 seeds 7、11、19，再判断新增数据的效果是否稳定。
4. 下一阶段引入多个动力学模型，检查模型间分歧是否能帮助识别未知区域。分歧不自动等于校准的不确定性；要用实际误差验证。
5. 确定方法后冻结参数，另取未使用初始条件作最终评测；再考虑更复杂的 MuJoCo 任务。

## 知识地图与一手资料

- [Pendulum 环境合同](https://gymnasium.farama.org/environments/classic_control/pendulum/)：检查状态、动作、奖励与时间。
- [PETS 论文](https://arxiv.org/abs/1805.12114)：后续理解概率动力学集成和 trajectory sampling；本课单模型没有实现这些机制。
- [MBRL-Lib](https://github.com/facebookresearch/mbrl-lib)：作为方法结构的阅读入口，本课不依赖其安装环境。
- 项目源文件 `loop03_world_model.py`：查看 `Dynamics`、`known_transition`、`CEMPlanner.action` 的具体实现。

前三课给你三个不同的可闭环对象：**学动作、用奖励学动作、学动力学再规划动作**。
接下来读 `../lessons/04_ACT与DiffusionPolicy.md`，把观测换成图像、输出换成动作序列。继续沿用原始数据审计、checkpoint、闭环评测和复盘习惯。

可贡献的小任务：一个记录规划耗时的评测器、一个区分预测误差与控制效果的实验报告，或可复现的模型失配案例。先定位真实缺口，再考虑上游贡献。
''')
save("03_世界模型与规划迭代.ipynb")
