# Physical AI · Day 0 上手包（新电脑自检 → 安装 → 冒烟测试 → 练习闭环）

> 配套《Physical AI 训战合一计划》使用。目标：一台全新电脑，在 1–2 天内变成能跑通
> `数据 → 训练 → 评测 → 归因` 完整闭环的工作站，并且每一步都有可验证的通过标准。

## 0. 硬件自检（10 分钟）

先搞清楚这台机器能做什么、不能做什么，再决定装什么。

```bash
# CPU / 内存 / 磁盘
lscpu | grep -E "Model name|^CPU\(s\)"
free -h
df -h /

# GPU（Linux）
nvidia-smi          # 有输出 → 有 NVIDIA GPU；没有 → 走 CPU/云 GPU 路线
```

判定标准：

| 配置 | 结论 |
|---|---|
| NVIDIA GPU ≥ 8GB 显存 | 本地可训 ACT / Diffusion Policy / SmolVLA（小任务） |
| NVIDIA GPU < 8GB 或 Apple Silicon | 本地做数据处理 + 评测 + 小规模训练，大训练上云 |
| 无 GPU | 本地只做数据侧（Profiler、审计、可视化），训练全部上云 |

云 GPU 备选（按需租用即可，不必买卡）：AutoDL / 阿里云 PAI / Lambda / RunPod。
一张 4090 按小时租，跑完练习 1–4 的总成本 < ¥100。

## 1. 系统层安装（30 分钟）

推荐 Ubuntu 22.04/24.04（机器人生态一等公民）。macOS 可用于数据侧全部工作 + MuJoCo。

```bash
# 基础工具
sudo apt update && sudo apt install -y git curl wget ffmpeg build-essential cmake

# NVIDIA 驱动（仅 Linux + NVIDIA GPU）
sudo ubuntu-drivers autoinstall && sudo reboot
# 重启后验证：
nvidia-smi   # 能看到驱动版本和 CUDA Version 即通过（不需要单独装 CUDA toolkit，PyTorch 自带）

# Python 环境管理：用 uv（比 conda 快一个数量级，锁版本可复现）
curl -LsSf https://astral.sh/uv/install.sh | sh
```

git 身份（留痕体系的前提）：

```bash
git config --global user.name  "Yudi Wang"
git config --global user.email "wyd1582@gmail.com"
```

## 2. 核心栈安装（每装一层，验证一层）

原则：**每一层装完立刻跑冒烟测试，绿了才装下一层**。本目录 `smoke/` 下有现成脚本。

```bash
mkdir -p ~/work && cd ~/work
uv venv pai --python 3.10 && source pai/bin/activate
```

### 第 1 层：PyTorch

```bash
uv pip install torch torchvision
python smoke/00_torch_gpu.py     # 通过标准：打印设备名，矩阵乘 benchmark 正常
```

### 第 2 层：LeRobot（数据 + 训练 + 评测的主框架）

```bash
uv pip install "lerobot[pusht]"
python smoke/01_lerobot_dataset.py   # 通过标准：拉取 pusht 数据集，打印 episode 结构
```

### 第 3 层：MuJoCo + Gymnasium（仿真）

```bash
uv pip install mujoco gymnasium gymnasium-robotics
python smoke/02_mujoco.py            # 通过标准：物理步进 1000 步，打印仿真耗时
```

### 第 4 层：实验记录

```bash
uv pip install wandb tensorboard
wandb login    # 或 export WANDB_MODE=offline 先离线用
```

### 一键自检

```bash
bash check_env.sh   # 汇总以上所有检查，输出 PASS/FAIL 清单
```

## 3. 练习闭环

见 `exercises/练习速览.md`。六个练习，每个都是一次完整的小闭环，
全部做完 = 你已经亲手完成过一遍「数据处理 → 训练 → 评估 → 归因 → 迭代」。

## 4. 留痕纪律（对齐《02 开源蓝图》）

- 每个练习的 notebook / 脚本 / 结论提交到 `physical-ai-notes` 仓库，允许粗糙、禁止事后美化。
- 每个练习开一个 issue，做完关闭；commit message 英文、动词开头、说清 why。
- 练习 2（数据消融）和练习 4（失败归因）的产出直接并入 `physical-ai-data-observatory`。
