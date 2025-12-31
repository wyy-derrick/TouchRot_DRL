# LeapHand 灵巧手手内旋转任务 - SAC强化学习

基于MuJoCo仿真和SAC算法的灵巧手手内物体旋转任务。

## 项目概述

使用强化学习控制Leap Hand灵巧手，利用关节感知和**17个触觉传感器**的二值化信息，完成立方体绕固定轴（Z轴）的定向旋转任务。

### 核心特性

- **核心算法**: SAC (Soft Actor-Critic)
- **多算法支持**: SAC / PPO / TD3 / Baseline 规则控制
- **触觉传感**: 17个触觉传感器，带容忍度的接触检测
- **训练模式**: 无头静默训练，TensorBoard可视化
- **演示模式**: MuJoCo可视化渲染

## 项目结构

```
TouchRot_DRL/
│
├── assets/                          # 网格资源文件 (.obj, .stl)
├── envs/                            # 环境定义
│   └── leap_hand_env.py             # 核心环境类 (Gym API)
│
├── utils/                           # 工具模块
│   ├── tactile_utils.py             # 触觉检测与二值化
│   ├── math_utils.py                # 旋转计算、四元数工具
│   └── logger.py                    # TensorBoard封装
│
├── algorithms/                      # 算法实现
│   ├── base_agent.py                # 算法基类
│   ├── sac.py                       # SAC Agent
│   ├── sac_models.py                # SAC网络架构 (Actor, Critic)
│   ├── replay_buffer.py             # 经验回放池
│   ├── ppo.py                       # [预留] PPO
│   └── td3.py                       # [预留] TD3
│
├── configs/                         # 配置文件
│   └── config.yaml                  # 超参数配置
│
├── checkpoints/                     # 模型权重输出
│   └── sac/
├── logs/                            # TensorBoard日志
│
├── scene_left(cubic).xml            # MuJoCo场景文件
├── left_hand(sensor).xml            # 灵巧手模型文件
├── train.py                         # 训练入口脚本
└── demo.py                          # 演示入口脚本
```

## 环境配置

### 依赖安装

```bash
pip install mujoco gymnasium numpy torch pyyaml tensorboard
```

### Python版本

- Python >= 3.8
- PyTorch >= 1.12
- MuJoCo >= 2.3

## 快速开始

### 1. 训练

```bash
# 使用默认配置训练
python train.py

# 使用自定义配置
python train.py --config configs/config.yaml

# 切换不同算法
python train.py --algo sac
python train.py --algo ppo
python train.py --algo td3
python train.py --algo baseline
```

训练过程中:
- 模型权重保存在 `checkpoints/<algo>/`
- TensorBoard日志保存在 `logs/`

### 2. 查看训练曲线

```bash
tensorboard --logdir=logs
```

### 3. 演示

```bash
# 使用最佳模型演示
python demo.py

# 指定模型路径
python demo.py --model checkpoints/sac/best_model.pth

# 指定算法 (SAC/PPO/TD3/Baseline)
python demo.py --algo ppo --model checkpoints/ppo/best_ppo_xxx.pth

# 指定演示episode数
python demo.py --episodes 10

# 不打印触觉信息
python demo.py --no-tactile
```

## 配置说明

## 算法一览

- `SAC`: 软Actor-Critic，最大熵目标、双Q网络和可选自适应熵系数。
- `PPO`: On-policy策略梯度，支持GAE、裁剪目标和多轮小批次更新。
- `TD3`: 双Q延迟DDPG，包含目标策略平滑和延迟策略更新。
- `Baseline`: 基于触觉启发的规则控制器，提供无学习的对照实验结果。

主要配置项在 `configs/config.yaml`:

### 环境配置
- `frame_skip`: 每个RL步的物理仿真步数
- `action_scale`: 动作缩放因子
- `tactile_threshold`: 触觉力阈值
- `reward_weights`: 奖励权重

### SAC配置
- `gamma`: 折扣因子 (0.99)
- `tau`: 软更新系数 (0.005)
- `lr_actor/lr_critic`: 学习率 (3e-4)
- `hidden_dim`: 网络隐藏层维度 (256)

### 训练配置
- `total_timesteps`: 总训练步数
- `batch_size`: 批次大小
- `buffer_size`: 回放池大小

## 观测空间

| 组成部分 | 维度 | 说明 |
|---------|------|------|
| qpos | 16 | 关节位置 |
| qvel | 16 | 关节速度 |
| tactile | 17 | 触觉传感器（二值） |
| target_axis | 3 | 目标旋转轴 |
| **总计** | **52** | |

## 动作空间

- 维度: 16 (16个关节)
- 范围: [-1, 1]
- 控制方式: 相对位置控制

## 奖励函数

```
r = w_rot * r_rotation + w_contact * r_contact + w_vel * r_velocity + w_action * r_action + w_drop * r_drop
```

- `r_rotation`: 绕Z轴旋转角度增量
- `r_contact`: 指尖接触奖励
- `r_velocity`: 物体速度惩罚
- `r_action`: 动作幅度惩罚
- `r_drop`: 物体掉落惩罚

## 触觉传感器

17个传感器分布:
- 手掌: 5个 (palm_1~3, palm_7~8)
- 近节: 4个 (if/mf/rf/th_px)
- 中节: 4个 (if/mf/rf_md, th_ds)
- 指尖: 4个 (if/mf/rf/th_tip)

## 扩展开发

### 添加新算法

1. 继承 `BaseAgent` 类
2. 实现 `select_action`, `update`, `save`, `load` 方法
3. 在 `train.py` 中添加算法选择逻辑

### 修改奖励函数

编辑 `envs/leap_hand_env.py` 中的 `_compute_reward` 方法。


