# XqRobotV2 轮腿机器人 RL 训练项目

> 基于 [UniLab](https://github.com/unilabsim/UniLab) 框架，参考 [CJ-003](https://github.com/Albusgive/wheel_legged_genesis) 设计。
> 更新：2026-06-27

---

## 一、快速开始

```bash
cd /home/robot/xiaoq/wheel_legged_RL_unilab

conda activate unilab

# Flat Walk 训练
CUDA_VISIBLE_DEVICES=1 bash shell/train_ppo_flat.sh

# 键盘验证
bash shell/eval_ppo_flat.sh --keyboard

# TensorBoard
bash shell/tensorboard.sh flat 8080
```

---

## 二、训练任务

| 任务 | 脚本 | 命令 | 地形 | 状态 |
|------|------|------|------|------|
| **Flat Walk** | `train_ppo_flat.sh` | 5D | 平面 | ✅ 主力 |
| Rough Walk | `train_ppo_rough.sh` | 5D | 6×6 网格 | 待验证 |
| Jump Flat | `train_ppo_jump_flat.sh` | 5D | 平面 | 待验证 |
| Toe Walk Flat | `train_ppo_toe_walk_flat.sh` | 4D | 平面 | 待验证 |

---

## 三、机器人物理参数

### 关节定义

| 索引 | 关节 | 轴 | 范围 | 默认角 | 控制 |
|------|------|-----|------|--------|------|
| 0 | left_joint_1 (左髋) | `1 0 0` (roll) | [-π, π] | 0.1 | 位置, kp=30 |
| 1 | left_joint_2 (左大腿) | `0 1 0` (pitch) | [0, 2.09] | 0.1 | 位置, kp=30 |
| 2 | left_joint_3 (左小腿) | `0 1 0` (pitch) | [-0.87, 0.87] | -0.1 | 位置, kp=30 |
| 3 | right_joint_1 (右髋) | `1 0 0` (roll) | [-π, π] | 0.1 | 位置, kp=30 |
| 4 | right_joint_2 (右大腿) | `0 1 0` (pitch) | [0, 2.09] | 0.1 | 位置, kp=30 |
| 5 | right_joint_3 (右小腿) | `0 1 0` (pitch) | [-0.87, 0.87] | -0.1 | 位置, kp=30 |
| 6 | left_joint_wheel | `0 1 0` | 无限 | 0 | **速度, kv=1** |
| 7 | right_joint_wheel | `0 1 0` | 无限 | 0 | **速度, kv=1** |

### 尺寸

| 部件 | 长度 |
|------|------|
| 大腿 | ~300mm |
| 小腿 | ~300mm |
| 轮径 | ~200mm |

### 控制架构

| 组件 | 配置 |
|------|------|
| 控制频率 | **100Hz** (ctrl_dt=0.01) |
| 腿关节 | 位置控制 (kp=30) |
| 轮关节 | **速度控制 (kv=1)**，action×10 → rad/s |
| 仿真步长 | 0.005s (每 ctrl 2 substeps) |

---

## 四、动作空间

8 维 `[-1, 1]`：

| 索引 | 关节 | 缩放 | 目标 |
|------|------|------|------|
| 0-5 | 腿 | action × 0.5 + default | 位置 (rad) |
| 6-7 | 轮 | action × 10.0 | 速度 (rad/s) |

---

## 五、观测空间

### 5D 命令

```
[vx, vy, vyaw, tsk, height_target]
```

| 维度 | 范围 | 说明 |
|------|------|------|
| vx | [-0.6, 0.6] | 前向速度 (m/s) |
| vy | [-0.3, 0.3] | 侧向速度 |
| vyaw | [-1.0, 1.0] | 角速度 (rad/s) |
| tsk | [-0.1, 0.1] | 髋差动 |
| height | [0.45, 0.85] | 目标机身高度 (m) |

### 观测组成

9 帧历史堆叠。单帧 33 维（actor）/ 36 维（critic）：

| 特征 | dim | 说明 |
|------|-----|------|
| gyro | 3 | 角速度 |
| -gravity | 3 | 重力方向 |
| leg_diff | 6 | 关节角 - 默认角 |
| leg_vel | 6 | 关节速度 |
| wheel_vel | 2 | 轮子速度 |
| last_actions | 8 | 上一步动作 |
| commands | 5 | 5D 命令 |
| linvel (critic only) | 3 | 线速度 |

- Actor: 33 × 9 = **297**
- Critic: 36 × 9 = **324**

---

## 六、奖励函数

| 奖励 | 权重 | 类型 | 说明 |
|------|------|------|------|
| tracking_lin_vel | +1.5 | exp | `exp(-err²/0.3)` |
| tracking_ang_vel | +1.5 | exp | `exp(-err²/0.3)` |
| lin_vel_z | -0.2 | err² | 垂直速度 |
| ang_vel_xy | -0.05 | err² | roll/pitch 角速度 |
| base_height | -5.0 | err² | `(z-0.65)²` |
| orientation | **-10.0** | err² | `gravity_xy²` |
| joint_action_rate | -0.1 | err² | 腿动作平滑 |
| wheel_action_rate | **-0.005** | err² | 轮动作平滑（速度控制放宽） |
| similar_calf | **-1.0** | err² | 全腿对称（髋+大腿+小腿） |
| tsk | -2.0 | err² | 髋差动跟踪 |
| feet_distance | -1.0 | linear | 轮距 [0.3, 0.6]m |
| alive | +1.0 | const | 存活 |

### similar_calf 全腿对称

```python
hip   = (left_hip + right_hip)²       # 髋: 镜像 left ≈ -right
thigh = (left_thigh - right_thigh)²   # 大腿: 平行
calf  = (left_calf - right_calf)²     # 小腿: 平行
```

---

## 七、课程学习

| 参数 | 值 |
|------|-----|
| 扩展方式 | **对称扩展** |
| vel_step | 0.002 |
| ang_vel_step | 0.004 |
| 初始 vx | 配置值的 30% |
| 更新间隔 | 25 steps |
| 误差阈值 | 0.35 |

---

## 八、终止条件

| 条件 | 阈值 |
|------|------|
| 倾斜 | > 60° |
| 底盘高度 | < 0.20m |
| 大腿塌陷 | < 0.02 rad |
| 小腿极限 | abs > 0.85 rad |

---

## 九、PPO 超参数

| 参数 | 值 |
|------|-----|
| num_envs | 1024 |
| num_steps_per_env | 25 |
| max_iterations | 5000 |
| learning_rate | 1e-4 |
| entropy_coef | 0.01 |
| network | [512,512,256,128] + elu |
| init_noise_std | 0.5 |
| ctrl_dt | **0.01s (100Hz)** |

---

## 十、训练结果 (v1 基准)

> 备份：`backup/XqRobotV2WalkFlat/flat_walk_ppo_v1/`

| 指标 | 值 |
|------|-----|
| 最终奖励 | 54.1 |
| 最佳奖励 | 66.4 |
| 存活 | 905 / 1000 |
| 线速度追踪 | ~1.03 / 1.5 |
| 角速度追踪 | ~1.37 / 1.5 |
| 倾斜 | -0.035 (极小) |
| 轮距 | 0.0 (始终合格) |
| 训练时长 | ~2.2h (RTX 4090) |

---

## 十一、键盘控制

| 按键 | 功能 |
|------|------|
| ↑/↓ | 前进/后退 (vx) |
| ←/→ | 左转/右转 (vyaw) |
| A/D | 左移/右移 (vy) |
| Q/E | 降低/升高 机身高度 |
| Enter | 刹停 |
| Backspace | 重置 |
| 空格 | 暂停/恢复 |

---

## 十二、文件结构

```
wheel_legged_RL_unilab/
├── conf/ppo/task/
│   ├── xqrobotV2_walk_flat/mujoco.yaml    # ★ Flat 训练配置
│   ├── xqrobotV2_walk_rough/mujoco.yaml   # Rough 训练配置
│   ├── xqrobotV2_jump_flat/mujoco.yaml    # Jump 训练配置
│   └── xqrobotV2_toe_walk_flat/mujoco.yaml
├── src/unilab/envs/locomotion/xqrobotV2/
│   ├── base.py       # DEFAULT_LEG_ANGLES, 关节定义
│   ├── joystick.py   # ★ Flat Walk 环境
│   ├── rough.py      # Rough Walk 环境
│   ├── jump.py       # Jump Flat 环境
│   ├── toe_walk.py   # Toe Walk 环境
│   └── __init__.py
├── src/unilab/assets/robots/xqrobotV2/
│   ├── xqrobotV2.xml         # MuJoCo 模型
│   ├── scene_flat.xml        # 场景 + keyframe
│   └── locomotion_task.xml   # 地形模式 keyframe
├── shell/
│   ├── train_ppo_{flat,rough,jump_flat,toe_walk_flat}.sh
│   ├── eval_ppo_{flat,rough,jump_flat,toe_walk_flat}.sh
│   └── tensorboard.sh
├── backup/
│   ├── README.md
│   └── XqRobotV2WalkFlat/flat_walk_ppo_v1/
├── docs/
│   ├── PROJECT_ARCHITECTURE.md
│   └── PROBLEMS.md           # ★ 问题记录与解决方案
└── README.md                 # 本文件
```

---

## 十三、训练命令

```bash
# Flat Walk (主力)
CUDA_VISIBLE_DEVICES=1 bash shell/train_ppo_flat.sh

# 多 GPU 并行
CUDA_VISIBLE_DEVICES=0 bash shell/train_ppo_flat.sh &
CUDA_VISIBLE_DEVICES=1 bash shell/train_ppo_rough.sh &

# TensorBoard
bash shell/tensorboard.sh flat 8080

# 评估
bash shell/eval_ppo_flat.sh --keyboard

# 备份恢复
bash backup/XqRobotV2WalkFlat/flat_walk_ppo_v1/restore.sh
```
