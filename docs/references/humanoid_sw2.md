# HumanoidSW2 (livelybot_pi_rl_baseline) 参考文档

> 路径: `/home/robot/Desktop/humanrobot/livelybot_pi_rl_baseline`
> 框架: Isaac Gym / rsl-rl | 12DOF 双足机器人

---

## 一、项目概述

基于 Isaac Gym 的 12DOF 双足人形机器人，采用**正弦轨迹跟踪**步态生成。训练策略学习跟踪预设的关节参考轨迹并添加修正，而非从零发现步态。这是 XqRobotV2 toe_walk v3 的设计来源。

---

## 二、机器人形态

### 2.1 关节布局 (12DOF 腿 + 10DOF 臂)

| 索引 | 关节 | 轴 | 功能 |
|------|------|-----|------|
| 5 | left_leg_roll_1 | X | **髋 pitch** (前屈/后伸) |
| 6 | left_leg_roll_2 | Z | **髋 roll** (外展/内收) |
| 7 | left_leg_roll_3 | Y | **膝** |
| 8 | left_leg_roll_4 | X | 踝 roll |
| 9 | left_leg_roll_5 | X | 踝 pitch |
| 10 | left_leg_roll_6 | Z | 踝 yaw |
| 16-21 | 右腿对称 | — | — |

**手臂不参与控制**（action 置零）。有效控制 DOF: **12** (6 × 2)。

### 对比 XqRobotV2

| | HumanoidSW2 | XqRobotV2 |
|------|-------------|-----------|
| 自由度 | **12 DOF** (含踝) | **8 DOF** (无踝) |
| 端部 | 扁平足底 | 轮子 |
| 平衡 | 踝关节主动调整 | 只能靠髋+轮子 |
| 髋设计 | 两轴 (pitch+roll) | 单轴 (roll) |

---

## 三、正弦轨迹跟踪

### 3.1 相位时钟

```python
phase = t / cycle_time  # cycle_time = 0.5s
sin_pos = sin(2π × phase + random_offset)

# 左腿 swing: sin < 0
# 右腿 swing: sin >= 0
```

`random_offset` 随机取 0 或 π，使一半 env 左腿先行，一半右腿先行。

### 3.2 参考轨迹

只用 **3 个关节**（髋 pitch、膝、踝 pitch），其余关节用正则化保持近零。

| 关节 | 摆动侧公式 | 支撑侧 |
|------|-----------|--------|
| 髋 pitch | `sin × 0.15` | 0 |
| 膝 | `-sin × 0.30` (双幅) | 0 |
| 踝 pitch | `sin × 0.15` | 0 |

### 3.3 双支撑过渡区

`|sin| < 0.1` (约 3.2% 周期) 时，双腿同时支撑，参考轨迹归零。

### 3.4 策略输出

策略输出的是**参考轨迹的修正量**：

```
joint_target = reference_trajectory + action × action_scale
```

`action_scale = 0.3`（修正量幅度远小于动作本身）。

---

## 四、奖励设计

### 4.1 主奖励

| 奖励 | 权重 | 说明 |
|------|------|------|
| tracking_lin_vel | 15 | XY 线速度追踪 |
| tracking_ang_vel | 30 | 角速度追踪 |
| joint_pos | 1.2 | 跟踪参考关节位置 |
| feet_contact_number | 1.5 | 足触地需匹配 stance mask |
| feet_air_time | 2.0 | 奖励摆动相空中时间 |
| **feet_clearance** | **8.0** | 摆动足离地高度 (> 0.02m) |

### 4.2 姿态约束

| 奖励 | 权重 | 说明 |
|------|------|------|
| orientation | 0.5 | 躯干直立 |
| base_height | 0.5 | 高度跟踪 (0.9m) |
| default_hip_roll | **3.0** | 髋 roll 保持 0 |
| default_thigh | 0.5 | 膝近 0 |

### 4.3 罚项

| 奖励 | 权重 |
|------|------|
| torques | -5e-6 |
| dof_vel | -2e-5 |
| action_smoothness | -0.001 |
| collision | -1.0 |

---

## 五、观测空间

| 分量 | 维度 | 缩放 |
|------|------|------|
| sin(phase), cos(phase) | 2 | ×1 |
| commands (vx,vy,dyaw) | 3 | ×2,2,1 |
| dof_pos − default | 12 | ×1 |
| dof_vel | 12 | ×0.05 |
| last_actions | 12 | ×1 |
| base_ang_vel | 3 | ×1 |
| base_euler | 3 | ×1 |
| **单帧** | **47** | |
| **frame_stack=15** | **705** | |

---

## 六、训练超参数

| 参数 | 值 |
|------|-----|
| num_envs | 4096 |
| frame_stack | **15** |
| learning_rate | 3e-5 |
| entropy_coef | 0.005 |
| gamma | 0.994 |
| lam | 0.9 |
| action_scale | 0.3 |
| PD kp | 15-60 Nm/rad |
| PD kd | 0.3-1.0 Nms/rad |
| ctrl_dt | 0.02 (50Hz) |
| episode | 12s (600 steps) |
| command resample | 每 8s |

---

## 七、对 XqRobotV2 的借鉴

### 已采用 (toe_walk v3)
- 正弦相位时钟 + 参考轨迹生成
- 策略输出修正量（action_scale 0.3）
- sin/cos 相位编码加入观测
- ref_tracking 奖励跟踪参考位置
- 双支撑过渡区

### 可借鉴
- **feet_clearance 奖励**：摆动腿离地高度约束，防止擦地
- **feet_contact_number**：需要触地传感器，当前不便实现
- **15 帧 frame_stack**：比当前 9 帧更长的时序记忆
- **低 learning_rate (3e-5)**：更稳定的参数更新

### 不适合借鉴
- 踝关节设计：XqRobotV2 无踝，脚部姿态无法独立控制
- 扁平足底：XqRobotV2 端部是轮子，触地特性不同
- Isaac Gym 框架：UniLab 使用 MuJoCo 后端
