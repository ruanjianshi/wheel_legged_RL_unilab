# PAI 12DOF (pi_12dof_release_v1) 参考文档

> 路径: `/home/robot/Desktop/humanrobot/livelybot_pi_rl_baseline/resources/robots/pi_12dof_release_v1`
> 框架: Isaac Gym / rsl-rl | 12DOF 双足机器人 | 质量 ~7kg

---

## 一、项目概述

PAI 是同项目 (livelybot_pi_rl_baseline) 中的**纯下肢双足机器人**，无手臂。与 HumanoidSW2 共享轨迹跟踪训练框架，但形态更紧凑（腿更短、质心更低、质量更轻）。

---

## 二、机器人形态

### 2.1 关节布局 (12DOF)

每腿 6 DOF (hip_pitch + hip_roll + thigh_yaw + knee + ankle_pitch + ankle_roll)。

| 索引 | 关节 | 轴 | 范围 [rad] | 功能 |
|------|------|-----|-----------|------|
| 0 | r_hip_pitch | Y | [-1.25, 1.75] | 右髋前屈/后伸 |
| 1 | r_hip_roll | X | [-0.50, 0.12] | 右髋外展/内收 |
| 2 | r_thigh | **Z** | [-0.60, 0.30] | 右大腿旋转 |
| 3 | r_calf | Y | [-0.65, 1.65] | **右膝** |
| 4 | r_ankle_pitch | Y | [-0.50, 1.30] | 右踝前屈 |
| 5 | r_ankle_roll | X | [-0.15, 0.15] | 右踝侧倾 |
| 6-11 | 左腿对称 | — | 镜像范围 | |

### 2.2 与 HumanoidSW2 差异

| | PAI | HumanoidSW2 |
|------|-----|-------------|
| 手臂 | **无** | 有 (10DOF, 不参与控制) |
| 质心高度 | **0.345m** | 0.90m |
| 总质量 | **~7kg** | ~8-9kg |
| 髋 range | 更大 (3.0 rad) | 更小 |
| 踝 roll | **±0.15** (极窄) | 更大 |
| PD kp | 20-40 | 15-60 |
| action_scale | **0.25** | 0.3 |
| lr | **1e-5** | 3e-5 |
| entropy_coef | **0.001** | 0.005 |

### 2.3 内置姿态

PAI 的 URDF 在关节 origin 处内置了**静态蹲姿偏移**：
- 髋 pitch: rpy=(0, -0.25, 0) → 髋前屈 14.3°
- 膝 (calf): rpy=(0, 0.65, 0) → 膝弯曲 37.2°
- 踝 pitch: rpy=(0, -0.40, 0) → 踝前屈 22.9°

这种设计使机器人在默认姿态（所有 joint=0）时即处于**微蹲状态**，无需额外关节角度即可站稳。这是 XqRobotV2 `DEFAULT_LEG_ANGLES = [0.1, 0.1, -0.1, ...]` 的等价实现——只不过 PAI 在机械结构上做了偏移，而 XqRobotV2 在代码层做了偏移。

### 2.4 髋关节不对称设计

| | 左髋 roll | 右髋 roll | 左大腿 yaw | 右大腿 yaw |
|------|----------|----------|-----------|-----------|
| 范围 | [-0.12, 0.50] | [-0.50, 0.12] | [-0.30, 0.60] | [-0.60, 0.30] |

**与 CJ-003 一致**——允许外展，限制内收防止碰撞机身。

---

## 三、训练设计

### 3.1 轨迹跟踪

与 HumanoidSW2 共享相同框架：正弦相位时钟 + 参考轨迹 + RL 修正。

### 3.2 奖励（特有项）

| 奖励 | 权重 | 说明 |
|------|------|------|
| feet_clearance | 高 | 摆动足离地高度 |
| knee_distance | — | 膝盖横向间距 |
| feet_distance | — | 脚间距 |

### 3.3 训练超参数

| 参数 | PAI | HumanoidSW2 |
|------|-----|-------------|
| num_envs | 4096 | 4096 |
| frame_stack | 15 | 15 |
| lr | **1e-5** | 3e-5 |
| entropy | **0.001** | 0.005 |
| action_scale | **0.25** | 0.3 |
| PD kp | 20-40 | 15-60 |
| ctrl_dt | 0.02 | 0.02 |

PAI 使用更保守的超参（更低的 lr、entropy、action_scale），适合这个更轻、更短的机器人。

---

## 四、两个 URDF 版本

| | `pi_12dof_release_v1.urdf` | `pi_12dof_release_v1_rl.urdf` |
|------|--------------------------|------------------------------|
| 用途 | ROS/仿真 | **RL 训练** |
| 踝 roll 范围 | [-0.3, 0.8] (不对称) | **[-0.15, 0.15]** (对称安全) |
| 足部碰撞 | 极小 box | **0.078×0.08×0.05** (真实) |
| MuJoCo 编译器 | 无 | 有 (balanceinertia=true) |

RL 版本特意收窄了踝关节范围并用了真实的足部碰撞几何。

---

## 五、对 XqRobotV2 的借鉴

### 可借鉴
- **内置姿态偏移**：URDF 里的关节 origin rpy 实现静态蹲姿，比代码 `DEFAULT_LEG_ANGLES` 更物理准确
- **RL 专用 URDF**：独立维护一份 URDF，收紧不安全的关节范围，简化碰撞几何
- **更低 learning_rate (1e-5) + entropy_coef (0.001)**：对小质量机器人更稳定
- **踝 roll 极窄限制 (±0.15)**：防止侧倾，对 XqRobotV2 的髋 roll 约束有参考价值

### 不适合借鉴
- 12DOF 含踝关节：XqRobotV2 无踝，步态设计逻辑不同
- 大腿 yaw DOF：XqRobotV2 无此自由度，不能参考
- 髋前屈偏移：XqRobotV2 髋关节只有 roll 轴，pitch 在 thigh 和 calf 上
