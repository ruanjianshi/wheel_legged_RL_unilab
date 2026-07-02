# WTW-NEW (Go1 + AER) 参考文档

> 路径: `/home/robot/xiaoq/projects/wtw_new`
> 框架: Isaac Gym | 机器人: Unitree Go1 (12DOF 四足) | MIT + UC Berkeley

- 论文: "Adaptive Energy Regularization for Autonomous Gait Transition" (arxiv: 2403.20001)
- 项目页: https://sites.google.com/berkeley.edu/efficient-locomotion

---

## 一、项目概述

基于 Walk These Ways (MIT) 的 Go1 四足 RL 项目，核心创新：

1. **自适应能量正则化 (AER)**：用速度归一化的能耗奖励，实现速度变化时自动切换步态
2. **RMA 学生-教师架构**：教师用特权信息训练，学生从观测历史预测环境参数
3. **学习型执行器网络**：从真实 Go1 数据训练的 MLP 替代 PD 控制
4. **步态条件化策略**：直接把步态参数（频率、相位、腾空高度）喂给策略

---

## 二、机器人形态

| 项目 | 值 |
|------|-----|
| 机器人 | Unitree Go1 (12DOF 四足) |
| 腿数 | 4 条，每腿 3 关节 (髋外展 + 髋俯仰 + 膝) |
| 躯干质量 | 4.8 kg |
| 大腿长度 | 0.213m |
| 小腿长度 | 0.213m |
| 足端 | 球体 (半径 0.02m) |

### 关节 (12 DOF)

| 关节 | 轴 | 扭矩 | 范围 |
|------|-----|------|------|
| Hip (髋外展) | X | 33.5 Nm | [-0.8, 0.8] rad |
| Thigh (髋俯仰) | Y | 33.5 Nm | [-1.0, 4.2] |
| Calf (膝) | Y | 33.5 Nm | [-2.7, -0.9] |

---

## 三、核心创新

### 3.1 自适应能量正则化 (AER)

```
energy_reward = exp(-energy_consume / divider)
divider = sigma_lin × |actual_vx| + sigma_ang × |actual_vyaw|
```

能耗除以**实际速度**——速度越快允许越多能耗，速度=0时能耗惩罚最大。效果：自动从步行→小跑→跳跃，不靠预设步态表。

### 3.2 步态条件化观测

策略观测包含 15 维**步态命令**：

| 参数 | 范围 | 含义 |
|------|------|------|
| gait_freq | [2.0, 4.0] Hz | 步频 |
| gait_phase | [0, 1] | 相位偏移 |
| gait_offset, bound, duration | [0, 1] | 步态参数 |
| footswing_height | [0.03, 0.35] m | 摆动足离地高度 |
| stance_width, length | [0.1-0.45] m | 支撑面参数 |
| body_pitch, roll | [-0.4, 0.4] rad | 躯干倾角命令 |

策略学会根据步频/腾空高度自动生成对应步态。

### 3.3 RMA 学生-教师架构

```
Teacher critic: obs + true_privileged_info (friction, restitution) → value
Student actor:  obs + predicted_latent (from adaptation MLP) → action
Adaptation MLP: 30帧观测历史 → 预测 privileged info
```

训练时：教师用真实环境参数，学生用预测值。部署时：只用学生，从 30 帧历史推断环境。

### 3.4 学习型执行器网络

```
[当前位差, 上一次位差, 上上次位差, 当前速度, 上一次速度, 上上次速度]
  → MLP(32→32) → 扭矩
```

从真实 Go1 数据预训练，替代理想 PD。所有 12 个关节共享权重。显著缩小 sim-to-real 差距。

### 3.5 Only-Positive Rewards

```
total_reward = positive_rew × exp(negative_rew / 0.02)
```

负奖励通过指数函数转换为 0→1 之间的乘数，乘到正奖励上。比直接相加更稳定。

---

## 四、训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| num_envs | **4000** | 大规模并行 |
| frame_stack | **30** | 超长历史（用于 RMA） |
| lr | 1e-3 | 自适应 KL 调度 |
| entropy_coef | 0.01 | |
| action_scale | 0.25 | 髋关节再砍半 |
| ctrl_dt | 0.02 (50Hz) | 仿真 200Hz, decimation=4 |
| episode | 20s | |

---

## 五、奖励结构 (完整 28 项)

### 5.1 速度追踪

| 奖励 | 权重 active | 公式 |
|------|-----------|------|
| tracking_lin_vel | +1.0 | `exp(-||cmd_xy - vel_xy||² / 0.25)` |
| tracking_ang_vel | +0.5 | `exp(-(cmd_yaw - vel_z)² / 0.25)` |

### 5.2 姿态约束

| 奖励 | 权重 | 公式 |
|------|------|------|
| lin_vel_z | -0.02 | `vel_z²` |
| ang_vel_xy | -0.001 | `||vel_xy||²` |
| orientation | -5.0 | `||gravity_xy||²` |

### 5.3 足端约束 (核心缺失项)

| 奖励 | 权重 active | 公式 | XqRobotV2 借鉴 |
|------|-----------|------|---------------|
| **feet_slip** | **-0.04** | `Σ[contact × ||foot_vel_xy||²]` | **轮子着地时不应滑动** |
| **tracking_contacts_shaped_force** | 0.0 | `Σ[-(1-desired_contact)×(1-exp(-||force||²/σ))]` | **摆动相检测地面接触→扣分** |
| **tracking_contacts_shaped_vel** | 0.0 | `Σ[-desired_contact×(1-exp(-||foot_vel||²/σ))]` | **支撑相检测足端移动→扣分** |
| **feet_clearance_cmd_linear** | 0.0 | `Σ[(target_height-foot_z)² × (1-contact)]` | **摆动相跟踪抬腿高度** |
| feet_impact_vel | 0.0 | `Σ[contact × clip(-foot_vz, 0)²]` | 轮子着地冲击 |
| feet_contact_forces | 0.0 | `Σ[clip(||force||-100, 0)]` | 过大的接触力 |

### 5.4 关节约束

| 奖励 | 权重 | 公式 |
|------|------|------|
| torques | -0.0001 | `||τ||²` |
| dof_vel | -1e-4 | `||dof_vel||²` |
| dof_acc | -2.5e-7 | `||acc||²` |
| **dof_pos_limits** | **-10.0** | `clip(超出90%限位)` |

### 5.5 平滑约束

| 奖励 | 权重 | 公式 |
|------|------|------|
| action_rate | -0.01 | `||Δaction||²` |
| action_smoothness_1 | -0.1 | 1阶差分 |
| action_smoothness_2 | -0.1 | 2阶差分 (jerk) |

### 5.6 碰撞

| 奖励 | 权重 | 公式 |
|------|------|------|
| collision | **-5.0** | 大腿/小腿触地次数 |

### 5.7 步态 / 接触模式 (全部 0.0, 未启用的 18 项)

还包括: `feet_air_time`, `feet_clearance`, `tracking_stance_width/length`, `hop_symmetry`, `stand_still`, `tracking_contacts`, `feet_stumble`, `jump`, `energy`, `raibert_heuristic` 等，**全部 weight=0**。

### 5.8 能耗 (AER 核心, 仅 AdaptiveConfig 启用)

| 奖励 | 公式 |
|------|------|
| energy_new_actual | `exp(-Σ|vel×τ| / (1000×|vel|+500×|vyaw|))` |
| energy_new_cmd | 同上, 但分母用命令速度 |

### 5.9 奖励组合方式

```
total = pos_sum × exp(neg_sum / 0.02)
```

负奖励通过指数映射到 (0,1] 乘数，乘到正奖励上。比直接加权和更稳定。

---

## 六、对 XqRobotV2 toe_walk 的关键借鉴

WTW-NEW 的 28 项奖励中，XqRobotV2 当前缺失的**最关键三项**：

### 1. 摆动相触地检测 → `tracking_contacts_shaped_force`
**当前 XqRobotV2**: `swing_lift` 只奖"腿动了"，不检查"轮子是否离地"
**应该有**: 摆动相检测轮子是否还在地上 → 如果在，重罚

### 2. 支撑相滑动惩罚 → `feet_slip`  
**当前**: 无
**应该有**: 着地轮子不应滑移 → `contact × ||wheel_vel||²`

### 3. 关节限位 → `dof_pos_limits`
**当前**: 无
**应该有**: 大腿/小腿到达极限 → 重罚 (-10.0)

这三项直接对应于 XqRobotV2 toe_walk 的"腿摆动了但没离地"问题。

---

## 七、训练超参数

### 可直接用

| 技术 | 方法 | 难度 |
|------|------|------|
| **步态条件化观测** | 把步频/相位/腾空高度喂给观测，策略自动生成对应步态 | 低 (改观测) |
| **Only-positive rewards** | `pos_rew × exp(neg_rew / sigma)` 替代直接相加 | 低 (改 `run_reward_dispatch`) |
| **能耗正则化** | `exp(-torque²/speed)` 压能耗，但绝对速越快越宽容 | 中 (加 reward) |
| **大 frame_stack** | 9→30 帧（需 RMA 支撑才有意义） | 中 |

### 需要配套基础设施

| 技术 | 前提条件 |
|------|---------|
| RMA 学生-教师 | 有真实特权信息 + 大量观测历史 |
| 学习型执行器 | 有真实机器人执行器数据 |
| 步态 curriculum | 有 gait 参数空间 + 多 gait 训练 |

### 不适合借鉴

- Go1 是四足，XqRobotV2 是双足，步态参数空间不同
- RMA 需要特权信息（摩擦、质量），当前 UniLab 无此通道
- 能量奖励需要扭矩传感器（当前 XML 可能没有）
