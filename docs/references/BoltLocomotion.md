# BoltLocomotion 核心机制深度分析

> 项目路径: `/home/robot/xiaoq/projects/BoltLocomotion`
> 框架: Isaac Lab + CleanRL PPO | 机器人: ODRi Bolt (6DOF 点足双足)
> 论文: "Constrained Reinforcement Learning for Unstable Point-Feet Bipedal Locomotion Applied to the Bolt Robot" (Humanoids 2025)

本文档聚焦 CaT (Constraints as Terminations) 框架的核心机制及其与常规 RL 方法的本质差异。

---

## 一、项目核心创新: CaT —— 约束即终止

### 1.1 问题: 传统奖励函数的问题

常规足式机器人 RL (如 rsl-rl 系列) 用 **10-20 个奖励项** 的加权和作为优化目标:

```
r = w₁·tracking + w₂·torque_penalty + w₃·orientation + w₄·action_rate + ...
```

这有 3 个核心问题:
1. **权重调优困难** — 10+ 个权重的手动调参是组合爆炸
2. **软约束不可靠** — 奖励惩罚只是"建议"，策略可能学会忍受惩罚而违反硬约束
3. **奖励尺度耦合** — 速度跟踪 (0~1) 和力矩惩罚 (0~100 Nm) 在同一个加和中，需要精细的 scale 匹配

### 1.2 CaT 的解法: 约束 → 概率化提前终止 + 乘性奖励惩罚

```python
# 传统方式 (rsl-rl):
r_total = 1.0 * tracking_vel - 0.01 * torque - 5.0 * orientation_error - ...

# CaT 方式:
cstr_prob = max(p₁, p₂, ..., p₁₃)        # 所有约束的最大违规概率
r_total = clip(r_task * (1 - cstr_prob), 0, ∞)  # 任务奖励被约束概率乘性缩放
# 同时: 以概率 cstr_prob 触发提前终止
```

**核心思想**: 约束不被优化进奖励函数，而是作为**独立的安全门控**。当约束违规时:
- 任务奖励被缩放 (乘性惩罚)
- 环境有可能提前终止 (概率化终止)

### 1.3 为什么要"概率化"而非"确定化"？

确定化终止 (`if violation: terminate()`) 在训练初期会导致**所有 episode 瞬间终止**，策略收不到任何有效学习信号。概率化 (`p_terminate = clamp(violation, 0, 1)`) 允许:
- 轻度违规: 低终止概率 → 给策略修正机会
- 严重违规: 高终止概率 → 快速剔除不良行为
- 违规概率通过**课程学习**从宽松 (p_max=0.05) 逐步收紧到严格 (p_max=0.25)

---

## 二、机器人: ODRi Bolt 点足双足

### 2.1 形态

```
base_link (躯干, 0.62kg)
  ├── FL_HAA (髋外展  ±0.9 rad)
  │     └── FL_HFE (髋屈伸  ±1.7 rad)
  │           └── FL_KFE (膝屈伸  ±3.4 rad)
  │                 └── FL_FOOT (点足, 1.3g — 几乎是零质量)
  └── FR_HAA / FR_HFE / FR_KFE → FR_FOOT (右侧镜像)

总质量: ~1.35 kg  |  6 个关节 (3 每腿)  |  点足 (无踝关节)
```

### 2.2 关键特征

| 特征 | Bolt | 常见四足 (Go1/Go2) | XqRobotV2 |
|------|------|-------------------|-----------|
| **腿数** | 2 | 4 | 2 |
| **足类型** | **点足** (质量 1.3g) | 有缓冲足 | **轮足** |
| **总 DOF** | 6 (HAA+HFE+KFE × 2) | 12 (HAA+HFE+KFE × 4) | 8 (3 腿 + 1 轮 × 2) |
| **总质量** | 1.35 kg | 12 kg | ~5 kg |
| **踝关节** | **无** (点接触) | 无 (刚性足) | 无 (轮) |
| **PD 刚度** | Kp=4.0, Kd=0.2 | Kp=20-60, Kd=0.5-1.5 | Kp=30 (腿), Kv=1 (轮) |
| **控制频率** | 50 Hz | 50 Hz | 100 Hz |

**为什么用点足？** Bolt 的脚是 1.3g 的零质量点——物理上几乎不贡献惯性。配合极低 PD 刚度 (Kp=4.0) 和 100 Nm 力矩上限，使机器人可以通过**欠驱动动力学**实现自然步态，而不是用强 PD 跟踪硬轨迹。

### 2.3 默认站立姿态

```python
default_joint_pos = {
    "FL_HAA": 0.0,         # 髋外展: 中性
    "FL_HFE": 0.398,       # 髋屈: 微前倾 (~22.8°)
    "FL_KFE": -0.691,      # 膝: 弯曲 (~39.6°)
    # 右侧完全镜像
}
```

策略输出的是**相对于默认姿态的偏移**，action_scale=0.5 → PD 目标 ∈ [默认-0.5, 默认+0.5] rad。

---

## 三、约束系统: 13 个约束项的完整设计

### 3.1 约束分类

| 类型 | max_p | 约束项 | 作用 |
|------|-------|--------|------|
| **硬约束** | 1.0 | `upsidedown` | 躯干翻转 → 100% 立即终止 |
| | 1.0 | `contact` | 躯干/大腿触地 → 100% 立即终止 |
| | 1.0 | `foot_contact_force` | 点足受力 > 50N → 100% 立即终止 |
| **软约束** | 0.25 (课程) | `joint_torque` | 力矩 > 4 Nm |
| | 0.25 | `joint_velocity` | 关节速度 > 16 rad/s |
| | 0.25 | `joint_acceleration` | 关节加速度 > 800 rad/s² |
| | 0.25 | `action_rate` | 动作变化率 > 90 rad/s |
| **风格约束** | 0.25 (课程) | `base_orientation` | 躯干倾斜 > 0.1 rad (~5.7°) |
| | 0.25 | `air_time` | 腾空时间 < 0.25s (避免拖步) |
| | 0.25 | `one_foot_contact` | 单脚接触 (强制行走步态) |
| | 0.25 | `HAA_position` | 髋外展 > 0.3 rad |
| | 0.25 | `HFE_position` | 髋屈伸 > 0.5 rad |
| | 0.25 | `KFE_position` | 膝屈伸 > 0.5 rad |

### 3.2 违规 → 终止概率的计算

```python
# 为每个约束单独计算违规值和概率

# 1. 计算违规值
def constraint_violation(state, limit):
    # 用指数移动平均 (tau=0.95) 维护每个约束的最大违规值
    self.max_violation[constraint_i] = max(
        self.max_violation[constraint_i] * 0.95,
        abs(current_value - limit)
    )

# 2. 归一化违规值
normalized = clamp(violation / max_violation, 0, 1)

# 3. 计算终止概率
p_i = min_p + normalized * (max_p - min_p)
# min_p = 0.0, max_p = 课程值 (0.05 → 0.25)

# 4. 合并所有约束 → 取并集
cstr_prob = max(p_1, p_2, ..., p_13)

# 5. 用约束概率缩放奖励
reward = task_reward * (1.0 - cstr_prob)
```

**关键设计**:
- 违规值用**指数滑动最大**做归一化 → 自动适应不同数量级的违规 (力矩 Nm vs 角度 rad vs 加速度 rad/s²)
- `min_p = 0` → 无违规时不影响奖励和终止
- 并集 (`max`) 而非交集 → 任何一个约束违规都会触发惩罚，策略必须同时满足所有约束

### 3.3 与传统奖励惩罚的对比

```
传统: r = tracking - 0.01*torque - 5.0*orientation
        问题: 当 tracking=0.8, orientation_error=0.5 时
        r = 0.8 - 0 - 1.25 = -0.45  ← 策略可能学会"仰面躺下"来优化这个

CaT:  r = tracking * (1 - p_orientation)
        当 违规轻微 (p=0):     r = 0.8 * 1.0 = 0.8   ← 不受影响
        当 违规中等 (p=0.1):   r = 0.8 * 0.9 = 0.72  ← 温和惩罚
        当 违规严重 (p=0.25):  r = 0.8 * 0.75 = 0.6  ← 显著惩罚
        且可能被提前终止 → 直接归零未来奖励
```

---

## 四、约束课程学习: max_p 从 0.05 到 0.25 的退火

### 4.1 为什么需要课程

- 训练初期: 策略不会控制机器人 → 所有约束大量违规 → 若 max_p 很大，所有 episode 瞬间终止 → 学习停滞
- 训练后期: 策略已经基本稳定 → 需要收紧约束 → 精修关节范围和姿态

### 4.2 课程函数

```python
# 期望的 episode 长度从 T_start=20s 线性过渡到 T_end=4s
T_start = 20.0     # 初始：期望 episode 长达 20s (最大长度)
T_end = 4.0        # 最终：期望 4s (即 1/0.25)

progress = min(steps / 24000, 1.0)          # 0→1 线性
max_p = 1 / (T_start + progress * (T_end - T_start))

# progress=0.0: max_p = 1/20 = 0.05     ← 很宽松
# progress=0.5: max_p = 1/12 = 0.083
# progress=1.0: max_p = 1/4  = 0.25    ← 严格
```

这是一个**双曲线退火** — `max_p = 1/expected_ep_len`。含义：随着训练推进，策略应该能在越来越短的时间内避免违规。课程不是改 terrain 或 command range，而是**改约束的严格程度**。

---

## 五、奖励函数: 只有 2 项

### 5.1 track_lin_vel_xy_exp (权重 1.0)

```python
r = exp(-‖v_xy_cmd - v_xy_actual‖² / 0.25)
# 即: r = exp(-4 * ‖error‖²)
```

### 5.2 track_ang_vel_z_exp (权重 0.5)

```python
r = exp(-(ω_z_cmd - ω_z_actual)² / 0.25)
```

### 5.3 总奖励

```python
r_total = clip( (1.0*track_lin + 0.5*track_ang) * (1 - max(all_constraint_probs)), 0, ∞ )
```

**为什么只有 2 项？** 因为约束系统接管了所有"安全行为"的监督。力矩、关节速度、姿态、接触模式等都由约束保证，不需要在奖励里重复惩罚。奖励只需要表达"走得快"这一个目标。

---

## 六、观测空间: 5 步历史堆叠

### 6.1 组成

```
单帧观测 (21D):
  base_ang_vel (3D)          躯干角速度 [roll_rate, pitch_rate, yaw_rate]
  velocity_commands (3D)     线速度指令 [vx, vy] + 角速度指令 [vyaw]
  projected_gravity (3D)     重力投影 [gx, gy, gz] (表征躯干姿态)
  joint_pos (6D)             [HAA_L, HFE_L, KFE_L, HAA_R, HFE_R, KFE_R]
  joint_vel (6D)             关节速度 × 6
  last_action (6D)           上一帧动作

历史堆叠: 5 帧 × 间隔 2 步
  t-0, t-2, t-4, t-6, t-8 → 总共回溯 8×0.02=0.16s 的历史

总维度: 21 × 5 = 105D
```

### 6.2 与 UniLab XqRobotV2 观测对比

| 维度 | Bolt (CaT) | XqRobotV2 |
|------|-----------|-----------|
| 历史帧数 | 5 (间隔 2 步) | 9 (连续) |
| 回溯时间 | 0.16s | 0.09s |
| 总维度 | 105 | 297 (actor) / 511 (critic) |
| last_action 注入 | ✅ | ✅ |
| 重力投影 | 3D | 3D (从 `upvector` 传感器) |
| 指令注入 | 3D [vx, vy, vyaw] | 5D [vx, vy, vyaw, tsk, height] |
| 地面高度扫描 | ❌ | ✅ (critic 独有, 187D) |
| 观测归一化 | ✅ RunningMeanStd | ❌ |

### 6.3 观测噪声

| 观测项 | 噪声类型 | 强度 |
|--------|---------|------|
| base_ang_vel | Additive Uniform | ±0.2 rad/s |
| projected_gravity | Gaussian 加性 + Uniform 偏置 | σ=0.05, bias=±0.05 |
| joint_pos | Uniform 加性 + Uniform 偏置 | ±0.01 rad, bias=±0.05 |
| joint_vel | Additive Uniform | ±1.5 rad/s |

噪声策略比 UniLab 更激进 (joint_pos bias ±0.05 rad ≈ 2.8°, 等价于一个大的静态安装误差)。

---

## 七、域随机化

| 参数 | 范围 | 说明 |
|------|------|------|
| 刚体质量 | [0.8, 1.2] × 默认 | 所有身体 |
| 惯性张量 | [0.8, 1.2] × 默认 | 所有身体 |
| 质心偏移 | ±0.02 m | 仅躯干 |
| 关节摩擦 | [0.01, 0.1] | 绝对值 (非乘数) |
| 接触摩擦 | static [0.4, 1.5], dynamic [0.4, 1.5] | 所有机器人刚体 |
| 初始位姿 | x,y∈[-0.5,0.5], yaw∈[-π,π] | 随机起始位置 |
| 初始速度 | lin[-0.3,0.3], ang[-0.1,0.1] | 随机初速 |
| 推力扰动 | 6DOF 随机 pertubation，每 5-8s | base_link |

**特点**: DR 在 env 创建时一次性采样（不随时间变化）。不涉及动力学参数的 episode 内重采样。

---

## 八、网络与 PPO 训练

### 8.1 网络架构

```
Actor mean:  Linear(105, 512) → ELU → Linear(512, 256) → ELU → Linear(256, 128) → ELU → Linear(128, 6)
Actor std:   learnable Parameter [1, 6], 初始化为 0 → log_std
Critic:      Linear(105, 512) → ELU → Linear(512, 256) → ELU → Linear(256, 128) → ELU → Linear(128, 1)

权重初始化: Orthogonal (hidden std=√2, critic_output std=1.0, actor_output std=0.01)
```

### 8.2 PPO 超参

| 参数 | Bolt (CaT) | UniLab XqRobotV2 |
|------|-----------|-------------------|
| num_envs | 4096 | 512-1024 |
| steps_per_env | 24 | 25 |
| total_iters | 2000 | 5000-20000 |
| total_timesteps | **196M** | 128M-256M |
| learning_rate | 3e-4 | 1e-4 |
| gamma | 0.99 | 0.99 |
| lambda | 0.95 | 0.95 |
| clip_param | 0.2 | 0.2 |
| n_epochs | 5 | 5 |
| minibatch_size | 16384 | 自动 (1/4 of batch) |
| entropy_coeff | 0.001 | 0.002 |
| value_loss_coeff | 2.0 | 1.0 |
| max_grad_norm | 1.0 | 1.0 |
| 观测归一化 | RunningMeanStd (Welford) | 无 |
| value 归一化 | RunningMeanStd | 无 |

### 8.3 单次 PPO 更新

```python
# CleanRL 风格 (与 rsl-rl 不同)

# 1. 归一化观测
obs = (obs - obs_rms.mean) / obs_rms.std

# 2. 计算动作分布
mean = actor(obs)
std = exp(log_std)
dist = Normal(mean, std)
log_prob = dist.log_prob(action).sum(-1)

# 3. PPO clipped loss
ratio = exp(log_prob - old_log_prob)
loss = -min(ratio * adv, clip(ratio, 0.8, 1.2) * adv).mean()
loss += 0.5 * 2.0 * (returns - value)^2.mean()   # value_loss_coeff=2.0
loss -= 0.001 * dist.entropy().mean()              # entropy_coeff=0.001

# 4. 单步更新 (所有 losses 在同一 backward pass)
```

---

## 九、地形系统

### 9.1 配置

```python
ROUGH_TERRAINS_CFG:
  num_rows = 10, num_cols = 20        # 200 个子地形
  size = (8.0, 8.0)                   # 每个 8×8m
  horizontal_scale = 0.1              # 10cm 水平分辨率
  vertical_scale = 0.005              # 高度图缩放

  sub_terrains:
    random_rough (100%):              # 只有一种地形
      noise_range = [0, 0.02]         # 最大 2cm 高度起伏
      noise_step = 0.005              # 5mm 垂直分辨率
```

**为什么只有一种地形？** 对于点足双足机器人，2cm 的随机起伏已经足够困难——脚只有 1.3g，放在凸起上极易打滑。而四足 (如 Go2) 需要台阶、斜坡、离散障碍等多样化地形来避免过拟合。

---

## 十、训练启动

```bash
# 训练
python scripts/clean_rl/train.py --task=Isaac-Velocity-CaT-Bolt-v0 --headless

# 播放 + 导出 ONNX/TorchScript
python scripts/clean_rl/play.py --task=Isaac-Velocity-CaT-Bolt-Play-v0 --headless --video --video_length 200
```

---

## 十一、与 UniLab XqRobotV2 的对比

| 维度 | BoltLocomotion (CaT) | UniLab XqRobotV2 |
|------|---------------------|-------------------|
| **奖励项数** | **2** (纯速度跟踪) | 13 (速度 + 姿态 + 动作 + 对称 + ...) |
| **安全约束** | 13 个概率化终止 + 乘性奖励惩罚 | 负奖励项 (软约束) + 硬终止 |
| **约束策略** | 独立门控，不被优化进梯度 | 是梯度优化目标的一部分 |
| **权重调优** | 只调 2 个奖励权重 | 调 13 个权重 |
| **课程** | 约束严格度退火 (max_p: 0.05→0.25) | 速度命令范围扩展 (vel_step) |
| **机器人** | 双足点足 (1.35kg, 6DOF) | 轮腿双足 (~5kg, 8DOF) |
| **框架** | Isaac Lab + CleanRL | MuJoCo + rsl-rl |
| **地形** | 单一随机粗糙 (0-2cm) | 6 种 (平/台阶/斜坡/波浪/随机/反斜坡) |
| **观测归一化** | RunningMeanStd (obs + value) | 无 |
| **网络架构** | [512,256,128] ELU, Orthogonal init | [512,512,256,128] ELU, 默认 init |
| **历史堆叠** | 5 帧间隔 (0.16s 回溯) | 9 帧连续 (0.09s 回溯) |

---

## 十二、可移植到 UniLab 的改进

### 优先级 1: 约束系统 (CaT)

**可行性**: 需要一个 `ConstraintManager` 和概率化终止逻辑，但核心算法简单。

**方案**:
```python
# 1. 每个约束定义违规函数 + max_p
constraints = [
    Constraint("joint_torque", fn=abs(torque) - 4.0, max_p=0.25),
    Constraint("base_orientation", fn=norm(gravity_xy) - 0.1, max_p=0.25),
    ...
]

# 2. 每步计算违规概率
norms = [ema_max(violation) for violation in violations]
probs = [n * max_p for n in norms]
cstr_prob = max(probs)

# 3. 乘性奖励缩放 + 概率化终止
reward = reward * (1.0 - cstr_prob)
if random() < cstr_prob: terminate()
```

**收益**: 减少需要调优的奖励权重数量 (从 13 个到 2-3 个)，同时强化硬安全约束。

### 优先级 2: 观测 + Value 归一化

Bolt 使用 RunningMeanStd 做观测和 value target 双重归一化。UniLab 的 `empirical_normalization` 已支持，但从未启用。

### 优先级 3: 间隔历史堆叠

Bolt 的 5 帧 × 间隔 2 步 (回溯 0.16s) 比 UniLab 的 9 帧连续 (回溯 0.09s) 更有效地捕捉低速动态。对 rough_walk 的崎岖地形感知可能有帮助。

---

## 十三、关键文件索引

| 机制 | 文件 |
|------|------|
| 环境配置 (全部参数) | `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/bolt/cat_env_cfg.py` |
| PPO 超参 | `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/bolt/agents/clean_rl_ppo_cfg.py` |
| CaT 约束管理器 (概率计算) | `exts/cat_envs/cat_envs/tasks/utils/cat/constraint_manager.py` |
| 13 个约束函数实现 | `exts/cat_envs/cat_envs/tasks/utils/cat/constraints.py` |
| 约束课程函数 | `exts/cat_envs/cat_envs/tasks/utils/cat/curriculums.py` |
| CaTEnv 主类 | `exts/cat_envs/cat_envs/tasks/utils/cat/cat_env.py` |
| 自定义观测项 | `exts/cat_envs/cat_envs/tasks/utils/mdp/observations.py` |
| 自定义终止项 | `exts/cat_envs/cat_envs/tasks/utils/mdp/terminations.py` |
| 历史堆叠观测管理器 | `exts/cat_envs/cat_envs/tasks/utils/history/observation_manager.py` |
| CleanRL PPO 实现 | `exts/cat_envs/cat_envs/tasks/utils/cleanrl/ppo.py` |
| 训练入口 | `scripts/clean_rl/train.py` |
| 播放/导出 | `scripts/clean_rl/play.py` |
| 地形生成 | `exts/cat_envs/cat_envs/assets/terrains/rough.py` |
| 机器人关节配置 | `exts/cat_envs/cat_envs/assets/odri.py` |
| URDF 模型 | `exts/cat_envs/cat_envs/assets/Robots/odri/bolt_description/urdf/bolt_description_isaac.urdf` |

---

*文档生成: 2026-07-08*
