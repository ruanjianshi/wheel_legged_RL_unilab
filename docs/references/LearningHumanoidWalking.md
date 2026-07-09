# LearningHumanoidWalking 核心机制深度分析

> 项目路径: `/home/robot/xiaoq/projects/LearningHumanoidWalking`
> 框架: MuJoCo + PPO + Ray | 机器人: JVRC (12DOF), Unitree H1 (10DOF)
> 论文: IEEE-RAS Humanoids 2024 / IEEE Access 2023 / Humanoids 2022

本文档对步态行走的**全部实现机制**做深度拆解，面向后续移植和改进参考。

---

## 一、行走是如何实现的？——完整的机制栈

**关键认知**: 步态相位时钟只是"时空坐标"，单凭它不能产生行走。行走是以下 **7 层机制叠加作用**的结果：

```
第 7 层: 奖励函数        ← 塑造行为方向（快慢、高低、正反）
第 6 层: 相位时钟        ← 提供"现在该迈哪只脚"的节奏信号
第 5 层: 动作平滑        ← 阻止高频抖动破坏步态
第 4 层: 镜像对称        ← 强制左右腿协调一致
第 3 层: 域随机化        ← 防止过拟合某一套动力学参数
第 2 层: 模式课程        ← 先学会站立再学行走的渐进路径
第 1 层: PD 位置控制     ← 将策略输出（位置目标）转为物理力矩
基底层: 半蹲位姿          ← 提供"零位"——所有关节归零 = 稳定微蹲
```

每一层缺一不可。去掉任何一层，步态质量都会退化。

---

## 二、第 1 层：PD 位置控制 + 半蹲位姿

### 2.1 为什么用位置控制而不是力矩控制？

策略输出的是**关节位置目标的偏移量**，不是力矩。这使动作空间有界且物理可解释——策略说"膝盖弯曲 50°"，PD 控制器负责"用力矩把膝盖推到 50°"。

```
策略输出 (12D, 每关节相对半蹲位的偏移)
  │
  ▼
target = 半蹲位姿 + 策略偏移
  │
  ▼
τ = Kp × (target - θ_current) + Kd × (0 - ω_current)     ← PD 控制律
  │
  ▼
τ_net = τ - τ_d × ω                                       ← 反向电动势模拟
  │
  ▼
τ_motor = τ_net / gear_ratio                              ← 减速比换算
  │
  ▼
mj_data.ctrl = τ_motor                                    ← 施加到物理引擎
```

### 2.2 半蹲位姿（half-sitting pose）

**JVRC 配置** (`envs/jvrc/configs/base.yaml`):

```yaml
half_sitting_pose: [-30, 0, 0, 50, 0, -24]    # 单位: 度，左右腿各 6 关节
#  关节顺序: HIP_P, HIP_R, HIP_Y, KNEE, ANKLE_R, ANKLE_P
```

含义:
- 髋俯仰 (HIP_P) = -30° → 身体微前倾，重心落在支撑域内
- 膝 (KNEE) = 50° → 弯曲站立，有弹簧储能空间
- 踝俯仰 (ANKLE_P) = -24° → 脚掌微翘，触地前有缓冲

**设计原理**: 策略以这个姿态为"零位"（action=0 时 PD 目标 = 半蹲位姿），确保策略输出为 0 时机器人处于稳定微蹲态，不会直接倒下。这是 RL 从零开始训练能快速站稳的物理基础。

### 2.3 PD 增益

```yaml
kp: [200, 200, 200, 250, 80, 80,      # 右腿 × 6
     200, 200, 200, 250, 80, 80]      # 左腿 × 6
kd: [20,  20,  20,  25,  8,  8,
     20,  20,  20,  25,  8,  8]
```

- 膝关节最硬 (Kp=250, Kd=25) —— 承载主要体重
- 踝关节最软 (Kp=80, Kd=8) —— 适应地面起伏
- 髋关节中等 (Kp=200, Kd=20) —— 平衡跟踪精度和顺应性

### 2.4 反向电动势（Back-EMF）

```python
τ_net = τ_PD - τ_d × ω_current
# τ_d: 随机采样的阻尼系数 [5, 40], 每约 2.5s 重新采样
```

模拟真实电机的物理效应——电机转动越快，反电动势越高，有效力矩越低。对 **Sim-to-Real** 至关重要。

---

## 三、第 2 层：步态相位时钟

### 3.1 相位状态机

```python
# 初始化
period = 2 × (swing_duration + stance_duration) × FREQ
#      = 2 × (0.75 + 0.35) × 40 = 88 控制步
phase = random(0, 88)     # 随机初始相位

# 每控制步
phase = (phase + 1) % 88
```

**一个完整步态周期的物理意义**:

```
阶段 0-29   (0.75s)  → 右腿摆动，左脚支撑
阶段 30-43  (0.35s)  → 双腿支撑（过渡）
阶段 44-73  (0.75s)  → 左腿摆动，右脚支撑
阶段 74-87  (0.35s)  → 双腿支撑（过渡）
```

### 3.2 时钟信号 → 策略观测

```python
clock = [sin(2π·phase/88), cos(2π·phase/88)]
```

用单位圆上的 (sinθ, cosθ) 编码相位。为什么不用 phase/88 标量？因为 sin/cos 是连续循环的——88 步后回到同一个点，不需要处理 88→0 的跳变。神经网络天然理解"靠近"+1 就是"靠近 -1"。

### 3.3 脚底力/速度目标曲线（PCHIP 样条）

对每只脚定义**两条时钟函数**：`force_clock(phase)` 和 `velocity_clock(phase)`，用 8 个控制点 + PCHIP 三次插值生成平滑的周期性函数。

**以右脚力时钟为例**:

```
控制点 x (phase):  [0,   30,  30,  44,  44,  74,  74,  88]
控制点 y (力目标):  [-1,  -1,  +1,  +1,  +1,  +1,  +1,  +1]

解读:
  phase ∈ [0,30)    y=-1 → "力应该为零"   (右脚在空中，摆动相)
  phase ∈ [30,44)   y=+1 → "力应该达到最大" (右脚着地，支撑相)
  phase ∈ [44,74)   y=+1 → "力应该达到最大" (左脚摆动，右脚是支撑腿)
  phase ∈ [74,88]   y=+1 → "力应该达到最大" (双腿支撑)
```

**右脚速度时钟**:

```
控制点 y (速度目标): [+1, +1, -1, -1, -1, -1, -1, -1]
解读:
  phase ∈ [0,30)    y=+1 → "速度应该最大" (右脚在空中移动，摆动相)
  phase ∈ [30,44)   y=-1 → "速度应为零"   (右脚着地静止，支撑相)
  ...
```

**左脚时钟 = 右脚时钟的 180° 平移** (phase+44 取模)。

**PCHIP vs 线性插值**: PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) 保证单调性——在 control point 间不会产生 overshoot。在摆动→支撑过渡区 (phase≈30)，PCHIP 产生 S 形平滑过渡。

### 3.4 力/速度时钟的互补性——核心设计

```
力时钟 r_frc_fn(phase):              速度时钟 r_vel_fn(phase):
 +1  ████████     ████████████████     +1  ████████                    ████████
     │        \   /                │       │        \                  /
     │         \ /                 │       │         \                /
  0  ──────────X────────────────── │    0  ──────────X─────────────── ────────
     │         / \                 │       │         / \             /
 -1  ─────────    ────────────────      -1  ─────────    ────────────────
     0    30   44    74    88              0    30   44    74    88
     摆动  双撑  摆动  双撑                  摆动  双撑  摆动  双撑
```

**关键**: 力和速度时钟在任何 phase 上都**互斥互补**——同一时刻，恰好一个是 +1（"做这个"），另一个是 -1（"不做那个"）。这物理地强制执行了"摆动相脚快速移动 + 不受力，支撑相脚静止 + 受力"的步态约束。这是双足行走最底层的物理事实。

---

## 四、第 3 层：奖励函数

### 4.1 核心奖励：脚底力/速度对齐

```python
def foot_frc_clock_reward(force, phase, clock_fn, robot_mass):
    # 1. 归一化力: 除以半体重 → [0,1] → 映射到 [-1,1]
    max_frc = robot_mass × 9.8 × 0.5
    normed = clip(force, 0, max_frc) / max_frc
    normed = normed × 2 - 1                       # → [-1, +1]

    # 2. 时钟目标值
    target = clock_fn(phase)                       # → [-1, +1]

    # 3. 对齐评分
    score = tan(π/4 × target × normed)

# 速度时钟同理，只是把 force 换成 foot_velocity_norm
```

**tan(π/4 × x) 评分矩阵**:

| target(来自时钟) | normed(实际值) | 积 | tan(π/4·积) | 含义 |
|:--:|:--:|:--:|:--:|------|
| +1 (该有力) | +1 (有力) | +1 | **+1.0** | 完美匹配 ✅ |
| +1 (该有力) | -1 (无力) | -1 | **-1.0** | 严重违反 ❌ |
| -1 (不该有力) | -1 (无力) | +1 | **+1.0** | 完美匹配 ✅ |
| -1 (不该有力) | +1 (有力) | -1 | **-1.0** | 严重违反 ❌ |

**为什么用 tan 而不是 MSE**？因为 tan 是**带符号**的对齐评分——它区分了"应该有力时无力"和"不该有力时有力"。MSE 同为 1，无法区分。这种"方向感知"让策略能在摆动相迅速撤力，支撑相迅速发力。

### 4.2 完整奖励权重

```python
reward = {
    "foot_frc_score":      0.225 × 脚力对齐,      # 步态匹配 45%
    "foot_vel_score":      0.225 × 脚速对齐,
    "com_vel_error":       0.150 × 速度跟踪,      # 速度跟踪 30%
    "yaw_vel_error":       0.150 × 偏航跟踪,
    "root_accel":          0.050 × 躯干平稳,      # 姿态正则 20%
    "height_error":        0.050 × 高度保持,
    "upper_body_reward":   0.050 × 上半身对齐,
    "posture_error":       0.050 × 关节姿态,
    "torque_penalty":      0.025 × 力矩平滑,      # 动作平滑 5%
    "action_penalty":      0.025 × 动作平滑,
}
```

**权重设计逻辑**:
- 75% 给"走好步态"（脚底力/速 + 速度跟踪）——这是**硬约束**
- 20% 给"稳定躯体"——这是**软约束**
- 5% 给"动作平滑"——这是**正则项**

这个分配比 UniLab 更集中（UniLab 用 13 项，速度跟踪仅 25%）。

### 4.3 速度跟踪的特殊设计

```python
# 前向速度: exp(-10 × ‖v - v_ref‖²)    ← 二次衰减，温和
# 偏航速度: exp(-10 × |ω - ω_ref|³)    ← 三次衰减，锐利

# 高度保持: 带死区的二次衰减
deadzone = 0.01 + 0.05 × 当前速度      ← 走得越快死区越大
if |h - h_target| < deadzone: error = 0
reward = exp(-40 × error²)
```

---

## 五、第 4 层：镜像对称

### 5.1 镜像排列

12 个腿关节的原始顺序: `[R_HP, R_HR, R_HY, R_KN, R_AR, R_AP, L_HP, ...]`

镜像操作:
```
mirror_obs = [
    -0, +1,                             # roll→取反, pitch→保持
    -2, +3, -4,                         # wx→取反, wy→保持, wz→取反
    +11, -12, -13, +14, -15, +16,       # 左腿位姿↔右腿(带符号取反)
    +5,  -6,  -7,  +8,  -9,  +10,       # 右腿位姿↔左腿
    +23, -24, -25, +26, -27, +28,       # 左腿速度↔右腿
    +17, -18, -19, +20, -21, +22        # 右腿速度↔左腿
]
```

镜像规则: **矢状面关节** (俯仰/膝/踝俯仰) 符号不变；**冠状面关节** (滚转/偏航/踝滚转) 符号取反。

### 5.2 Mirror Loss

```python
mirror_loss = 0.4 × ‖policy(obs) - mirror(policy(mirror(obs)))‖²
```

强制策略满足"左腿行为 = 镜像翻转后的右腿行为"。如果没有这个 loss，策略可能学出不对称步态（如身体倾斜或单腿跛行）。

---

## 六、第 5 层：动作平滑

```python
target = 0.5 × action + 0.5 × prev_action     # 指数移动平均
```

作用: 低通滤波。防止策略输出高频抖动（相邻步的目标位置跳跃），后者经过 PD 控器会变成高频力矩，直接破坏行走的稳定性。

---

## 七、第 6 层：域随机化

### 7.1 动力学随机化

```yaml
dynamics_randomization:
  interval: 500ms          # 每 500 个物理步 (0.5s) 重采样
  joint_friction: [0, 2.0]
  joint_damping:  [0.02, 2.0]
  body_mass:      [0.95, 1.05]
  body_com:       [-0.01, 0.01]
```

**关键差异**: 随机化发生在 episode **内部**，不是仅在 reset 时。这迫使策略**连续适应**变化的动力学，而不是"记住这一个 episode 的参数是多少"。

### 7.2 扰动推力

```python
# 每 5 秒在 pelvis/torso_link 上施加随机力
data.xfrc_applied[:3] = random_uniform(-10, 10, 3)    # 推力 ±10N
data.xfrc_applied[3:] = random_uniform(-2, 2, 3)     # 扭矩 ±2Nm
# 50% 概率撤销（零力步）
```

### 7.3 初始化噪声

```python
def apply_init_noise(qpos):
    qpos[root_z] += random(0, 0.02)                   # 高度 ±2cm
    qpos[root_quat] = euler2quat(±3°, ±3°, 0)        # 倾角 ±3°
    qpos[joints] += random(-3°, 3°)                   # 关节 ±3°
```

### 7.4 实现细节

```python
# 直接修改 MuJoCo mjModel 参数（不需要模型变体）
model.dof_frictionloss[jnt] = random(0, 2)
model.dof_damping[jnt] = random(0.02, 2)
model.body("R_KNEE_S").mass[0] = default_mass × random(0.95, 1.05)
model.body("R_KNEE_S").ipos = default_ipos + random(-0.01, 0.01, 3)
```

这与 UniLab 的 `InitRandomizationPlan/ModelVariantSpec` 不同——它直接在已有模型上改参数，不需要预编译多种模型变体。对 UniLab 的腿长 DR 实现有参考价值。

---

## 八、第 7 层：模式系统（课程学习）

### 8.1 三种模式

```python
class WalkModes:
    STANDING = [0,0,1]    # 速度参考 = [0,0,0], 时钟覆盖为"双脚支撑" → 站立
    INPLACE  = [0,1,0]    # 速度参考 = [±rand, 0, 0], 正常时钟 → 原地踏步
    FORWARD  = [1,0,0]    # 速度参考 = [0, rand(0,0.4), 0], 正常时钟 → 前向行走
```

模式信息以 **one-hot 编码** 注入观测。

### 8.2 模式切换算法

```python
# 每控制步执行:

# 切换1: INPLACE ↔ STANDING (1%/步, 仅双腿支撑期)
if random() < 0.01 and is_double_support():
    INPLACE → STANDING or STANDING → INPLACE

# 切换2: FORWARD ↔ INPLACE (0.5%/步, STANDING时不触发)
if random() < 0.005 and mode != STANDING:
    FORWARD → INPLACE or INPLACE → FORWARD
```

**设计理由**:
- STANDING ↔ INPLACE 必须发生在**双腿支撑期**（两只脚都着地），此时最安全
- 不直接从 STANDING 跳到 FORWARD（必须经过 INPLACE）——梯度
- 平均约 100 步切换一次 INPLACE/STANDING，200 步切换一次 FORWARD/INPLACE
- Episode 初始分布: 60% STANDING，20% INPLACE，20% FORWARD

---

## 九、观测空间

### 9.1 JVRC 行走: 37 维

```
自感知 (29D):
  [0]   roll            躯干滚转角 (rad)，从根 quaternion 转换
  [1]   pitch           躯干俯仰角 (rad)
  [2:5] ang_vel         qvel[3:6] — 角速度 [wx, wy, wz]
  [5:17] motor_pos      12 个关节位置 (rad, 执行器编码器值)
  [17:29] motor_vel     12 个关节速度 (rad/s)

外部状态 (8D):
  [29]  sin(2π·phase/88)        时钟信号
  [30]  cos(2π·phase/88)
  [31]  mode_encode[0]          FORWARD → [1,0,0]
  [32]  mode_encode[1]          INPLACE → [0,1,0]
  [33]  mode_encode[2]          STANDING → [0,0,1]
  [34]  yaw_vel_ref             偏航速度参考 (INPLACE 模式下非零)
  [35]  vx_ref                  前向速度参考 (FORWARD 模式下非零)
  [36]  vy_ref                  侧向速度参考 (CURVED/LATERAL 模式下非零)
```

### 9.2 观测归一化

```python
obs_mean = [0, 0, 0,0,0,   deg2rad(半蹲位姿12个),  0×12,   0,0, 0.5×3, 0×3]
obs_std  = [0.2,0.2, 1,1,1,  0.5×12,              4×12,   1,1, 1×3, 0.5×3]
obs = (obs - mean) / std
```

归一化在**网络内部**执行 (`actor.py:161: state = (state - self.obs_mean) / self.obs_std`)。

---

## 十、网络 & PPO 训练

### 10.1 网络架构

```python
Actor:  Linear(37, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 12)
Critic: Linear(37, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 1)
```

**权重初始化 (normc_init)**:
```python
# 隐藏层: 每行权重的 L2 范数归一化到 1
w = normal(0, 1)
w /= ‖w‖₂

# 输出层: 额外 ×0.01
means.weight *= 0.01
```

### 10.2 动作分布

```python
# 策略输出均值 μ，标准差 σ 是固定的 per-dimension 标量
σ_init = 0.2        # ~11.5°
action ~ Normal(μ, σ)

# 确定性执行时直接返回均值
action = μ
```

### 10.3 PPO 超参

| 参数 | 值 | 说明 |
|------|-----|------|
| lr | 3e-4 | FF 网络 |
| lr (LSTM) | 1e-3 | RNN 用更高学习率 |
| γ | 0.99 | 折扣因子 |
| λ | 0.95 | GAE λ |
| clip | 0.2 | PPO 裁剪 |
| epochs | 3 | 每批数据学习轮数 |
| minibatch | 64 | 小批量大小 |
| max_grad_norm | 0.5 | 梯度裁剪 |
| entropy_coeff | 0.0 | 不使用熵正则 |
| mirror_coeff | 0.4 | 镜像损失权重 |
| n_iters | 20000 | 训练迭代数 |
| n_procs | 12 | 并行 worker 数 |

### 10.4 单次 PPO 更新（完整流程）

```python
# 1. 计算新旧策略的 log_prob
pdf = policy.distribution(obs)
log_probs = pdf.log_prob(action).sum(-1)      # 对角高斯，求和所有维度
old_log_probs = old_policy.distribution(obs).log_prob(action).sum(-1)
ratio = exp(log_probs - old_log_probs)

# 2. PPO Clipped 损失
actor_loss = -min(ratio × advantage, clip(ratio, 0.8, 1.2) × advantage).mean()

# 3. Value 损失
critic_loss = MSE(returns, values)

# 4. Mirror 损失
mirror_loss = MSE(policy(obs), mirror_action(policy(mirror_obs)))

# 5. 总损失 = actor + critic + 0.4×mirror + 0.0×entropy
total_loss = actor_loss + critic_loss + 0.4×mirror_loss
total_loss.backward()
clip_grad_norm(0.5)
optimizer.step()
```

---

## 十一、足端目标跟踪（Stepping Task）—— 另一种行走范式

`jvrc_walk` (行走任务) 和 `jvrc_step` (步进任务) 是项目的**两个不同任务**，分别代表两种行走实现范式。walking task 在前文已详细分析，本节聚焦 stepping task 的足端目标跟踪机制。

### 11.1 两种任务的本质区别

| | jvrc_walk (行走) | jvrc_step (步进) |
|------|----------------|-----------------|
| **目标信号** | 连续速度命令 | 离散落脚点坐标 |
| **策略学什么** | "以速度 v 移动" | "把脚踩到点 P" |
| **步态来源** | 时钟驱动 + 奖励涌现 | 落脚点引导 + 时钟辅助 |
| **Reward 主导项** | 速度跟踪 30% + 步态匹配 45% | **step_reward 45%** + 步态匹配 30% |
| **适用场景** | 连续行走、速度控制 | 精确落脚、障碍跨越、台阶 |
| **模式数量** | 3 种 | **6 种** (FORWARD, BACKWARD, LATERAL, CURVED, INPLACE, STANDING) |

### 11.2 足端目标的工作流程 —— "规划 → 观测 → 执行 → 检测"

整个足端目标跟踪系统遵循以下四步循环：

```
┌──────────────────────────────────────────────────────────────┐
│ 1. 高层规划: 根据当前模式，程序化生成落脚点序列               │
│    generate_step_sequence(mode) → [(x₁,y₁,z₁,θ₁), (x₂,...]  │
│                          ↓                                   │
│ 2. 观测注入: 将未来 2 步的目标坐标变换到机器人局部坐标系       │
│    update_goal_steps() → [_goal_steps_x, _goal_steps_y, ...] │
│                          ↓                                   │
│ 3. 策略执行: 策略看到 (状态, 目标点) → 输出关节动作           │
│    动作 → PD → 力矩 → 机器人运动 → 脚掌靠近目标               │
│                          ↓                                   │
│ 4. 检测成功: 任一脚进入目标半径 → 停留延迟 → 推进到下一个目标  │
│    ‖foot_pos - target‖ < 0.20m → delay_frames → t1++, t2++  │
└──────────────────────────────────────────────────────────────┘
```

### 11.3 规划：落脚点序列的程序化生成

`generate_step_sequence()` 根据**当前模式**，生成完整的目标序列（20 步）。序列中的每一步是 `(x, y, z, θ)`，其中 `(x, y, z)` 是绝对世界坐标，`θ` 是步的朝向。

**前向行走 (FORWARD):**

```python
# 生成 zigzag 交替落脚点
sequence = []
# 第一步: 随机左或右开始
first_step = [0, ±step_gap, 0, 0]    # y=±0.15m
sequence.append(first_step)

x = 0
for i in range(1, num_steps - 1):
    x += step_size                    # 每步前进 0.3m
    y *= -1                           # 左右交替 (zigzag)
    z += step_height                  # 台阶高度递增 (课程学习)
    sequence.append([x, y, z, 0])

# 最后一步: 闭合到中线
sequence.append([x + step_size, -y, z, 0])
```

**后退行走 (BACKWARD):**

```python
step_size = -0.1                      # 步幅更小 (后退更难)
# 其余与前向相同
```

**侧向行走 (LATERAL):**

```python
y = 0
c = random_choice([-1, 1])           # 随机方向
for i in range(1, num_steps):
    y += step_size if i%2 else -(2/3)*step_size
    sequence.append([0, c*y, 0, 0])
```

**曲线行走 (CURVED):**

```python
# 从预生成文件加载随机曲线路径
plan = random.choice(self.plans)      # 来自 utils/footstep_plans.txt
sequence = [[x, y, 0, theta] for each step in plan]
```

预生成文件格式:
```
0.0,0.0,0.0
0.3,0.03,0.05
0.6,0.06,0.10
---
0.0,0.0,0.0
0.3,-0.03,-0.05
...
```

每条曲线以 `---` 分隔。曲线路径通过程序生成，一般沿着圆弧排列，模拟转弯。

**原地踏步 (INPLACE):**

```python
step_size = random(-0.05, 0.05)      # 极小步幅 = 原地
```

**站立 (STANDING):**

```python
num_steps = 1                         # 只有 1 步 = 不移动
```

### 11.4 坐标变换：绝对坐标 → 机器人局部坐标

生成的目标序列是**绝对世界坐标**，但在每个控制步需要转换为**机器人当前局部坐标系**下的相对坐标，供策略使用。

```python
def update_goal_steps(self):
    root_pos  = get_root_world_position()          # 躯干世界坐标
    root_quat = get_root_world_orientation()       # 躯干世界朝向
    
    for idx in [t1, t2]:                          # 未来两步
        # 构建目标的世界变换矩阵
        abs_goal_pos = sequence[t][0:3]
        abs_goal_rot = euler2mat(0, 0, sequence[t][3])
        absolute_target = compose(abs_goal_pos, abs_goal_rot)
        
        # 构建机器人的世界变换矩阵
        ref_frame = compose(root_pos, quat2mat(root_quat))
        
        # 将目标从世界系变换到机器人局部系
        relative_target = inv(ref_frame) @ absolute_target
        
        # 提取相对坐标的各分量 → 注入观测
        _goal_steps_x[idx] = relative_target[0, 3]
        _goal_steps_y[idx] = relative_target[1, 3]
        _goal_steps_z[idx] = relative_target[2, 3]
        _goal_steps_theta[idx] = euler_from_mat(relative_target[:3,:3])[2]
```

**为什么用局部坐标而不是世界坐标？** 策略需要知道"目标在我前面 0.3 米、左边 0.05 米"，而不是"目标在世界坐标 (1.5, 0.3, 0)"。局部坐标随机器人朝向旋转，使得策略对不同朝向具有不变性。

### 11.5 观测注入：目标点作为策略输入

目标信息以 **8 维向量**注入策略观测：

```
goal_steps (8D):
  [0] _goal_steps_x[t1]    下一步的 x (前向距离)
  [1] _goal_steps_y[t1]    下一步的 y (侧向偏移)
  [2] _goal_steps_z[t1]    下一步的 z (高度)
  [3] _goal_steps_theta[t1] 下一步的朝向
  [4] _goal_steps_x[t2]    再下一步的 x
  [5] _goal_steps_y[t2]    再下一步的 y
  [6] _goal_steps_z[t2]    再下一步的 z
  [7] _goal_steps_theta[t2] 再下一步的朝向
```

**为什么是未来两步？** 一步信息只够"迈到当前点"，两步信息让策略能**规划步态过渡**——知道当前摆动腿需要跨越到哪个位置，以及下一摆动腿需要预留的空间。

### 11.6 检测：目标达成判定与推进

```python
def step(self):
    # 每一步执行:
    
    # 1. 更新脚位置
    self.l_foot_pos = get_left_foot_world_pos()
    self.r_foot_pos = get_right_foot_world_pos()
    
    # 2. 检查任一脚是否进入目标半径
    target_pos = self.sequence[self.t1][0:3]
    lfoot_hit = ‖self.l_foot_pos - target_pos‖ < 0.20    # 半径 20cm
    rfoot_hit = ‖self.r_foot_pos - target_pos‖ < 0.20
    
    if lfoot_hit or rfoot_hit:
        self.target_reached = True
        self.target_reached_frames += 1                   # 累加停留帧数
    else:
        self.target_reached = False
        self.target_reached_frames = 0
    
    # 3. 持续停留 → 推进目标
    if self.target_reached and self.target_reached_frames >= delay_frames:
        self.update_target_steps()                       # t1++, t2++
        self.target_reached = False
    
    # 4. 重新计算相对坐标供下一步使用
    self.update_goal_steps()
```

**关键设计细节:**

1. **目标半径 0.20m**: 不是要求精确踩点，而是进入 20cm 半径即判定成功。这给予策略一定的容错空间。

2. **停留延迟 `delay_frames`**: 脚进入目标区域后，必须**连续停留**至少 `swing_duration / control_dt = 0.75s / 0.025s = 30 帧` 才认为真正到达。防止"经过目标但不落地"的伪成功。

3. **目标推进**: `update_target_steps()` 执行 `t1 = t2; t2 += 1`。始终有未来两步的目标可用。到达序列末尾后 `t2` 停留在最后一步。

### 11.7 奖励函数: step_reward 是绝对主导

```python
reward = {
    "step_reward":  0.450 × step_reward(),              # 目标达成 45%
    "foot_frc":     0.150 × foot_frc_clock_reward(),    # 步态匹配 30%
    "foot_vel":     0.150 × foot_vel_clock_reward(),
    "orient_cost":  0.050 × body_orient_reward(),       # 姿态正则 15%
    "height_error": 0.050 × height_reward(),
    "upper_body":   0.050 × upper_body_reward(),
}
```

**step_reward 的组成:**

```python
def step_reward(self):
    # 到达奖励: 踩中目标时给，距离越近奖励越高
    if self.target_reached:
        hit_reward = exp(-‖foot - target‖ / 0.25)
    else:
        hit_reward = 0
    
    # 前进奖励: 躯干到目标区域中心的距离
    target_midpoint = (target_t1[:2] + target_t2[:2]) / 2
    root_dist = ‖root_xy - target_midpoint‖
    progress_reward = exp(-root_dist / 2)
    
    return 0.8 × hit_reward + 0.2 × progress_reward
```

到达奖励和前进奖励的设计意图：hit_reward 鼓励**精确落脚**，progress_reward 鼓励**躯干向目标区域移动**。即使脚没有精确踩到，策略也能从向目标靠近中获得前进奖励。

### 11.8 课程学习：台阶高度递增

```python
# 在 FORWARD 模式中，台阶高度随训练进度线性增长
h = clip((iteration_count - 3000) / 8000, 0, 1) × 0.1
step_height = random_choice([-h, h])   # 上台阶或下台阶随机切换

# 3000 步前: h=0（平地行走）
# 3000~11000 步: h 从 0 线性增长到 0.1m
# 11000 步后: h=0.1m（10cm 台阶）
```

这比 walking task 的模式切换课程更有针对性——stepping task 的核心能力就是"踩到不同高度的点"。

### 11.9 地形可视化

```python
# 用 20 个 MuJoCo box geom 可视化落脚点位置
for box_idx, step_coord in enumerate(sequence):
    box_height = model.geom(box_name).size[2]
    model.body(box_name).pos = [step_x, step_y, step_z - box_height]
    model.body(box_name).quat = euler2quat(0, 0, step_theta)
    model.geom(box_name).size = [0.15, 1, box_height]      # 15cm 深的台阶
    model.geom(box_name).rgba = [0.8, 0.8, 0.8, 1]        # 灰色
```

这不仅是可视化——box geom 是**物理碰撞体**，机器人必须踩在上面。前进模式下 floor 被下沉 2 米，迫使机器人只能站在 box 上。

### 11.10 walking task vs stepping task 对比总结

| 维度 | walking task | stepping task |
|------|-------------|---------------|
| **目标信号** | 连续: `[vx, vy, vyaw]` | 离散: `[(x,y,z,θ), ...]` |
| **策略任务** | 学习"用速度 v 行走的关节模式" | 学习"把脚精确放到位置 P 的关节控制" |
| **成功率** | 连续存活（不摔倒） | 离散命中（进入半径） |
| **地形** | 平坦地面 | 可编程的台阶/障碍 |
| **泛化能力** | 适应不同速度 | 适应不同落脚位置和高度 |
| **奖励设计** | 速度跟踪 + 步态匹配并重 | **step_reward 主导 (45%)**，步态匹配辅助 |
| **观测维度** | 37D (含 3D 模式+3D 速度参考) | 37D+8D (含 4D×(t1+t2) 目标坐标) |

---

## 十二、与同类方案的对比

### 12.1 vs Isaac Gym 系列（RSL-RL / rsl_rl）

| 维度 | LearningHumanoidWalking | Isaac Gym (UniLab使用的框架族) |
|------|------------------------|-------------------------------|
| **步态机制** | **显式相位时钟** + 脚底力/速对齐奖励 | 无显式时钟，策略自己学会节奏 |
| **动作空间** | 关节位置增量（经过 PD） | 关节位置增量（经过 PD）或直接力矩 |
| **奖励复杂度** | 10 项，75% 集中 | 13 项，分散在各维度 |
| **并行方案** | Ray 多进程 (12 workers)，每个 worker 1 个 env | GPU 批量 (1024+ envs)，vmap 并行 |
| **控制频率** | 40Hz | 50-100Hz |
| **步态质量** | 强制周期性，步态规整 | 自由演化，可能退化到边角解 |
| **收敛速度** | 较快（时钟提供先验） | 较慢（需自己学节奏） |

### 12.2 vs 轨迹跟踪（PAI / HumanoidSW2）

| 维度 | LearningHumanoidWalking | PAI 12DOF (轨迹跟踪) |
|------|------------------------|---------------------|
| **步态来源** | 奖励塑造 + 时钟驱动 → **涌现性步态** | 预定义正弦参考轨迹 + RL 修正 |
| **参考数据** | 无 | 有（正弦关节轨迹） |
| **灵活性** | 高（可自适应地形） | 低（受限于参考轨迹形态） |
| **复杂度** | 中等（需调时钟参数） | 高（需设计每条腿的参考轨迹） |
| **退化风险** | 中（时钟约束防退化） | 低（轨迹强约束） |

### 12.3 vs AMP (Adversarial Motion Priors)

| 维度 | LearningHumanoidWalking | AMP |
|------|------------------------|-----|
| **参考数据** | 无（步态涌现） | 有（动捕数据） |
| **奖励结构** | hand-crafted 奖励 | 鉴别器输出（style reward） |
| **样本效率** | 较高 | 较低（需要训练鉴别器） |
| **步态自然度** | 中等 | 高（模仿人类步态） |

---

## 十三、可移植到 UniLab XqRobotV2 的改进

### 优先级 1: 步态相位时钟

**可行性**: 高。XqRobotV2 是轮腿双足，有 6 个腿关节 + 2 个轮子，步态更复杂但核心机制通用。

**移植方案**:
```python
# 为每条腿定义 3 种关节组的相位时钟:
# - 髋关节 (hip): 控制侧向稳定，相位与腿摆动/支撑同步
# - 大腿+小腿 (thigh+calf): 控制腿伸缩，相位与 foot clearance 同步
# - 轮子 (wheel): 控制推进力，相位与支撑相重合（着地时驱动，离地时不驱动）

# 奖励: 脚底力匹配时钟 (和原版相同) + 轮速匹配时钟 (新增)
# 观测: 注入 [sin(θ), cos(θ)] 到 actor obs 中
```

### 优先级 2: Episode 内间歇 DR

**可行性**: 中。需要在 `build_interval_randomization_plan` 中实现动态参数修改。

**移植方案**: 参考 `domain_randomization.py` 直接改 `mjModel.dof_damping/mass/ipos`，无需模型变体。

### 优先级 3: 模式切换课程

**可行性**: 高。在 UniLab 的命令系统中增加"模式"维度。

**移植方案**: 命令从 5D `[vx,vy,vyaw,tsk,height]` 扩展到 5D+mode，训练前 30% 迭代以站立指令为主。

### 优先级 4: 观测归一化

**可行性**: 高。`empirical_normalization` 已有基础设施，只需开启。

---

## 十四、关键文件索引

| 机制 | 文件 |
|------|------|
| 步态相位时钟定义 | `tasks/rewards.py:196-300` (`create_phase_reward`) |
| 脚底力/速对齐奖励 | `tasks/rewards.py:107-193` (`calc_foot_frc/vel_clock_reward`) |
| 完整奖励调度 | `tasks/walking_task.py:83-147` (`calc_reward`) |
| 模式系统 + 阶段推进 | `tasks/walking_task.py:21-41, 149-205` |
| PD 控制律 | `envs/common/robot_interface.py:493-508` (`step_pd`) |
| 反向电动势 | `robots/robot_base.py:41-62` (`_do_simulation`) |
| 域随机化 (直接模型修改) | `envs/common/domain_randomization.py:29-56` |
| 观测空间 + 归一化 | `envs/jvrc/jvrc_walk.py:42-67` |
| 镜像排列 | `envs/jvrc/jvrc_base.py:73-110` |
| PPO 更新 (含 mirror loss) | `rl/algos/ppo.py:299-406` |
| 网络架构 + normc_init | `rl/policies/actor.py:122-189`, `rl/policies/base.py:5-22` |
| Step 目标达成奖励 | `tasks/stepping_task.py:66-77` (`step_reward`) |
| URDF→MJCF 模型精简 | `envs/jvrc/gen_xml.py:58-164` |
| 足端目标生成 + 检测 | `tasks/stepping_task.py:139-179, 225-243` |
| 坐标变换 (世界→局部) | `tasks/stepping_task.py:125-137` (`transform_sequence`) |
| 目标坐标注入观测 | `tasks/stepping_task.py:181-199` (`update_goal_steps`) |
| 台阶高度课程 | `tasks/stepping_task.py:312-313` |
| 预生成曲线路径 | `utils/footstep_plans.txt` |

---

*文档生成: 2026-07-08*
