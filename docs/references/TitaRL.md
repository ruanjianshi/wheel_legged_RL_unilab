# Tita RL 核心机制深度分析

> 项目路径: `/home/robot/xiaoq/projects/tita_rl`
> 框架: Isaac Gym + NP3O (约束型 PPO) | 机器人: DEEPRobotics Tita (8DOF 轮腿双足)
> 许可证: MIT | 硬件: RTX 4090 / Jetson Orin NX

本文档聚焦 NP3O 算法、Barlow Twins 自监督表示学习、Teacher-Student 架构与 Sim2Real 部署管线的核心机制。

---

## 一、机器人: DEEPRobotics Tita 轮腿双足

### 1.1 形态

```
躯干 (base_link, 13.2kg)
├── 左腿:
│   ├── leg_1: 髋外展/内收 (HAA)  ±0.785 rad, 60 Nm
│   ├── leg_2: 髋屈/伸   (HFE)  -1.92~3.49 rad, 60 Nm
│   ├── leg_3: 膝屈/伸   (KFE)  -2.67~-0.698 rad, 60 Nm
│   └── leg_4: 轮子       (wheel) 无限制, 15 Nm
└── 右腿: 完全镜像

总质量: ~27.7 kg  |  8 DOF  |  轮半径 0.0925m
默认站位:
  leg_1 (髋): 0.0 rad     — 中性
  leg_2 (大腿): 0.8 rad   — 前倾
  leg_3 (小腿): -1.5 rad  — 弯曲
  leg_4 (轮): 0.0 rad     — 轮静止
```

### 1.2 与 UniLab XqRobotV2 的关键形态差异

| 维度 | Tita (tita_rl) | XqRobotV2 (UniLab) |
|------|---------------|-------------------|
| **质量** | 27.7 kg | ~5 kg |
| **DOF** | 8 (4/腿: HAA+HFE+KFE+Wheel) | 8 (3 腿 + 1 轮/腿) |
| **髋关节** | HAA (外展/内收) + HFE (屈/伸) | 仅 HAA (外展/内收)，pitch 在 thigh/calf |
| **膝** | 标准 KFE | thigh + calf 两根骨联动 |
| **轮控制** | 位置模式×10 增益 + 0.5×阻尼 | 速度控制 (kv=1) |
| **PD 刚度** | Kp=40, Kd=1.0 | Kp=30, Kd=1.0 (腿), Kv=1 (轮) |
| **髋 scale** | hip_scale_reduction=0.5 | action_scale=0.25~0.35 |
| **动作平滑** | 0.8×new + 0.2×old | 无 |

---

## 二、动作空间: 8 维 + Sim2Real 重映射

### 2.1 端到端控制链

```
策略输出 (8D)
  │
  ├─ reindex: [4,5,6,7,0,1,2,3]     ← Sim2Real 重映射
  │    策略: [L_leg3, R_leg3, L_wheel, R_wheel, L_leg1, R_leg1, L_leg2, R_leg2]
  │    UDRF: [L_leg1, R_leg1, L_leg2, R_leg2, L_leg3, R_leg3, L_wheel, R_wheel]
  │
  ├─ hip_scale: action[leg_1] *= 0.5  ← 髋关节动作幅度减半
  │
  ├─ action_scale: action *= 0.5      ← 通用 scaling
  │
  ├─ 动作平滑: target = 0.8×action + 0.2×prev_action  ← 低通滤波
  │
  ├─ target += default_joint_angles   ← 加回默认站位
  │
  ▼
PD 力矩生成:
  腿 (leg_1/2/3): τ = Kp × (target - θ_current) + Kd × (0 - ω_current)
  轮 (leg_4):     τ = Kp × 10 × (target - θ) - 0.5 × Kd × ω  ← 速度模式
```

### 2.2 轮子的特殊控制

轮子使用**准速度模式**:
```python
τ_wheel = 40 * 10 * pos_target - 0.5 * 1.0 * ω
# = 400 * pos_target - 0.5 * ω
```

为什么? 传统位置模式 (`τ = Kp·Δθ + Kd·Δω`) 在轮子上效果差——轮子的绝对角位置无关紧要，策略需要驱动的是转速。通过 10× 增益放大位置误差 + 半阻尼，控制律近似表现为"速度跟踪":
```
τ_wheel ≈ 400 * pos_target - 0.5 * ω
# 稳态时 ω ≈ 800 * pos_target，轮速正比于位置目标
```

这比 UniLab 的纯速度控制 (`kv=1`) 更精细——保留了位置反馈的稳定性，同时通过增益调整实现了速度跟踪。

---

## 三、NP3O 算法: 带约束的 PPO

### 3.1 为什么需要 NP3O？

标准 PPO 通过**奖励惩罚**处理约束 (力矩过大罚 -0.01、关节速度过大罚 -0.1 等)。但这个方案的缺陷:
1. 惩罚权重是人为设定的，可能偏大或偏小
2. 约束违规的**代价**融进了奖励的**梯度**中，优化方向扭曲
3. 约束是"软的"——策略可以学会忍受惩罚来最大化主奖励

NP3O 引入**独立的 cost critic 网络**，用约束违规的**累积代价**作为独立的优化目标，与主奖励解耦。

### 3.2 双价值函数结构

```
Actor:  输入 (33+16+10=59) → [512,256,128] ELU → 8D 动作
        其中 16=history_latent, 10=priv_prediction

Critic: 输入 (33+32+36+32=133) → [512,256,128] ELU → 1D 状态价值
        其中 33=proprio, 32=scan_enc, 36=priv_latent, 32=hist_enc

Cost Critic: 输入同上 133 → [512,256,128] ELU → 6D → Softplus → 累积代价
        输出 6 维，分别对应 6 个约束项的长期代价估计
```

### 3.3 多目标损失函数

```python
# 1. PPO 标准项
surrogate_loss  = -min(ratio*adv, clip(ratio,1-ε,1+ε)*adv).mean()
value_loss      = 0.5 * (returns - V(s))^2

# 2. Cost Critic 项
cost_value_loss = 0.1 * (cost_returns - V_cost(s))^2  # 约束价值估计
viol_loss       = 0.1 * max(0, V_cost(s) - d) * ratio.mean()
# viol_loss 约束: 当 cost critic 预测的累积约束超过阈值 d 时，惩罚策略

# 3. 熵 + 模仿
entropy_loss    = 0.01 * action_dist.entropy()
imitation_loss  = 1.0 → 0 (线性衰减)  # Barlow Twins

# 总损失
loss = surrogate_loss + value_loss + cost_value_loss + viol_loss
     - entropy_loss + imitation_loss
```

### 3.4 K 乘数渐进收紧

```python
k = initial_k  # 初始值: pos_limit=0.3, torque_limit=0.3, dof_vel=0.3, acc=0.1, force=0.1, stumble=0.1

# 每迭代 ×1.0004
k = min(1.0, k * 1.0004^iteration)
# iter=0:       k = 0.1~0.3
# iter=2500:    k ≈ min(1, 0.3*1.0004²⁵⁰⁰) ≈ min(1, 0.3*2.72) ≈ 0.82
# iter=5000+:   k = 1.0 (完全收紧)
```

这与 BoltLocomotion 的 CaT 框架有相似之处——都在**渐进收紧约束**，但方式不同:
- CaT: 提高终止概率的 max_p (0.05→0.25，双曲线退火)
- NP3O: 提高约束违反在损失中的权重 (0.3→1.0，指数增长)

---

## 四、Barlow Twins 自监督表示学习

### 4.1 问题: 为什么需要 Barlow Twins？

标准的 Teacher-Student 架构用 MSE 损失让学生模仿教师。但 Barlow Twins 更进一步——它不直接模仿动作，而是**对齐表示**。

### 4.2 架构

```
输入: 10 步历史 (330D 轨迹)
  │
  ├─ 左路 (无增广的历史)
  │     HistoryEncoder (1D CNN): 330 → 512 → 256 → 128 → 16 (latent)
  │     PrivPredictor: 16 → 10 (预测 privileged info: mass, friction, motor_strength, etc.)
  │
  └─ 右路 (有增广的历史)
        HistoryEncoder (共享权重) → 16 (latent')

Barlow Twins 损失:
  C = (Z_left^T · Z_right) / batch_size          # 跨批次协方差矩阵 (16×16)

  on_diag  = Σ(1 - C_ii)^2                        # 对角线 → 1 (不变性)
  off_diag = λ × Σ_ij(C_ij^2)                     # 非对角线 → 0 (去冗余)
  # λ = 5e-3

  priv_loss = 0.01 × MSE(predicted_priv, true_priv)  # 辅助预测
  total = on_diag + off_diag + priv_loss
```

### 4.3 Barlow Twins 的物理意义

1. **不变性** (`on_diag → 1`): 让 history encoder 对数据增广 (时间偏移、dropout) 产生**一致的表示**。这意味着 encoder 提取的是"本质特征"（如步态相位），而不是"表面特征"（如噪声）。

2. **去冗余** (`off_diag → 0`): 让 16 维潜在表示的**各维度彼此独立**。防止 encoder 过拟合到某个单一维度（如全部维都在编码质心速度），确保信息最大化分散。

3. **特权预测** (`priv_loss`): 让表示中包含"可以从历史中推断但无法直接观测"的信息——如车身线速度（privileged 中有的 base_lin_vel）、触地状态等。

### 4.4 为什么用 Barlow Twins 而不是 SimCLR 或 BYOL？

- **SimCLR** 需要负样本和大 batch size → 不适合 RL 的在线学习场景
- **BYOL** 需要动量编码器 → 增加实现复杂度
- **Barlow Twins** 在协方差矩阵层面约束 → 天然适应 RL 的 batch 收集-更新循环

---

## 五、Teacher-Student 非对称架构

### 5.1 信息流

```
┌─────────────────────────────────────────────────────────┐
│ Teacher (训练时):                                        │
│   观测: proprio (33D) + scan (187D) + priv (36D)        │
│     + history (330D)                                     │
│                                                          │
│   输出: 8D 动作 + 16D latent + 10D priv_pred             │
│                                                          │
│   Critic 输入:           student_obs + priv + scan_enc   │
│   (评论家可以"看到"特权信息，做出更准确的价值估计)         │
│                                                          │
│ Student (推理时):                                        │
│   观测: proprio (33D) + scan (187D) + history (330D)    │
│   (没有 priv 信息)                                       │
│                                                          │
│   输出: 8D 动作 + 16D latent                             │
│   (借助 history encoder 推断 priv 信息)                   │
└─────────────────────────────────────────────────────────┘
```

### 5.2 特权信息 (36D) 的组成

| 分组 | 维度 | 内容 | 作用 |
|------|------|------|------|
| base_lin_vel | 3 | 躯干线速度 (世界系) | **关键** — 学生必须从历史中推断 |
| contact_filt | 2 | 左右脚低通滤波后的触地状态 | 帮助 critic 判断步态 |
| lag | 1 | 当前环境的通信延迟步数 | 帮助 critic 理解延迟 |
| mass_params | 4 | 质心质量 + COM 偏移 (×3) | DR 参数 → critic 可评估当前动力学 |
| friction | 1 | 地面摩擦系数 | DR 参数 |
| restitution | 1 | 弹性系数 | DR 参数 |
| motor_strength | 8 | 每关节的力矩乘数 | DR 参数 |
| kp_factor | 8 | 每关节的刚度乘数 | DR 参数 |
| kd_factor | 8 | 每关节的阻尼乘数 | DR 参数 |

**为什么 critic 可以看特权信息？** 这是标准的 Actor-Critic 非对称设计——critic 只在训练时存在，用于提供更好的梯度；actor 只接收学生观测，确保部署时不需要特权信息。

---

## 六、域随机化: 9 参数全覆盖

| 参数 | 范围 | 方法 |
|------|------|------|
| 地面摩擦 | [0.2, 2.75] | 64 bins 离散采样 / env |
| 弹性系数 | [0, 1.0] | 64 bins |
| 基座质量 | [-1.0, +3.0] kg | Uniform / env |
| 基座 COM | [-0.1, 0.1] m × 3 | Uniform / env |
| 电机强度 | [0.8, 1.2] × 8 joints | Uniform / env |
| Kp 刚度 | [0.8, 1.2] × 8 joints | Uniform / env |
| Kd 阻尼 | [0.8, 1.2] × 8 joints | Uniform / env |
| 推力扰动 | max_vel=1 m/s，每 15s 推一次 | 随机方向 |
| 通信延迟 | 0-2 步滞后 | / env |

**关键差异**: 每个 env 的 DR 参数**在创建时随机采样，episode 内不变化**。这与 LearningHumanoidWalking (每 0.5s 重采样) 和 BoltLocomotion (创建时固定) 在策略上不同。Tita 选择固定是因为 4096 envs 已经提供了足够的参数多样性。

---

## 七、观测空间: 586 维

### 7.1 完整观测分解

```
proprio (33D):                    # 每帧
  base_ang_vel × 2.0 (3D)        # 缩放躯干角速度
  projected_gravity (3D)          # 重力投影
  commands × scale (3D)           # 速度指令 [vx, vy, vyaw]
  dof_pos_err (8D)                # 关节位置偏差 (轮位置归零)
  dof_vel × 0.05 (8D)            # 缩放关节速度
  last_action (8D)                # 上一帧动作

history (330D):                   # 10 步 × 33D
  1D CNN encoder → 16D latent

scan (187D):                      # 17 x-points × 11 y-points
  MLP encoder → 32D scan_enc

priv_latent (36D):                # 只给 Teacher
  直接使用，不编码

Critic 输入 (133D):
  proprio (33D) + scan_enc (32D) + priv_latent (36D) + hist_latent (32D)

Actor 输入 (59D):
  proprio (33D) + hist_latent (16D) + priv_pred (10D)
```

### 7.2 关键设计选择

1. **轮位置归零**: `dof_pos[:, [3,7]] = 0` — 轮是无限转动的回转关节，绝对角位置无意义
2. **base_lin_vel 不出现在学生观测中**: 策略需要从历史 + 扫描中**隐式推断**线速度，这是一个经典的 Privileged Learning 设定
3. **扫描仅 critic 可用 (类似 UniLab 的 critic 独有 height_scan)**: 减轻 actor 的输入负担，让 critic 有地形信息做更好的价值估计

---

## 八、Sim2Real 部署管线

### 8.1 完整链路

```
Isaac Gym 训练 (.pt)
  │
  ├─ 1. torch.onnx.export → model.onnx
  │     opset 16, constant folding
  │
  ├─ 2. trtexec → model.engine
  │     --fp16 --optShapes for batch inference
  │
  ├─ 3. 部署到 Jetson Orin NX
  │     TensorRT runtime → 实时推理
  │
  └─ 4. Sim2Sim 验证 (Webots 2023)
        确保 sim2sim 转移一致性
```

### 8.2 为什么这个管线可靠？

1. **reindex 重映射**: 策略输出的动作顺序经过 `[4,5,6,7,0,1,2,3]` 重排，匹配真实硬件的电机索引——这防止了"仿真中正常、真机上关节错位"的常见问题

2. **lag buffer**: 每 env 随机 0-2 步通信延迟，策略训练时会遇到指令滞后 → 真机上同样情况已有预期

3. **Barlow Twins 的去冗余表示**: History encoder 产出的 16D 表示各维独立 → 压缩表示对噪声/传感器漂移更鲁棒

4. **动作低通滤波**: `0.8*new + 0.2*old` 平滑高频 jitter → 真机电机不会剧烈抖动

---

## 九、与 BoltLocomotion CaT 的对比: 两种约束方法

| 维度 | Tita NP3O | Bolt CaT |
|------|----------|----------|
| **约束机制** | 独立 cost critic + viol_loss 梯度约束 | 概率化终止 + 乘性奖励惩罚 |
| **约束收紧** | K 乘数指数增长 (0.3→1.0) | max_p 双曲线退火 (0.05→0.25) |
| **约束影响** | 进入策略梯度 (viol_loss) | 不进入梯度，独立门控 |
| **约束数量** | 6 项 | 13 项 |
| **奖励与约束关系** | 在损失中加权组合 | 奖励被约束概率乘性缩放 |
| **适合场景** | 轮腿双足 (相对稳定的动力学) | 点足双足 (极不稳定) |

**直觉**: NP3O 适合 Tita 这样相对稳定 (轮子提供支撑) 的机器人——约束可以以"软"方式融入梯度。CaT 适合 Bolt 这样高度不稳定 (点足) 的机器人——约束必须"硬"门控，否则策略永远学不会。

---

## 十、与 UniLab XqRobotV2 的对比

| 维度 | Tita RL | UniLab XqRobotV2 |
|------|---------|-------------------|
| **算法** | **NP3O** (cost critic + viol loss + K 乘数) | 标准 PPO |
| **表示学习** | **Barlow Twins** (协方差对齐 + 去冗余) | 无 |
| **架构** | Teacher-Student (BarlowTwins) | Actor-Critic (对称) |
| **约束** | 6 项 → cost critic | 13 项 → 奖励加权 |
| **动作平滑** | ✅ 0.8×new + 0.2×old | ❌ 无 |
| **hip scale** | ✅ hip_scale_reduction=0.5 | ❌ 纯 action_scale |
| **轮控制** | 准速度模式 (10×Kp, 0.5×Kd) | 纯速度控制 (kv=1) |
| **动作重映射** | ✅ Sim2Real reindex | ❌ 无 |
| **通信延迟** | ✅ lag buffer (0-2 步) | ❌ 无 |
| **Sim2Sim** | Webots 验证 | ❌ (仅 MuJoCo 评估) |
| **部署** | ONNX → TensorRT → Jetson | 无 (当前仅仿真) |
| **观测** | 586D (含 history encoder) | 297/511D (actor/critic 分头) |
| **特权信息** | Student 不可见 lin_vel (需推断) | Actor/Critic 对称 |
| **质量** | 27.7 kg | ~5 kg |
| **DR** | 9 参数全随机 | mass + com (刚修复) |

---

## 十一、可移植到 UniLab XqRobotV2 的改进

### 优先级 1: 动作平滑 (低通滤波)

**可行性**: 极高。一行代码。
```python
filtered_action = 0.8 * action + 0.2 * prev_action
```
防止策略输出高频切换，减少 PD 力矩抖动和真机电机应力。

### 优先级 2: 轮子准速度控制

**可行性**: 高。UniLab 的轮子是纯速度控制 (kv=1)，不可调力矩。可添加增益参数。
```python
# 当前 UniLab:
τ_wheel = wheel_target_vel  # 纯速度控制

# Tita 风格 (保留位置反馈):
τ_wheel = Kp * 10 * pos_target - 0.5 * Kd * ω
```
这使策略可以同时控制轮子扭矩和速度，更适合爬坡和不平坦地形。

### 优先级 3: Teacher-Student 架构 + Barlow Twins

**可行性**: 中。需要添加 history encoder + Barlow Twins 损失。

**方案**:
- Critic 保留当前的全景观测 (含 height_scan)
- Actor 移除 height_scan，用 Barlow Twins encoder 从历史中提取地形特征
- 添加 16D 历史编码器 + priv predictor
- 目标: actor 在不直接观测 terrain scan 的情况下仍能感知地形

### 优先级 4: Sim2Real 动作重映射

**可行性**: 高。在 ONNX 导出前添加 reindex 层。确保仿真训练的动作顺序可映射到真机电机索引。

---

## 十二、关键文件索引

| 机制 | 文件 |
|------|------|
| NP3O 算法 (双 value + viol_loss) | `algorithm/np3o.py` |
| NP3O 训练循环 + Cost GAE | `runner/on_constraint_policy_runner.py` |
| RolloutStorage (含 cost) | `runner/rollout_storage.py` |
| BarlowTwins Actor-Critic | `modules/actor_critic.py` |
| History Encoder (1D CNN) | `modules/common_modules.py` |
| Barlow Twins 损失 | `modules/actor_critic.py:L~80-130` |
| 任务配置 (所有超参) | `configs/tita_constraint_config.py` |
| 主训练环境 (奖励+约束+DR+地形) | `envs/legged_robot.py` |
| Tita 无约束版环境 | `envs/no_constrains_legged_robot.py` |
| 动作 reindex | `envs/legged_robot.py:L~200` |
| Lag buffer | `envs/legged_robot.py:L~540` |
| 轮子控制律 (10×Kp, 0.5×Kd) | `envs/legged_robot.py:L~490` |
| 域随机化 (9 参数) | `envs/legged_robot.py:L~1370-1600` |
| 地形生成 (只含楼梯) | `utils/terrain.py` |
| 训练入口 | `train.py` |
| ONNX 导出 + TensorRT | `export_policy_as_onnx.py` |
| Sim2Sim Webots 播放 | `simple_play.py` |
| URDF 模型 | `resources/tita/urdf/tita_description.urdf` |
| 全局配置路径 | `global_config.py` |

---

*文档生成: 2026-07-08*
