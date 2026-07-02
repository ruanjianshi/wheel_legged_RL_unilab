# 07. 参数调优实战

## 目录

- [调优策略](#调优策略)
- [关键超参逐一分析](#关键超参逐一分析)
- [奖励权重调优](#奖励权重调优)
- [课程学习调优](#课程学习调优)
- [地形配置调优](#地形配置调优)
- [域随机化调优](#域随机化调优)
- [诊断工具](#诊断工具)

---

## 调优策略

### 改进手段优先级 (从低风险到高风险)

| 优先级 | 手段 | 配置路径 | 风险 |
|--------|------|----------|------|
| 1 | 调整奖励权重 | `reward.scales.*` | 最低 |
| 2 | 调整课程参数 | `env.curriculum.*` | 低 |
| 3 | 调整 PPO 超参 | `algo.algorithm.{entropy_coef,lr,noise_std}` | 中 |
| 4 | 调整命令范围 | `env.commands.vel_limit` | 中 |
| 5 | 调整地形比例 | `generator.sub_terrains` | 中 |
| 6 | 调整网络结构 | `algo.policy.*_hidden_dims` | 高 |
| 7 | 调整观测空间 | `_compute_obs()` | 高 |
| 8 | 调整机械结构 | XML, Keyframe | 最高 |

### 单一变量原则

> 每次只改一个条件，保留对照组。如果同时改 entropy_coef 和 learning_rate，不知道是哪个起了作用。

---

## 关键超参逐一分析

### 1. entropy_coef (探索系数)

| 值 | 效果 | 现象 |
|-----|------|------|
| 0.01 (默认) | 偏探索 | action_std 保持在 0.3-0.5 |
| 0.002 (当前) | 低探索 | action_std 下降到 0.05-0.10 |
| 0.001 | 几乎确定 | action_std → 0, 策略过度拟合 |

**何时调**:
- action_std 不下降 → 提高 entropy_coef
- action_std 太快塌缩到 <0.01 → 降低 entropy_coef.

**配置文件**: `conf/ppo/task/*/mujoco.yaml` → `algo.algorithm.entropy_coef`

### 2. init_noise_std (初始噪声)

| 值 | 效果 |
|-----|------|
| 1.0 (默认) | 大探索 → 难以学习精细控制 |
| 0.3 (当前) | 中等 → 适合初期训练 |
| 0.1 | 小探索 → 可能陷入局部最优 |

### 3. learning_rate (学习率)

| 值 | 效果 |
|-----|------|
| 1e-3 (默认) | 快收敛, 可能不稳定 |
| 1e-4 (当前) | 稳定收敛 |
| 5e-5 | 极慢收敛 |

### 4. desired_kl (目标 KL 散度)

| 值 | 效果 |
|-----|------|
| 0.01 (默认) | 每次更新允许较大步长 |
| 0.005 (当前) | 严格控制更新步长 |
| 0.002 | 极保守更新 |

**注意**: `desired_kl` 配合 `schedule: adaptive` 使用。当实际 KL > desired_kl 时，学习率自动衰减。

### 5. 命令范围 (vel_limit)

```yaml
env.commands.vel_limit:
  - [-0.6, -0.3, -1.0, -0.1, 0.45]   # [vx, vy, vyaw, tsk, height] 下限
  - [ 0.6,  0.3,  1.0,  0.1, 0.85]   # 上限
```

**典型调整**:
- 地形不平增加 vx 上限 → `-1.0` 到 `1.0`
- 需要侧移增加 vy 上限 → `-0.5` 到 `0.5`

### 6. 命令重采样时间 (resampling_time)

| 值 | 效果 |
|-----|------|
| 3.0 | 每 3 秒换命令 → 策略学习"跟随命令" |
| 10.0 | 每 10 秒换命令 → 策略学习"长期稳定" |
| 999.0 | 评估用 — 全程不变 |

---

## 奖励权重调优

### 权重数量级对照

| 奖励 | 作用 | 典型值 | 调高效果 | 调低效果 |
|------|------|--------|----------|----------|
| `tracking_lin_vel: 1.5` | 线速度跟踪 | +1.0 ~ +2.0 | 更精确跟踪，可能振荡 | 更平稳，可能不跟踪 |
| `tracking_ang_vel: 1.5` | 角速度跟踪 | +1.0 ~ +2.0 | 更快转向 | 转向慢 |
| `orientation: -10.0` | 倾斜惩罚 | -5 ~ -20 | 更直立，更僵硬 | 更灵活，可能倒 |
| `base_height: -5.0` | 高度保持 | -3 ~ -10 | 高度稳定 | 高度波动 |
| `joint_action_rate: -0.1` | 动作平滑 | -0.05 ~ -0.5 | 动作更慢/更僵 | 高频抖动 |
| `wheel_action_rate: -0.005` | 轮平滑 | -0.001 ~ -0.01 | 轮速变化慢 | 轮速突变 |
| `alive: 1.0` | 存活动机 | 0.5 ~ 2.0 | 求存优先级高 | 更愿意冒风险 |

### 调优流程

```
1. 看 TensorBoard 的 reward/* 曲线
2. 找到始终 ≈ 0 的奖励 → 其 weight 太低，未被学习
3. 找到始终远大于其他的惩罚 → 其 weight 太高，支配策略
4. 每次只改 1 个 reward scale，调 ±50%
5. 再训练 500-1000 iter 观察效果
```

### 常见场景

```
场景 A: 机器人站住但不走
  → tracking_lin_vel 和 alive 太接近
  → 降低 alive (0.5) 或提高 tracking_lin_vel (2.0)

场景 B: 机器人一走就倒
  → orientation 惩罚不够
  → 提高 orientation (-15 → -20)

场景 C: 机器人高度不稳定
  → base_height 惩罚不够
  → 提高 base_height (-8 → -10)

场景 D: 机器人高频抖腿
  → action_rate 惩罚不够
  → 提高 joint_action_rate (-0.2 → -0.5)
```

---

## 课程学习调优

### 速度课程

```yaml
env:
  curriculum:
    enabled: true
    vel_step: 0.002              # 每次扩展的速度步长 (m/s)
    ang_vel_step: 0.004          # 角速度步长 (rad/s)
    min_vel_range_frac: 0.3      # 初始速度 = 上限的 30%
    update_interval: 25          # 每 25 步检查
    err_threshold: 0.35          # 追踪误差阈值
```

**工作原理**:
- 训练开始时，命令范围 = 配置上限的 30%（例如 vx ∈ [-0.18, 0.18]）
- 当追踪误差降到 `err_threshold` 以下 → 扩展范围 + `vel_step`
- 对称扩展上下限

**调试**:
- `vel_step` 太大 → 速度范围增长过快，策略跟不上
- `vel_step` 太小 → 训练过半还到不了全速
- `err_threshold` 太高 → 过早扩展，策略质量低

### 地形课程

```python
# config.py → STAIRS_TERRAINS_CFG
TerrainGeneratorCfg(
    curriculum=True,        # ★ 课程模式
    num_rows=10,            # 10 个难度级别
    difficulty_range=(0.0, 1.0),
    sub_terrains={...},
)
```

课程模式下，每行 = 一个难度级别，`difficulty = row_index / num_rows`。

---

## 地形配置调优

### 调整子地形比例

```python
# rough.py → XqRobotRoughTerrainCfg
sub_terrains={
    "flat":              flat(proportion=0.20),          # 20% 平地
    "pyramid_stairs":    pyramid_stairs(proportion=0.15), # 15% 台阶
    "pyramid_stairs_inv": pyramid_stairs_inv(proportion=0.15), # 15% 反向台阶
    "hf_pyramid_slope":  hf_pyramid_slope(proportion=0.05),    # 5% 斜坡
    "random_rough":      random_rough(proportion=0.30),        # 30% 粗糙
    "wave_terrain":      wave_terrain(proportion=0.15),        # 15% 波浪
}
```

**调优思路**:
- 多放平地 → 机器人先学会基本行走，再逐步加难
- 少放极端地形 → `pyramid_stairs` 的 `step_height_range` 设低
- 平衡分布 → 不要让某个地形占比 > 40%

### 调整地形参数

```python
# 降低台阶高度 (适合初期训练)
pyramid_stairs(proportion=0.15, step_height_range=(0.0, 0.05))
# 降低粗糙度
random_rough(proportion=0.30, noise_range=(0.0, 0.03))
# 减小斜坡坡度
hf_pyramid_slope(proportion=0.05, slope_range=(0.0, 0.10))
```

---

## 域随机化调优

### DR 配置

```yaml
env:
  domain_rand:
    randomize_base_mass: false
    randomize_ground_friction: false
    randomize_kp: false
    randomize_kd: false
    random_com: false
    randomize_leg_length: false
```

### 推荐启用顺序

```
阶段 1: 全部关闭 (当前)
  → 最快收敛，验证算法和奖励

阶段 2: 开启地面摩擦随机化
  → randomize_ground_friction: true  (0.5 ~ 2.0 倍)

阶段 3: 开启基座质量随机化
  → randomize_base_mass: true  (±5kg)

阶段 4: 开启 PD 增益随机化
  → randomize_kp: true, randomize_kd: true (±20%)

阶段 5: 开启质心偏移
  → random_com: true  (±2cm)

阶段 6: 开启腿长随机化
  → randomize_leg_length: true  (80%~120%)
```

**每开一个 DR 维度后重训 1000-2000 iter 观察**，不要一次全开。

---

## 诊断工具

### TensorBoard

```bash
bash shell/tensorboard.sh flat 8080
# → http://localhost:8080

# 关键曲线:
# - reward/tracking_lin_vel    — 速度跟踪质量
# - reward/orientation          — 倾斜程度 (越接近 0 越好)
# - mean_episode_length        — 存活步数
# - action_std                  — 探索程度 (应缓慢下降)
# - learning_rate               — 自适应 LR 变化
```

### Assess 评估

```bash
# 快速评估 (6 场景)
uv run assess/runner.py -t flat_walk -a ppo -r <run> -c <ckpt>

# 全量评估 (16 场景)
uv run assess/runner.py -t flat_walk -a ppo -r <run> -c <ckpt> -s full

# 趋势分析
uv run assess/runner.py -t flat_walk -a ppo -r <run> \
    --trend --ckpts 1000,2000,3000,4000,5000
```

### 训练日志

```bash
# 监控实时训练
tail -f /tmp/flat_train.log | grep -E "iter|mean_reward|episode_length|action_std"

# 死亡环境比例
tail -f /tmp/flat_train.log | grep -oP 'dead=\K\d+'
```

---

## 完整的调优流程图

```
训练 5000 iter → 评估
  │
  ├─ action_std > 0.3? → entropy_coef 太高, 降到 0.001
  ├─ action_std < 0.01? → entropy_coef 太低, 升到 0.005
  │
  ├─ Vx RMSE > 0.3? → tracking_lin_vel 权重太低, 升到 2.0
  ├─ Vy 串扰 > 0.3? → hip 不对称问题, 检查 DEFAULT_ANGLES
  │
  ├─ 倾斜 > 15°? → orientation 权重到 -20
  ├─ 高度波动 > 0.1m? → base_height 权重到 -10
  │
  ├─ 存活率 < 50%? → alive 权重升到 2.0, 放松终止条件
  ├─ 轮速突变? → wheel_action_rate 权重到 -0.01
  │
  └─ 都 OK?
       ├─ 开 friction DR → 再训
       ├─ 开 mass DR → 再训
       └─ 增加地形难度 → 再训
```

---

## 关键要点

1. **单变量变更** — 每次只改一个，做对照
2. **先跑平路再跑粗糙** — 先让策略在平坦地面学会行走
3. **TensorBoard 是眼睛** — reward 曲线暴露一切
4. **entropy_coef 是关键旋钮** — 控制探索 vs 利用
5. **DR 逐步开启** — 一次开一个维度，确认不崩

---

> 上一章: [06. 奖励函数设计](./06_reward_function.md)
> 
> 🎉 恭喜完成全部教程！
> 
> 返回 [教程首页](./README.md)
