# 06. 奖励函数设计

## 目录

- [奖励系统架构](#奖励系统架构)
- [RewardContext 快照模式](#rewardcontext-快照模式)
- [奖励分发器 (run_reward_dispatch)](#奖励分发器-run_reward_dispatch)
- [内置奖励函数](#内置奖励函数)
- [自定义奖励函数](#自定义奖励函数)
- [完整示例: 添加新的奖励函数](#完整示例-添加新的奖励函数)
- [调参思路](#调参思路)

---

## 奖励系统架构

```
YAML 配置                   Python 代码                 每条步进
────────                    ──────────                  ────────
reward.scales:              _reward_fns dict:           _compute_reward():
  tracking_lin_vel: 1.5     {"tracking_lin_vel": fn}      ↓
  orientation: -10.0        {"orientation": fn}          build RewardContext
  alive: 1.0                {"alive": fn}                ↓
  ...                                ↓                  run_reward_dispatch()
                                   dispatch:            ↓
                                   for name, scale:     fns[name](ctx) × scale
                                     rew = fn(ctx)      sum(全部) × ctrl_dt
                                     reward += rew*scale → 返回
```

**设计原则**:
1. 奖励函数是**纯函数**，只依赖 `RewardContext` — 不访问 `self`，不依赖环境状态
2. 权重通过 YAML 配置，不在代码中硬编码
3. 奖励计算不阻塞仿真步进

---

## RewardContext 快照模式

**文件**: `src/unilab/envs/locomotion/common/rewards.py:21-50`

```python
@dataclass
class RewardContext:
    """奖励函数的全部输入，在 _compute_reward 中构建，之后不可变。"""

    info: dict                                        # 包含 commands, actions 等
    linvel: np.ndarray              # (N, 3) 局部线速度
    gyro: np.ndarray                # (N, 3) 角速度
    gravity: np.ndarray | None      # (N, 3) 重力投影
    dof_pos: np.ndarray             # (N, num_actions) 关节位置
    dof_vel: np.ndarray | None      # (N, num_actions) 关节速度
    num_envs: int = 0
    default_angles: np.ndarray       # 默认关节角
    tracking_sigma: float = 0.25     # exp 跟踪核宽度
    base_height_target: float = 0.0  # 目标高度
    base_height: np.ndarray          # (N,) 当前底座高度
```

**在环境中构建** (`joystick.py:350-368`):
```python
def _compute_reward(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
    ctx = RewardContext(
        info=info,
        linvel=linvel,
        gyro=gyro,
        dof_pos=dof_pos[:, :6],         # 只传 6 个腿关节 (排除轮子)
        dof_vel=dof_vel[:, :6],
        num_envs=linvel.shape[0],
        default_angles=DEFAULT_ANGLES[:6],
        tracking_sigma=0.3,
        base_height_target=0.65,
        base_height=base_height_values,
        gravity=gravity,
    )
    return run_reward_dispatch(scales=..., fns=..., ctx=ctx, ...)
```

---

## 奖励分发器 (run_reward_dispatch)

**文件**: `src/unilab/envs/locomotion/common/rewards.py:339-374`

```python
def run_reward_dispatch(
    *,
    scales: Mapping[str, float],          # {"tracking_lin_vel": 1.5, ...}
    fns: Mapping[str, Callable],          # {"tracking_lin_vel": fn_impl, ...}
    ctx: RewardContext,
    info: dict,
    enable_log: bool,
    ctrl_dt: float,
    only_positive: bool = False,
) -> np.ndarray:

    reward = np.zeros((ctx.num_envs,))
    step_count = info.get("steps", ...)
    should_log = enable_log and (step_count[0] % 4 == 0)  # 每 4 步记录一次

    for name, scale in scales.items():
        if scale == 0 or name not in fns:
            continue               # 跳过零权重和未注册的奖励
        rew = fns[name](ctx)      # ① 调用奖励函数
        reward += rew * scale     # ② 加权累加
        if should_log:
            info["log"][f"reward/{name}"] = float(np.mean(rew * scale))

    if only_positive:
        np.maximum(reward, 0.0, out=reward)  # ③ clamp 负奖励

    return reward * ctrl_dt        # ④ 时间归一化
```

**关键机制**:
1. **按名分发**: 遍历 YAML 的 `scales`，通过 `fns` 字典查找实现
2. **日志门控**: 每 4 步记录一次 (减少日志量)
3. **时间归一化**: `reward × ctrl_dt` 使总奖励与步长无关
4. **零权重跳过**: `scale=0` 的项直接跳过，不计算

---

## 内置奖励函数

**文件**: `src/unilab/envs/locomotion/common/rewards.py`

### 跟踪类

```python
# 线速度跟踪 — 指数核
def tracking_lin_vel(ctx: RewardContext) -> np.ndarray:
    commands = ctx.info["commands"]                     # (N, 5) → [vx, vy, vyaw, tsk, height]
    lin_vel_error = np.sum(np.square(commands[:, :2] - ctx.linvel[:, :2]), axis=1)
    return np.exp(-lin_vel_error / ctx.tracking_sigma)  # ↑ σ 越小越严格

# 角速度跟踪
def tracking_ang_vel(ctx: RewardContext) -> np.ndarray:
    commands = ctx.info["commands"]
    ang_vel_error = np.square(commands[:, 2] - ctx.gyro[:, 2])
    return np.exp(-ang_vel_error / ctx.tracking_sigma)
```

### 稳定性惩罚

```python
# 垂直速度惩罚
def lin_vel_z(ctx: RewardContext) -> np.ndarray:
    return np.square(ctx.linvel[:, 2])                  # v_z² — 不希望上下晃

# 横滚/俯仰惩罚
def ang_vel_xy(ctx: RewardContext) -> np.ndarray:
    return np.sum(np.square(ctx.gyro[:, :2]), axis=1)   # ω_x² + ω_y²

# 倾斜惩罚 (最重要)
def orientation(ctx: RewardContext) -> np.ndarray:
    return np.sum(np.square(ctx.gravity[:, :2]), axis=1) # gravity_xy²

# 高度偏差
def base_height(ctx: RewardContext) -> np.ndarray:
    return np.square(ctx.base_height - ctx.base_height_target)
```

### 动作平滑

```python
def action_rate(ctx: RewardContext) -> np.ndarray:
    current = ctx.info["current_actions"]
    last = ctx.info["last_actions"]
    return np.sum(np.square(current - last), axis=1)    # 惩罚剧烈变化

def torques(ctx: RewardContext) -> np.ndarray:
    return np.sum(np.square(ctx.info.get("torques", 0)), axis=1)  # 节能
```

### 步态与存活

```python
def alive(ctx: RewardContext) -> np.ndarray:
    return np.ones(ctx.num_envs)                        # 存活常数 = 1.0

def feet_air_time_positive_biped(ctx: RewardContext) -> np.ndarray:
    # 鼓励双脚交替离地
    ...
```

---

## 自定义奖励函数

### 规则

1. 签名: `def my_reward(ctx: RewardContext) -> np.ndarray:`
2. 返回值: `(N,)` 形状的 numpy 数组
3. 不访问 `self`，不修改 `ctx`

### XqRobotV2 自定义示例

**文件**: `src/unilab/envs/locomotion/xqrobotV2/joystick.py`

#### 1. 腿对称 (similar_calf)

```python
def _reward_similar_calf(ctx: RewardContext) -> np.ndarray:
    # 左髋+右髋 = 0 (镜像), 大腿左-右=0 (平行), 小腿左-右=0 (平行)
    hip   = ctx.dof_pos[:, 0] + ctx.dof_pos[:, 3]    # L_hip + R_hip ≈ 0
    thigh = ctx.dof_pos[:, 1] - ctx.dof_pos[:, 4]    # L_thigh - R_thigh ≈ 0
    calf  = ctx.dof_pos[:, 2] - ctx.dof_pos[:, 5]    # L_calf - R_calf ≈ 0
    return np.square(hip) + np.square(thigh) + np.square(calf)
```

#### 2. 髋关节内收

```python
def _reward_hip_roll(ctx: RewardContext) -> np.ndarray:
    # 运动时收髋，防止外展过大
    moving = np.abs(ctx.info["commands"][:, 0]) + np.abs(ctx.info["commands"][:, 1])
    hip_mag = np.square(ctx.dof_pos[:, 0]) + np.square(ctx.dof_pos[:, 3])
    return hip_mag * np.clip(moving / 0.2, 0.0, 1.0) * 0.3
    # ↑ 仅在前进/侧移时惩罚。静止不收髋。
```

#### 3. 轮子对称

```python
def _reward_wheel_symmetry(ctx: RewardContext) -> np.ndarray:
    commands = ctx.info["commands"]
    turning = np.abs(commands[:, 2]) > 0.1            # |vyaw| > 0.1 = 转弯
    wheel_actions = ctx.info["current_actions"][:, -2:] # 最后 2 维 = 轮 action
    diff = np.square(wheel_actions[:, 0] - wheel_actions[:, 1])
    return diff * (1.0 - turning.astype(float)) * 0.5
    # ↑ 只在直行时要求轮速对称 (转弯时自然不等)
```

#### 4. 髋差动跟踪 (TSK)

```python
def _reward_tsk(ctx: RewardContext) -> np.ndarray:
    tsk_cmd = ctx.info["commands"][:, 3]              # 髋差动命令
    hip_diff = ctx.dof_pos[:, 0] - ctx.dof_pos[:, 3]  # L_hip - R_hip
    return np.square(hip_diff - tsk_cmd)
```

#### 5. 轮距

```python
def _reward_feet_distance(ctx: RewardContext) -> np.ndarray:
    dist = ctx.info.get("feet_distance")              # 轮间距 (m)
    if dist is None:
        return np.zeros((ctx.num_envs,))
    over = np.maximum(0.0, dist - 0.6)               # 太宽 > 0.6m
    under = np.maximum(0.0, 0.3 - dist)              # 太窄 < 0.3m
    return (over + under) * 0.3
```

---

## 完整示例: 添加新的奖励函数

### Step 1: 写函数

```python
# 在 joystick.py 或 rewards.py 中
def _reward_forward_progress(ctx: RewardContext) -> np.ndarray:
    """奖励前进速度 (仅在 Vx 命令 > 0 时生效)"""
    commands = ctx.info["commands"]
    moving_forward = commands[:, 0] > 0.1              # 仅当前进命令
    forward_vel = np.maximum(0.0, ctx.linvel[:, 0])      # 仅正速度
    return forward_vel * moving_forward.astype(float)
```

### Step 2: 注册

```python
# 在 _init_reward_functions() 中
self._reward_fns["forward_progress"] = _reward_forward_progress
```

### Step 3: 配置权重

```yaml
# 在 mujoco.yaml 的 reward.scales 中
forward_progress: 0.5    # 新奖励
```

### Step 4: 训练并验证

```bash
uv run train --algo ppo --task xqrobotV2_walk_flat --sim mujoco
# TensorBoard 中查看 reward/forward_progress 曲线
```

---

## 调参思路

### 权重数量级关系

| 奖励类型 | 典型 range | raw 值数量级 | 推荐 scale | 作用 |
|----------|-----------|-------------|-----------|------|
| exp 跟踪 | [0, 1] | ~0.8 | +1.0 ~ +2.0 | 主要驱动 |
| 平方惩罚 | [0, 0.1] | ~0.01 | -5 ~ -20 | 姿态约束 |
| 常数 | 1.0 | 1.0 | +1.0 | 存活动机 |
| 差值 | [0, 0.5] | ~0.05 | -1 ~ -3 | 对称/平滑 |

### 调试流程

```
1. 先从简单开始
   → tracking_lin_vel (+1.0) + alive (+1.0) + orientation (-5.0)
   → 看机器人能否站住

2. 逐项添加惩罚
   → + base_height (-2.0) → 调高度
   → + ang_vel_xy (-0.01) → 稳姿态
   → + action_rate (-0.1) → 平滑动作

3. 添加对称 / 步态奖励
   → + similar_calf (-1.0) → 腿对称
   → + hip_roll (-2.0) → 收髋

4. 观察 TensorBoard 中每个 reward 曲线
   → 如果某个奖励始终接近 0 → 权重可能太低
   → 如果某个惩罚持续很大且不改善 → 可能是矛盾条件
```

### 常见问题

| 现象 | 可能原因 | 解决 |
|------|----------|------|
| 机器人不停抖腿 | action_rate 权重太低 | 提高 `joint_action_rate` |
| 机器人总倾斜 | orientation 权重太低 | 提高 `orientation` → -20 |
| 不走路，原地站 | 跟踪权重太低或惩罚过高 | 降低惩罚或提高 tracking |
| 高度不稳 | base_height 太低 | 提高到 -8 ~ -10 |
| 奖励不收敛 | 多个奖励矛盾 | 去掉多余的，回到最小集 |

---

## 关键要点

1. **奖励函数是纯函数** → 签名 `(ctx: RewardContext) → (N,) np.ndarray`
2. **权重在 YAML** → 不在代码中，方便调参
3. **时间归一化** → `× ctrl_dt`，换步长不换行为
4. **每条奖励独立可观测** → TensorBoard 中独立曲线
5. **最小集开始** → 先走起来再加惩罚

---

> 上一章: [05. URDF 机器人移植](./05_urdf_import.md)
> 下一章: [07. 参数调优实战](./07_parameter_tuning.md)
