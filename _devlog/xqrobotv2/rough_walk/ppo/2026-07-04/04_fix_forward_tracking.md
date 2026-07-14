# 04 修复前向速度跟踪失败

**日期**: 2026-07-05
**来源**: 评估 ckpt=19999 full suite（16 场景），发现所有 Vx 正向指令均被逆向执行
**关联**: [_devlog/rough_walk/ppo/2026-07-01/03_start_training.md](2026-07-01/03_start_training.md)

---

## 问题描述

训练至 iter=19999 的策略在 `full` suite (16场景) 评估中表现异常：

- **所有正向 Vx 指令 (vx=+0.1~0.6) 的实际 avg_vx 均为负值**（约 -0.15 ~ -0.29 m/s），机器人一致向后行走
- 反向指令 vx=-0.3 时 avg_vx=-0.247，跟踪相对良好
- Vx RMSE 在正向指令下 0.30~0.95，反向指令下 0.30~0.43
- Vy 跟踪 RMSE 0.21~0.34，基础不稳

**量化数据（ckpt=19999 @ full suite）**:

| 指令 | avg_vx | Vx RMSE | 预期 |
|------|--------|---------|------|
| vx=+0.3 | **-0.254** | 0.617 | +0.3 |
| vx=+0.6 | **-0.289** | 0.947 | +0.6 |
| vx=-0.3 | **-0.247** | 0.299 | -0.3 |
| vx=-0.6 | **-0.298** | 0.427 | -0.6 |

机器人学到的是"向后走 ~0.25 m/s"的稳定步态，不论前向指令大小。

## 根因分析

共发现 **4 个问题叠加**，分别定位如下：

### 1. 奖励失衡 — 正前向跟踪奖励被生存惩罚淹没

`tracking_lin_vel` 使用平方误差的指数衰减函数，sign-agnostic。指令 vx=+0.3，实际 vx=-0.25，误差 = (0.3 - (-0.25))² / 0.3² ≈ -3.36 → exp(-3.36) * 1.5 ≈ 0.052。

单步 tracking_lin_vel 仅贡献 ~0.05，而 `alive=1.0` / `orientation=-10.0` / `base_height=-5.0` 压倒单步正向跟踪信号。策略收敛到"安全后退"这个 reward 更优的局部最优解。

**证据**: 训练曲线的 `mean_reward=51.53` 不低，但来自 alive+orientation+height，Vx 跟踪反而退化。

### 2. 命令解耦不一致

`XqRobotRoughDRProvider._sample_commands`（reset 时调用）生成 **vx+vy 同时非零** 的指令：
```python
# rough.py:141-151（原始）
cmds = np.random.uniform(low=low, high=high, ...)
# 无 decoupling，vx 和 vy 同时非零
```

而 `_update_commands`（episode 内 resample）强制 decoupling（择一归零）。reset 和 resample 的命令结构不一致，增加策略学习难度。

### 3. 课程速度过快

对比 flat walk：`vel_step=0.001`, `ang_vel_step=0.002`；rough walk 原为 `vel_step=0.002`, `ang_vel_step=0.004`，**2x 速度扩展命令范围**。在崎岖地形上，命令范围扩大太快，策略还没学会跟踪小范围就被推到全范围。

### 4. 轮子驱动力不足 + 指令持久过长

- `wheel_action_scale=5.0`（flat walk=10.0），轮子最大转速只有平地的一半
- `resampling_time=10.0s`（flat walk=3.0s），命令持久达 1000 步，在崎岖地形上出现大姿态偏差时很久不换指令

## 解决方案

### 修改 1: rough.py — 命令解耦
```
src/unilab/envs/locomotion/xqrobotV2/rough.py:148-156
```
在 `_sample_commands` 末尾加入与 flat walk 相同的 decoupling 逻辑，确保 reset 时 vx/vy 择一为非零。

### 修改 2: mujoco.yaml — 奖励权重再平衡
```
conf/ppo/task/xqrobotV2_walk_rough/mujoco.yaml:82-96
```
- `tracking_lin_vel`: 1.5 → **2.5**（+67%，增强前向激励）
- `tracking_ang_vel`: 1.5 → **1.0**（崎岖地形降低偏航要求）
- `orientation`: -10.0 → **-8.0**（略微放宽姿态约束）
- `alive`: 1.0 → **2.0**（鼓励更长的存活时间积累跟踪奖励）

### 修改 3: mujoco.yaml — 控制参数
```
conf/ppo/task/xqrobotV2_walk_rough/mujoco.yaml:45-51
```
- `action_scale`: 0.25 → **0.35**（增大腿部动作范围）
- `wheel_action_scale`: 5.0 → **8.0**（增大轮驱，接近 flat walk=10.0）
- `resampling_time`: 10.0 → **6.0**（命令更换更频繁）

### 修改 4: mujoco.yaml — 课程参数
```
conf/ppo/task/xqrobotV2_walk_rough/mujoco.yaml:72-78
```
- `vel_step`: 0.002 → **0.001**
- `ang_vel_step`: 0.004 → **0.002**
匹配 flat walk 的扩展速度，给策略时间学习跟踪。

### 附加修复: assess/runner.py — 趋势分析 bug
```
assess/runner.py:324
```
`run_trend()` 中 `pair` 变量未定义，导致 `--trend` 模式崩溃。改为直接使用 `task` 参数。

## 修改文件

| 文件 | 行号 | 改动 |
|------|------|------|
| `src/unilab/envs/locomotion/xqrobotV2/rough.py` | L148-156 | 添加 _sample_commands 命令解耦 |
| `conf/ppo/task/xqrobotV2_walk_rough/mujoco.yaml` | L36-45 | action_scale=0.35, wheel_action_scale=8.0, resampling=6.0 |
| `conf/ppo/task/xqrobotV2_walk_rough/mujoco.yaml` | L73-74 | vel_step=0.001, ang_vel_step=0.002 |
| `conf/ppo/task/xqrobotV2_walk_rough/mujoco.yaml` | L83-96 | tracking_lin_vel=2.5, orientation=-8.0, alive=2.0, tracking_ang_vel=1.0 |
| `assess/runner.py` | L324 | 修复 run_trend() pair 未定义 bug |

## 验证方法

1. `make test-all` 通过
2. 重训 rough_walk PPO，每 1000 iter 评估 decoupling suite
3. 目标：iter=20000 时 forward Vx 跟踪 RMSE < 0.3, avg_vx 符号与指令一致

## 评估结果

> 旧策略（ckpt=19999）评估数据供对比基准：

| metric | ckpt=19999 (旧) | 新训练目标 |
|--------|----------------|-----------|
| avg_vx (vx=+0.3) | -0.254 | > +0.15 |
| vx_rmse (vx=+0.3) | 0.617 | < 0.30 |
| base_height_mean | 0.618 | > 0.62 |

## 后续计划

- [x] 启动新训练
- [ ] iter=5000 评估 check 跟踪是否改善
- [ ] iter=20000 跑 full suite 全面评估
- [ ] 必要时进一步调 reward 或 terrain 配置

---

*记录人: AI (opencode)*
