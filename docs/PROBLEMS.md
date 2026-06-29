# XqRobotV2 问题记录与解决方案

## 目录

1. [训练崩溃：episode length 恒为 1](#1-训练崩溃episode-length-恒为-1)
2. [Vx 速度方向反转](#2-vx-速度方向反转)
3. [左右腿一前一后（步态不对称）](#3-左右腿一前一后步态不对称)
4. [机身向一侧倾斜](#4-机身向一侧倾斜)
5. [线速度追踪不佳 (0.95/1.5)](#5-线速度追踪不佳-09515)
6. [髋关节改为 0.0 导致训练崩溃](#6-髋关节改为-00-导致训练崩溃)
7. [KeyboardCommander vel_limit 不兼容](#7-keyboardcommander-vel_limit-不兼容)
8. [课程学习只向正方向扩展](#8-课程学习只向正方向扩展)
9. [feet_distance: -100.0 导致策略崩溃](#9-feet_distance--1000-导致策略崩溃)
10. [leg_length DR 开启后训练崩溃](#10-leg_length-dr-开启后训练崩溃)
11. [轮子速度控制引入后线速度仍不佳](#11-轮子速度控制引入后线速度仍不佳)
12. [键盘 Q/E 高度控制无效](#12-键盘-qe-高度控制无效)

---

## 1. 训练崩溃：episode length 恒为 1

**现象**：训练启动后所有环境 1 步就终止，entropy 从正变负，action_std → 0。

**根因**：`obs_frame_dim` 与命令维度不匹配。config 改了 5D 命令但代码里 `_obs_frame_dim` 还是 32（4D 命令用）。`_compute_obs` 生成 32 维 frame 塞不进 33 维 history buffer，NumPy 赋值报错，观测乱码 → 策略输出垃圾 → 第 1 步就终止。

**解决**：改动 `_obs_frame_dim` 必须与 config 的 `vel_limit` 维度同步。5D 命令 → `_obs_frame_dim = 33`, `_critic_frame_dim = 36`。

**文件**：`src/unilab/envs/locomotion/xqrobotV2/joystick.py` `_obs_frame_dim`、`_critic_frame_dim`

---

## 2. Vx 速度方向反转

**现象**：键盘给正向 Vx → 机器人后退；给负向 Vx → 机器人前进。

**根因**：不是传感器或运动学符号问题（物理验证正轮速 → 正 vx）。是因为 `DEFAULT_LEG_ANGLES` 左右不对称（左髋 0.1，右髋 0.0），静态就有左倾力矩。策略为了保持平衡学出了异常步态——腿部动作产生的反向推力超过了轮子前进力。

**解决**：保持双髋 0.1（提供必要的外展支撑），加大 `orientation: -10.0` 和 `similar_calf: -1.0` 强制策略纠正倾斜，避免策略为了补倾斜学出异常步态。

**验证**：物理测试 `mujoco.mj_step` 正轮速 → `local_linvel[0]` 为正，确认传感器链无误。

---

## 3. 左右腿一前一后（步态不对称）

**现象**：行走时左右腿无法保持同步，一前一后。

**根因**：`similar_calf` 奖励原来只约束小腿 `(calf_L - calf_R)²`，髋关节和大腿没有任何对称约束。策略可以随意让腿关节各自运动。

**解决**：扩展 `similar_calf` 函数覆盖全部腿关节：

```python
hip = (left_hip + right_hip)²      # 髋镜像
thigh = (left_thigh - right_thigh)² # 大腿平行
calf = (left_calf - right_calf)²    # 小腿平行
```

权重从 -0.3 加到 -1.0。

**文件**：`src/unilab/envs/locomotion/xqrobotV2/joystick.py` `_reward_similar_calf`

---

## 4. 机身向一侧倾斜

**现象**：行走时机器人整体向一侧偏斜。

**根因**：两个髋关节都用 `axis="1 0 0"`（同向 convention），正确镜像需要 `left = -right`，即 `left + right ≈ 0`。旧代码完全没有髋关节约束，策略随意把双髋往同方向转。

**解决**：在 `similar_calf` 中加入 `hip = (left + right)²` 项。加权 -1.0。

**文件**：同上 `_reward_similar_calf`

---

## 5. 线速度追踪不佳 (0.95/1.5)

**现象**：5000 轮训练后线速度追踪只能到 ~0.95，角速度追踪能到 ~1.37。

**根因**：三个因素叠加：
1. **轮子用位置控制**：XML 里 `<position kp="3">`，策略给轮子设目标位置而非目标转速，响应极慢
2. **控制频率低**：`ctrl_dt=0.02` (50Hz)，轮子每 20ms 才更新一次指令
3. **奖励权重偏向稳定**：`orientation: -10.0` 是 `tracking_lin_vel: 1.5` 的 6.7 倍，策略优先保平衡

**解决**：
1. 轮子改为 `<velocity kv="1">` 速度控制（参考 CJ-003）
2. `ctrl_dt` 从 0.02 降到 **0.01** (100Hz)
3. `wheel_action_rate` 从 -0.015 放宽到 -0.005
4. `apply_action` 轮子不再叠加 `DEFAULT_ANGLES`

**文件**：`xqrobotV2.xml` actuator, `joystick.py` apply_action, `mujoco.yaml` reward scales, `base.py` ctrl_dt

---

## 6. 髋关节改为 0.0 导致训练崩溃

**现象**：把 `DEFAULT_LEG_ANGLES` 左右髋都改为 0.0 后，训练立即崩溃（ep=1，entropy 负）。

**根因**：XqRobotV2 结构需要髋外展提供横向支撑。髋=0.0 时两腿完全直立，支撑面如刀刃，任何微小扰动都导致摔倒。CJ-003 能用髋=0.0 是因其结构不同（髋间距、COM 高度、轮径均不同）。

**解决**：保持双髋 0.1。Python `DEFAULT_LEG_ANGLES = [0.1, 0.1, ...]`，keyframe 右髋=0 的不一致是**故意设计的**——控制器自动把右髋从 0 拉到 0.1，建立双髋外展的稳定支撑。

**文件**：`base.py` DEFAULT_LEG_ANGLES, `scene_flat.xml` keyframe

---

## 7. KeyboardCommander vel_limit 不兼容

**现象**：键盘评估时崩溃 `commands.vel_limit must have shape (2, 3), got (2, 4)`。

**根因**：`KeyboardCommander.from_vel_limit` 硬编码要求 shape (2, 3)。XqRobotV2 是 4D/5D vel_limit。

**解决**：改为接受 (2, K) K≥3，取前 3 列 (vx, vy, vyaw)。命令写回时只更新前 3 列 `commands[:, :3]`。

**文件**：`src/unilab/visualization/interactive_playback.py` `from_vel_limit`

---

## 8. 课程学习只向正方向扩展

**现象**：配置了对称 vel_limit `[-0.6, 0.6]`，但训练中 vx 永远为正。

**根因**：课程学习硬编码 `full_low_x = 0.0`，只向正方向扩展：`low[0] = max(low[0] + step*0, 0) → 永远是 0`。

**解决**：改为对称扩展：
```python
vx_range = max(abs(low[0]), abs(high[0]))
low[0] = max(low[0] - step, -vx_range)
high[0] = min(high[0] + step, vx_range)
```

**文件**：`joystick.py` `_update_curriculum`

---

## 9. feet_distance: -100.0 导致策略崩溃

**现象**：加了 `feet_distance: -100.0` 后训练崩溃（ep=1，entropy 负）。

**根因**：`-100.0` 惩罚过重。XqRobotV2 默认轮距约 0.33m，接近下限 0.3m。随机探索时稍低于 0.3m 就触发 `(0.3 - 0.29) * 0.3 * (-100) = -0.3`，累积惩罚压倒其他奖励。策略为了避免惩罚学会输出零动作 → entropy 崩。

**解决**：权重降到 **-1.0**。

**文件**：`mujoco.yaml` `feet_distance: -1.0`

---

## 10. leg_length DR 开启后训练崩溃

**现象**：`randomize_leg_length: true` 时 steps/sec 从 38k 暴跌到 10k，训练崩溃。

**根因**：1024 个 env 各生成独立模型变体（不同腿长 mesh 缩放），每个变体需要 `MjSpec.compile()` 编译新模型二进制，编译极慢且每个变体物理属性不同，动态差异破坏训练稳定。

**解决**：推迟到策略收敛后再开。当前关闭。

**文件**：`mujoco.yaml` `randomize_leg_length: false`

---

## 11. 轮子速度控制引入后线速度仍不佳

**现象**：改为速度控制 + 100Hz 后，iter 332 时追踪 0.93，未明显提升。

**分析**：iter 332 还在早期探索阶段（熵 1.83），策略尚未充分利用速度控制的优势。等待 2000+ 轮评估。

**状态**：训练中，待观察。

---

## 12. 键盘 Q/E 高度控制无效

**现象**：按 Q/E 调整机身高度，机器人无反应。

**根因**：旧模型训练时 `base_height_target` 固定 0.65，策略观测中无高度指令，无法响应动态变化的高度目标。

**解决**：将高度目标加入 5D 命令向量第 5 列 `[vx, vy, vyaw, tsk, height_target]`，策略训练时学会响应。键盘 Q/E 写入 `commands[:, 4]` 和 `env._reward_cfg.base_height_target`。

**文件**：`play_interactive.py` keyboard commander, `joystick.py` 5D obs
