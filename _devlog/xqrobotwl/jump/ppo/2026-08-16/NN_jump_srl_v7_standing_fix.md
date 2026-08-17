# SRL v7: 站立稳定修复 (无指令自跳 + 站立振荡)

**日期**: 2026-08-16
**来源**: 负责人可视化观察: SRL 无指令也在跳跃 + 站立姿态偏差大 → 量化诊断 (P3) 确认: 跳后 trigger=0 仍持续"蹲-起"振荡 (\|gyro\| mean 2.05 / max 5.2 / 末期 4.2), FSM 全程 idle, 属策略自持极限环。

## 修改了什么

1. **`src/unilab/envs/locomotion/xqrobotwl/jump_srl.py`** `_reward_landing_recovery`
   - 门控从 `landing_timer > 0` (落地后 30 步窗口) 改为 **trigger≤0.5 站立期恒开** (文献 standing cost)。
   - 文献依据: 站立不会从移动/跳跃奖励自然涌现, 必须显式奖励 (`docs/references/2026-08-16_jump_standing_stabilization_rl.md`)。
2. **`conf/ppo/task/xqrobotwl_jump_srl_flat/mujoco.yaml`**
   - `control_config.action_smoothing: 0.3` (新增, 动作一阶低通 = 文献 4Hz LPF, 压高频摆腿)
   - `ang_vel_xy: -0.15 → -0.5` (强化站立期摆动惩罚)
3. **`src/unilab/envs/locomotion/xqrobotwl/jump_srl.py`** `apply_action` (新增 override)
   - 裁剪膝位置目标到 ±0.85 (机械极限), 防蹬伸膝过伸 (实测 1.04)。

## 根因分析

- 训练命令 trigger 每 4s 随机重采样, ~50% 时间在跳跃窗口; 跳跃奖励 (jump_height 25 + height_progress 20 + vertical_thrust 30 ≈ 75) 远大于站立惩罚 (ang_vel_xy 仅 -0.15)。
- 旧 landing_recovery 只覆盖落地后 0.3s 窗口, 窗口一过回落到弱惩罚区 → 策略把"蹲-起"当站立平衡态。
- FSM 全程 idle (诊断确认 500 步 idle), 与 FSM 无关, 是 PPO 反馈策略自持振荡。

## 参数调整好坏

| 参数 | 旧 | 新 | 预期 |
|---|---|---|---|
| landing_recovery 门控 | 落地后 0.3s | 站立期恒开 | 站立期每步可挣恢复奖励, 持续推向稳定站立 |
| ang_vel_xy | -0.15 | -0.5 | 更强摆动惩罚 (需防压垮跳高, 跳跃期同样生效) |
| action_smoothing | 0 (无) | 0.3 | 一阶低通, 消高频抖动 |
| 膝目标裁剪 | 无 | ±0.85 | 防膝过伸 (机械安全) |

## 验证方法

- 环境冒烟: `smoke_v7.py` 四 env 构建/步进通过; VMC pytest 10 passed。
- 重训 4000 iter (`logs/train/jump_srl_v7.log`, GPU1)。
- 训练后用 `diag_jump_problems.py` + `diag_jump_postjump.py` 重新诊断: 站立 |gyro| / 跳后自跳数 / 站立关节角。

## 训练后效果

### iter-1000 中间评估 (未收敛)
| 指标 | v6 基线 | v7 iter-1000 |
|---|---|---|
| 站立 \|gyro\| 均值 | 0.92 | **0.60** |
| 站立 \|gyro\| 最高 | 2.9 | **2.0** |
| 跳后 tail \|gyro\| 均值 | 2.05 | **1.07** |
| 末期 \|gyro\| | 2.03 | **0.74** |
| 跳高 | 0.266m | 0.289m |
| 膝过伸 | 1.04 | 0.95 (略降, 目标裁剪后残余动量过冲) |
| 相位序列 | 第二跳跳过 flight | 两周期均含 flight |

站立显著稳定化, 跳后不再持续"蹲-起"; 尚有短暂 \|gyro\| 尖峰 (5.5)。待 4000 iter 收敛后复测。

### 最终 (model_3999, 4000 iter) — P3 解决 ✅
| 指标 | v6 基线 | v7 最终 |
|---|---|---|
| 站立 \|gyro\| 均值 | 0.92 | **0.51** |
| 站立末期 \|gyro\| | 2.03 | **0.13** |
| 跳后5s \|gyro\| 均值 | 2.05 | **0.20** |
| 跳后5s \|gyro\| 最高 | 5.2 | **1.75** |
| 无指令自跳 | — | **0 次** |
| 跳高 | 0.266m | **0.315m** |
| 髋外展 | 0.032 | **0.04** |
| 膝过伸 | 1.04 | 0.994 (动态瞬态残余) |

站立稳定化 + 跳后不持续蹲-起 + 跳高提升。膝过伸为动态瞬态残余 (目标已裁剪 ±0.85, MuJoCo 高速蹬伸允许 ~0.13 过冲)。

## 后续计划

- 若站立仍振荡: 阶梯式站立稳定奖励 / 训练命令分布提高 trigger=0 占比 (gait-conditioned)。
- 落地恢复窗口若不足: 按文献延长到覆盖整个恢复期。

**关联**: [[NN_jump_srl_v6_posture]], [[assess: 2026-08-16_four_algo_visual_problems_diag]], [[2026-08-16_jump_standing_stabilization_rl]]
