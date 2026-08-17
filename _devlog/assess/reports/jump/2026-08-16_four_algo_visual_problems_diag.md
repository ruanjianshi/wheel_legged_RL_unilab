# 四算法可视化问题量化诊断 (2026-08-16)

**来源**: 负责人用可视化脚本 (`scripts/play/play_interactive.py` 交互查看器) 观察四算法, 报告四个问题:
- P1 纯PPO    — 基本没有跳跃过程
- P2 PPO+VMC  — 有跳跃但无正常"下蹲蓄力→起跳"时序 (姿态尚可)
- P3 SRL      — 一直在跳跃, 没有给指令也在跳; 站立姿态偏差很大
- P4 SRL+VMC  — 无下蹲起跳; 跳跃过程是髋关节外展的跳跃方式

**模型**: 四算法最终定稿
- 纯PPO: `logs/rsl_rl_ppo/XqRobotWLJumpFlat/2026-08-16_01-53-39_mujoco/model_9999.pt`
- PPO+VMC: `logs/rsl_rl_ppo/XqRobotWLJumpVMC/2026-08-16_01-53-43_mujoco/model_1000.pt`
- SRL v6: `logs/rsl_rl_ppo/XqRobotWLJumpSRLFlat/2026-08-16_13-36-25_mujoco/model_3999.pt`
- SRL+VMC v6: `logs/rsl_rl_ppo/XqRobotWLJumpSRLVMC/2026-08-16_14-08-53_mujoco/model_3999.pt`

**工具**: 新增 `tools/xqrobotwl/diag_jump_problems.py` + `diag_jump_postjump.py`
- 确定性策略 (action mean), 禁随机命令重采样 (`resampling_time=0`, 已验证 `cmd4` 全程==trigger)
- 测试 A `self_trigger`: trigger 恒 0 共 3s → 站立稳定性 + 自跳检测
- 测试 B `jump`: settle 0.8s + pulse 1.6s + tail 1.2s 单次脉冲 → 下蹲/起跳/髋外展/膝过伸/相位时序
- 测试 C `postjump` (SRL/SRL+VMC): 跳完一次后 trigger=0 持续 5s → 是否持续弹跳
- 原始数据: `logs/pose_data/jump_problem_diag.json` / `jump_postjump_diag.json`
- 控制频率 ctrl_dt=0.01s (100Hz), 膝关节极限 ±0.85 rad, 髋外展阈值 |roll|>0.5

---

## 1. 逐算法问题与量化数据

### P1 纯PPO — 无跳跃过程 (固有极限) ✅数据确认

| 指标 | 实测 | 判定 |
|---|---|---|
| 下蹲深度 | **0.006 m** (几乎不蹲) | ❌ |
| 跳高 | 0.089 m | ❌ <0.20 目标 |
| 腾空步数 | 16 步 | ⚠️ 勉强离地 |
| FSM 相位 | 全程 -1 (无参考结构) | — |
| hip_roll / 膝过伸 | 0.039 / 0.556 | ✅ 干净 |
| 站立 | z=0.567, \|gyro\| 最高 2.9, 末期 0.2 | ⚠️ 站得住但初始晃 |

**根因**: 无 SLIP 参考结构, 纯 PPO 学不到"蹲→蹬"时序 (7 轮迭代验证的固有极限, 跳高 0.09m 封顶)。**不是 bug, 是算法能力边界。**

### P2 PPO+VMC — 跳跃但下蹲浅 + 膝过伸 ✅数据确认

| 指标 | 实测 | 判定 |
|---|---|---|
| 下蹲深度 | **0.07 m** (参考 L0 0.28 只压 0.087m) | ⚠️ 浅蹲 |
| 起跳时序 | 延迟到脉冲后 52 步 (0.52s, 等 FSM thrust) | ⚠️ 慢 |
| **膝过伸** | **L=1.045 / R=1.034 (超 ±0.85 极限!)** | ❌ 安全隐患 |
| hip_roll | 0.003 | ✅ 干净 |
| 一次脉冲跳跃次数 | 2 次 (phase 序列两个完整周期) | ⚠️ |
| 跳高 | 0.254 m | ✅ |
| 站立 | z=0.554, \|gyro\| 最高 3.4, 末期 0.42 | ✅ 相对稳 |

**根因**: ① VMC 虚拟腿长参考 `crouch_length=0.28` 从站立 L0=0.367 只压缩 0.087m → 下蹲浅; ② 蹬伸相 `thrust_length=0.50` 需要膝越过 ±0.85 → 膝过伸 (VMC 力控不感知关节机械极限); ③ FSM 一旦离开 idle 就走完整个周期, 1.6s 脉冲内完成 2 跳。

### P3 SRL — 站立不稳 + 跳后持续"蹲-起"振荡 (无指令也动) ✅数据确认

| 指标 | 实测 | 判定 |
|---|---|---|
| 全新 trigger=0 (3s) | 自跳 0 次, 但站立 \|gyro\| 最高 **2.9**, z=0.568 偏高, 膝微屈 | ❌ 站立偏差大 |
| **跳后 trigger=0 (5s)** | **\|gyro\| mean 2.05 / max 5.2 / 末期 4.2** — 剧烈摆动 | ❌ 持续弹跳 |
| 跳后 z 变化 | 深蹲到 0.25m 再站起, 反复"蹲-起" | ❌ |
| 跳后 FSM | 全程 idle (500 步) | → 非 FSM 重复触发 |
| 触发后相位序列 | 单次干净周期 crouch→thrust→flight→preland→landing→idle | ✅ FSM 正常 |
| 膝过伸 | L=1.04 / R=0.98 | ❌ |
| 跳高 | 0.266 m | ✅ |

**根因**:
1. **策略自持振荡, 与 FSM 无关**: 跳后 FSM 全程 idle, 是 PPO 反馈策略把"蹲-起"当成了无指令时的平衡态 (极限环), 且 `alive=1.0` 每步保住不死, 振荡被宽容。
2. **训练统计失衡**: 训练时 trigger 每 4s 随机重采样, ~50% 时间在跳跃窗口; 跳跃奖励 (`jump_height 25 + height_progress 20 + vertical_thrust 30 ≈ 75`) 远大于站立惩罚 (`ang_vel_xy -0.15`), 策略把"跳"泛化进站立。
3. **站立期姿态惩罚太弱**: `ang_vel_xy -0.15`、`lean_forward 1.5 (3×0.5)`、`anti_drift -3` 压不住自持摆动; idle FSM 前馈增益仅 0.15, 无法锚定站姿。
4. **跳后无收敛锚**: `landing_recovery 8.0` 只在 `landing_timer` 窗口 (30 步=0.3s) 内有效, 窗口一过立刻回落到弱站立惩罚区。

### P4 SRL+VMC — 髋外展跳跃 + 无下蹲 ✅数据确认 (与观察一致)

| 指标 | 实测 | 判定 |
|---|---|---|
| **髋外展** | **跳跃期 hip_roll max L=0.518 / R=0.611 (超 0.5 外展阈值!)** | ❌ 髋外展跳跃 |
| **无下蹲** | 触发即起跳 (第一触发步 vz>0.2); "下蹲相" z 反从 0.478 升到 0.54 | ❌ 无下蹲蓄力 |
| 相位序列 | `[-1,0,1,3,4,...]` 两次都**跳过 flight 态** | ❌ 轮未真正离地 |
| 膝过伸 | L=0.955 / R=0.916 | ❌ |
| 跳高 | 0.261 m | ⚠️ 含站立期弹跳能量 |
| 跳后 5s | 完成脉冲内启动的第二次跳 (FSM 周期收尾); \|gyro\| mean 1.26 / 末期 0.91 | ⚠️ 比 SRL 稳 |

**根因**:
1. **奖励缺口 — 跳跃期无髋外展约束**: SRL+VMC 奖励集里 hip_roll 只在 `crouch_prep`/`crouch_depth` 的 `roll_ok` 门控 (权重小且仅下蹲相) 和 `lean_forward` (仅 trigger≤0.5 站立期) 约束。**thrust/flight/landing 期完全没有 hip_roll 惩罚** → 策略发现"髋外展借力"也能起跳。
2. **VMC roll 动作有实际权限**: `action_scale_roll=0.3`, 配合 `kp_roll=40` 力矩, 实际 hip_roll 可达 ±0.6。
3. **跳高奖励不对 roll 门控**: `jump_height 30` / `height_progress 25` 只要 base_z 升高就发, 外展跳同样拿满 → 无压力纠正姿态。

---

## 2. 跨算法发现

| 发现 | 详情 |
|---|---|
| **三算法膝过伸全超限** | 除纯PPO 外: PPO+VMC 1.045 / SRL 1.04 / SRL+VMC 0.955, 全部超过膝关节 ±0.85 极限 → 机械安全隐患 (关节限位) |
| **SRL/SRL+VMC 站立本身不稳** | 站立期 \|gyro\| 2~5 rad/s, "跳高"测量部分含弹跳能量, 非干净蹲-跳 |
| **FSM 周期不随 trigger 停止** | FSM 离开 idle 后走完整周期 (0.45~0.5s crouch + thrust + flight + landing), 短脉冲 (0.8s) 会跳 1.5 个周期; 属 FSM 设计, 非 bug, 但交互体验差 |
| **纯PPO 姿态最干净** | hip_roll/膝过伸全在限内, 站立末期 \|gyro\| 0.2, 只是不会跳 |

---

## 3. 判定与优先级

| 问题 | 算法 | 严重度 | 是否需修复 |
|---|---|---|---|
| P3 跳后持续蹲-起振荡 (无指令也动) | SRL | 🔴 高 (微动平衡 §1.4 直接违反) | **必修** |
| P4 髋外展跳跃 + 无下蹲 | SRL+VMC | 🔴 高 (姿态异常 + 髋受力) | **必修** |
| 膝过伸超 ±0.85 | PPO+VMC / SRL / SRL+VMC | 🟠 中高 (机械安全) | 必修 (三算法) |
| P2 下蹲浅 + 起跳慢 | PPO+VMC | 🟡 中 | 改善 |
| P1 无跳跃 | 纯PPO | ⚪ 固有极限 | 接受或换参考结构 |
| FSM 周期不随 trigger 停止 | SRL / SRL+VMC / PPO+VMC | 🟢 低 | 交互体验优化 |

## 4. 建议修复方向 (待批准, 已按文献调研 §5 修正)

1. **膝过伸 (三算法通用, 机械安全最高优先)**: VMC 逆运动学求"膝角 ±0.85"对应虚拟腿长可行域 `[L0_min_knee, L0_max_knee]`, 裁剪 `_jump_leg_reference` 的 thrust_length 与 `apply_action` 的 L0; 纯PPO/SRL 在 `apply_action` 裁剪膝动作 (文献: MARCO Hopper II 把膝限当 VMC 核心约束; CaT/动作投影)。
2. **P4 髋外展**: `jump_srl_vmc.py` 跳跃期 (thrust/flight/landing) 加 `lateral_posture` 奖励 `−(|roll_L−0.1|+|roll_R+0.1|)` + 左右对称 `−(roll_L+roll_R)²`, 或给 `jump_height`/`height_progress` 加 `roll_ok` 门控 (文献: landing/lateral symmetry 姿态奖励)。
3. **P3 站立振荡**: ①策略输出加一阶/Butterworth LPF (文献: 4Hz LPF 消抖动) ②站立窗 `ang_vel_xy` 加强 + 显式站立稳定奖励 (文献: standing cost 需显式) ③`landing_recovery` 窗口从 30 步延长覆盖整个恢复期 ④训练命令分布提高 trigger=0 站立期占比 (gait-conditioned 模式平衡)。
4. **P2 下蹲浅 + P4 无下蹲**: 修"触发即起跳"——加强 crouch 相 L0 参考主导 (fb 0.35→0.2) + "蹲到位才给 vertical_thrust"门控 (文献: CMJ 时序奖励先蹲满再蹬)。
5. **SRL+VMC 跳过 flight 态**: 检查 `_update_fsm_state` flight 条件 (vz<0 & z<dh+0.20) 与几何接触冲突, 修复后 jump_upright 才能生效。

## 5. 外部调研 (开发需借鉴 §2)

针对四问题查询文献后, 新增参考文档:

| 参考文档 | 覆盖问题 | 核心借鉴 |
|---|---|---|
| `docs/references/2026-08-16_jump_standing_stabilization_rl.md` | P3 站立振荡 | 站立显式奖励 + 动作 LPF (4Hz) + smoothing + gait-conditioned 奖励路由 + 站立命令课程 |
| `docs/references/2026-08-16_jump_posture_knee_limit_vmc.md` | P4 髋外展 + 膝过伸 | 侧向/对称姿态奖励 + 反向下蹲 (CMJ) 时序 + 虚拟腿 L0 膝限可行域裁剪 + CaT/动作投影 + W-SLIP 空中姿态 |
| (已有) `wheel_legged_lab.md` | SRL+VMC 同构工程 | 同款轮腿机器人 Isaac Lab 的 VMC+六态 FSM, 阶段增益/分阶段跳跃奖励可对照 |
