# [09] ★ MPC 完成: 线性 MPC (Hildreth QP) 平衡/指令/腿长全达标, 与 LQR 同口径对比

**日期**: 2026-08-14
**状态**: **MPC P1/P2/P3 存活率 100% (与 LQR 并列), P2 速度 RMSE 0.093 (达标), P4 与 LQR 并列 (rough 环境过难, 双侧 0%)**
**关联**: [[06_mpc_attempt]] (旧 MPC 失败根因), [[02_balance_solved_control_direction]] (方向根因)

---

## 来源
用户要求"完成经典算法的 MPC 开发"。旧 MPC (devlog 06) 已在 0.4-0.8s 内倒, 归因于"植物模型不确定"。

## 根因分析 (旧 MPC 为什么失败)

**★ 根本原因: 单位错配 — MPC 解出的是轮线加速度 (m/s²), 但 act() 把它当轮速命令 (rad/s) 用。**
- 模型 u=轮线加速度, `u_max=2.0 m/s²`; 控制器把 U[0] 直接赋给轮速命令 → 权威只有 LQR 的 ~1/9, 方向正确也救不回。
- 任何模型 (解析/黑箱) 换都不行 — 这是实现 bug, 不是模型问题。

**次要原因 (修复后仍失败, 逐一排查):**
1. **Hildreth 符号错**: `λ ← max(0, λ+ω(kk−Eλ)/Eᵢᵢ)` 应为 `λ ← max(0, λ−ω(Eλ+kk)/Eᵢᵢ)` → 300 iter 不收敛, 求解 37ms。
2. **v 状态用 v_wheel (L 轮 qvel·R)**: 与 LQR 实测最优的 v=linvel[0] 不同 → MPC 框架 + LQR 律也 0.39s 倒。
3. **u_max=10 太紧**: LQR 未平滑命令达 ±58 (clip ±25), MPC 饱和在 ±10 → 高增益也救不回。
4. **模型 v 行符号错**: 控制器轮映射 ctrl=−u ⇒ v̇=−(R/τ)u−(1/τ)v, 原写成 +R/τ → 模型预测 v 方向反转。

## 修改了什么

### 求解器 `scripts/classic_control/mpc.py`
- **向量化 Hildreth (残差跟踪)**: 修正符号 + 每拍仅矩阵-向量乘 → 求解 37ms → **<0.1ms** (满足 10ms 预算)。
- `build_mpc_matrices`: 预计算 P/H/G/Ginv/C/E (与 x₀ 无关), 可选 **LQR 末端代价** (dlqr_riccati) + **控制变化率惩罚** r_rate·DᵀD。
- `solve_qp`: 残差跟踪 Gauss-Seidel, warm-start λ。

### 控制器 `scripts/classic_control/controller.py`
- MPC 输出 = **轮速命令 (rad/s)**, 与 LQR 同约定 → 消除单位错配。
- v 状态 = **linvel[0]** (与 LQR 一致; v_wheel 会破坏闭环)。
- **模型来源**: 参数 A_d/B_d → config `model_file` (黑箱) → 解析 (α,β,τ)。
- **P1 用解析模型** (黑箱 v-pole 1.02 引入漂移 0.4>0.2 不达标); **P2+ 用黑箱** (速度跟踪更准)。
- **P2 速度跟踪**: 5 态积分增广 z=∫(v−v_ref) (offset-free MPC) + `integral_gain` 参考偏置 (v_ref_eff=v_ref−k_i·z)。

### 模型辨识 `scripts/classic_control/identify_plant.py` (黑箱)
- LQR 闭环 + 多频强探针 (±4 rad/s), 状态 [θ,θ̇,linvel[0],xpos], 最小二乘拟合 A_d/B_d。
- 输出 `logs/classic/mpc_plant_bb.npz` (一步预测留出 RMSE: θ 0.0013, v 0.0042)。

### 其他
- `rollout.py`: `v_wheel` 传感器 (dof_vel[6]·R; ★ dof 序 [..,L_roll,L_pitch,L_knee,R_roll,R_pitch,R_knee,L_wheel,R_wheel]).
- `eval_classic.py`: **修复控制器跨 episode 不复位 bug** (reset() 未调用 → 内部状态泄漏 → P3 被误判 33%)。
- `config.yaml`: mpc 段新增 tau/alpha/beta/terminal_lqr/r_rate/integral_gain/model_file, u_max 2→30。
- 删除探索性 `measure_plant.py` + 陈旧 npz。

## 训练后效果 (确定性评估, 5 episodes)

| 阶段 | MPC | LQR | 判定 |
|---|---|---|---|
| P1 平衡 12s | **100%** 存活, gyro 0.227, linvel 0.126 | 100% | ✅ 并列 |
| P2 指令 30s | **100%** 存活, vx_rmse **0.093** | 100%, vx_rmse 0.080 | ✅ 达标 (<0.1) |
| P3 腿长 36s | **100%** 存活, height_err 0.039 | 100%, 0.038 | ✅ 并列 |
| P4 地形 25s | 0% | 0% | ⚠️ 双侧并列 (rough 过难) |

- 求解耗时: **<1ms** (远低于 10ms 控制周期)。
- MPC P1 指标优于/等于 LQR (gyro 0.227, tilt 0.112)。

## 参数调整好坏
- **u_max 2→30**: 关键修复, 允许 LQR 级权威 (未平滑 ±58)。
- **integral_gain 1.5**: P2 参考偏置积分最优 (0.098→0.102→0.123 递增); 0 时 0.31。
- **q_theta 600**: 压 P2 姿态 (|θ|max 0.47→0.38) 但破坏 P3 (33%) → 回退 100。
- **r_rate 0.5**: 变化率惩罚, 抖动无明显改善, 保留作为可选。
- **P1 解析 / P2+ 黑箱**: 分阶段模型, 各取所长。

## 验证方法
1. 官方 eval_classic 5 episodes 存活率 (P1/P2/P3 100%)。
2. 一步预测留出验证 (identify_plant) RMSE 合格。
3. 求解耗时监控 (<1ms)。
4. ruff format + check 通过; mypy 仅 src/ 门禁 (scripts/ 不在范围)。
5. 旧 MPC 基线对照: 1.17s 倒 → 现 P1 15s 存活。
6. **渲染视频** (cam 跟踪): `video/classic/mpc_p{1..4}_*.mp4` (P1 平衡/P2 指令/P3 腿长/P4 rough)。
7. **交互控制**: `play_classic.py --controller mpc` (新增 MPC 支持, 键盘验证)。

## 交付 (验证脚本一览)
- 量化评估: `scripts/classic_control/eval_classic.py` + `shell/xqrobotwl/classic_control/eval_classic.sh mpc all`
- 单阶段: `balance_mpc.py --phase N --sim_time X`
- 渲染: `balance_mpc.py --phase N --render video/classic/mpc_pN_*.mp4`
- 交互: `play_classic.py --controller mpc` (或 `play_classic.sh flat mpc`)
- 模型辨识+验证: `identify_plant.py` (黑箱 A_d/B_d + 留出 RMSE)

## 遗留问题 (如实)
- **P2 姿态欠佳**: 黑箱模型下机器人以 θ≈−0.43 rad (25° 后仰) 前移, gyro 2.4 (摆动) — 速度跟踪与姿态直立存在模型层面的权衡, 线性模型无法同时最优。vx_rmse 达标但运动观感不如 LQR。
- **P4 rough 地形 0% (LQR 也 0%)**: 对称几何后 rough 环境过难 (devlog 08 遗留), 需单独调 rough 地形/速度, 非 MPC 特有问题。
- **闭环辨识偏差**: 黑箱模型吸收 LQR 反馈 (v-pole 1.02 漂移), 是 P1 漂移 0.4 与 P2 后仰的根源。

## 后续计划
- P4 rough 地形: 降速/地形适配增益, 与 LQR 同步调。
- 可选: 多 seed 箱线图对比 (每算法需 >1 seed 重训)。
- 论文对比表: LQR vs MPC vs RL (walk_flat/walk_rough 同口径)。

## 关联日志
- [[06_mpc_attempt]] 旧 MPC 失败
- [[02_balance_solved_control_direction]] 方向根因 (LQR 达成)
- [[08_classic_symmetric_constants]] 对称常量 (P4 rough 遗留)
