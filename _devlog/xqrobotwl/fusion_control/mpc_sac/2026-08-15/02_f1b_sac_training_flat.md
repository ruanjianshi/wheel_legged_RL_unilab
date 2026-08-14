# [02] F1b 平地 SAC 高层训练 + 验证 (融合控制)

**日期**: 2026-08-15
**状态**: 完成 — F1 平地验收全部达标 (存活率 100%, vx_rmse 0.039, gyro 0.136, θ 违犯率 0)
**关联**: [[2026-08-14/01_f1a_branch_skeleton_mpc_port]] (骨架+MPC 移植), [[INDEX.md]]

---

## 来源
F1b: 在 meta_env (低层 MPC 冻结) 上训练高层 SAC 策略 (残差式), 平地验证。

## 架构 (AugMPC 层次化移植)
```
高层 SAC (obs 11D: θ,θ̇,v,ω_z,base_z,des(3),prev_a(3))
  → 残差校正 a∈[-1,1] → cmd = des + res_scale·a  → 低层 MPC → a8 → env
```
**残差式** (★ 关键): RL 输出"校正"而非直接命令 → a≈0 时 cmd≈des → MPC 基线直接跟踪。
MPC 已扛平衡+速度跟踪, RL 只学需要的校正 (防自旋/地形自适应/补偿模型误差)。

## 训练设置
- 64 env 向量化 (每 env 独立 MPC), SAC (GaussianActor + TwinQ + 温度自学习), CUDA
- 3000 iter, obs 归一化 (RunningMeanStd), reward: 存活 + vx/vyaw/height 跟踪 + θ²/ω²/残差²惩罚
- run: `logs/fusion_control/mpc_sac/walk_flat/2026-08-15_00-25-57_f1/model_final.pt`

## 验收结果 (5 ep, vx=0.4)
| 指标 | 融合 | 阈值 | 判定 |
|---|---|---|---|
| 存活率 (15s) | **100%** (5/5) | ≥95% | ✅ |
| 平均时长 (35s sim) | **31.75s** (均 >10s) | ≥10s | ✅ |
| gyro_rms | 0.136 | <1 | ✅ |
| vx_rmse | **0.039** | <0.1 | ✅ |
| height_err_mean | 0.017 | <0.05 | ✅ |
| θ_max 违犯率 | 0 | =0 | ✅ |
| v_max 违犯率 | 0 | =0 | ✅ |
| cmd_track_err (残差) | 0.010 | — | a≈0 不发散 |
| solve_ms_mean | 0.14ms | <10ms | ✅ |

对比: 纯 MPC P2 vx_rmse 0.087 → 融合 0.039 (残差学习改善速度跟踪)。

## 调试历程 (数据驱动, 5 轮根因闭环)
| 轮 | 现象 | 根因 | 修复 (commit) |
|---|---|---|---|
| 1 | eval 100→22% 退化 | 高层直接发令, 策略学"自旋" (vyaw 0.86 失控) | 收窄 vyaw ±0.05 + ω_z² 惩罚 (6382894 前) |
| 2 | 策略震荡欠发令 | policy+MPC 闭环震荡 | 改**残差式** cmd=des+res_scale·a (6382894) |
| 3 | 起步 0.6s 跌倒 | MPC cmd_ramp 首步跳变 (经典轨 bug, 评估用站姿起步掩盖); vyaw≠0 也不稳 | ramp 从 0 起步 + meta_env 站姿预热 1.5s + runner 前插 2s 站姿 (9dfb157) |
| 4 | eval 跨 episode 失败 (5s) | runner 只 init 一次 env | 无条件 env.init_state() → 24.4s (40562d4) |
| 5 | vx_rmse 0.151 过冲 | 策略学 +0.14 超发令 | w_corr=-0.3 残差²惩罚 + res_vx 0.3→0.2 → vx_rmse 0.039 |

**关键发现**: 两轮足 MPC 的 yaw 差分跟踪不稳 (vyaw 0.15 即 6s 跌倒), 经典评估因命令表 vyaw=0 未暴露;
mpc_sac 融合轨把训练/评估收窄到 MPC 已验证范围 (vx±0.6, vyaw±0.05) — 地形自适应留给 F2 rough。

## 数据+视频
- 报告: `logs/fusion_control/mpc_sac/report_f1_vx04_15s.md` (5ep 100%)
- 视频: `video/fusion_control/mpc_sac/f1_fusion_vx04_15s.mp4`
- 训练日志: `logs/fusion_control/mpc_sac/train_f1.log` (eval@200-3000 全 100%)

## 后续计划
- F2: rough 融合 (高层 obs 加地形, 残差校正腿高/速度 → 治 pure MPC/RL rough 0%)
- F3: 三栏对比 (RL vs MPC vs 融合)

## 关联
- [[../INDEX.md]] [[2026-08-14/01_f1a_branch_skeleton_mpc_port]]
- 经典轨: [[../../classic_mpc/INDEX.md]] [[../../classic_lqr/INDEX.md]]
