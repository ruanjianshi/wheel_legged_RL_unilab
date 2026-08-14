# xqrobotwl/jump/ppo 开发索引

XqRobotWL 跳跃训练 — PPO 算法 + phase-gated 奖励 / Wheeled-SRL。

## 2026-08-06

| 序号 | 标题 | 文件 |
|------|------|------|
| 16 | 新增 PPO+VMC 与 SRL+VMC 两种跳跃算法 (论文 2×2 对比) | [→](2026-08-06/16_ppo_vmc_srl_vmc_algorithms.md) |
| 17 | 修复 VMC 版本无法起跳: feedforward 不足 + L0 动作范围过小 | [→](2026-08-06/17_vmc_jump_param_fix.md) |
| 18 | 修复 SRL+VMC: FSM 参考主导 + 策略残差 | [→](2026-08-06/18_srl_vmc_fsm_reference_dominant.md) |
| 19 | 四算法跳跃训练完成 + 对比评估 | [→](2026-08-06/19_four_algo_training_compare.md) |
| 20 | 方案A: PPO+VMC 加 SLIP-FSM 参考(全动作) + 纯PPO 落地存活修复 | [→](2026-08-06/20_vmc_slip_refactor_ppo_landing.md) |

## 2026-08-07

| 序号 | 标题 | 文件 |
|------|------|------|
| 21 | 四算法最终对比 + 论文图 (优化完成) | [→](2026-08-07/21_final_compare_paper_figs.md) |
| 22 | 纯PPO v5 重训 (landing_recovery 4.0) + VMC 变体重训 | [→](2026-08-07/22_vmc_v3_pure_ppo_v5_retrain.md) |
| 23 | 修复 verify_jump 配置漂移 bug: 验证 env 与训练配置不一致 | [→](2026-08-07/23_verify_jump_config_drift.md) |

## 2026-08-08

| 序号 | 标题 | 文件 |
|------|------|------|
| 24 | 最终四算法对比 + 论文图 (10000 轮, 修复后配置) | [→](2026-08-08/24_final_four_algo_compare_figs.md) |
| 25 | 诊断三个算法跳跃弱点 + 针对性改进 (±50 力矩 / 长 FSM 时序 / 破局奖励) | [→](2026-08-08/25_jump_weakness_diag_fix.md) |
| 26 | VMC 变体改进完成 + 最终四算法对比 + 论文图 | [→](2026-08-08/26_final_compare_improved_vmc.md) |
| 27 | 定位并修复纯PPO landing_soft 白嫖 exploit (落地奖励门控 + 二次蹬伸) | [→](2026-08-08/27_landing_soft_exploit_fix.md) |

## 2026-08-09

| 序号 | 标题 | 文件 |
|------|------|------|
| 28 | 纯PPO 奖励对抗三连 (v10-v12): 从白嫖到真实跳跃 (0.203m / air 11%) | [→](2026-08-09/28_pure_ppo_reward_hacking_v10_v12.md) |
| 29 | 论文图重出: 训练曲线 0.8 EMA 平滑 + 参考风格高度曲线 (2x2 + FSM 相位) | [→](2026-08-09/29_paper_figs_smooth_single_view.md) |
| 30 | 论文图 nature/dataviz 规范重出: 新增图4.1 训练指标 (2x3) + 全图统一风格 | [→](2026-08-09/30_paper_figs_nature_style.md) |

## 2026-08-14

| 序号 | 标题 | 文件 |
|------|------|------|
| 36 | 论文图 v2.0 重出: 严格遵循 nature-draw.md (IBM 配色/EMA 0.02/PDF+PNG600) | [→](2026-08-14/36_paper_figs_v2_nature_draw.md) |
| 37 | 论文补充图 5 张 (训练全景/最终性能/轨迹+速度/关节2×3/奖励分项) + 图片归档规范 picture/paper | [→](2026-08-14/37_paper_supp_figs_and_archive.md) |

## 2026-08-11

| 序号 | 标题 | 文件 |
|------|------|------|
| 35 | 小论文重构: 单方法占位稿 → 四算法 2×2 对照研究 (真实数据+图件+参考文献) | [→](2026-08-11/35_rewrite_small_paper_4algo.md) |

## 2026-08-10

| 序号 | 标题 | 文件 |
|------|------|------|
| 34 | 补全 xqrobotwl/jump 缺失的 VMC 验证脚本 | [→](2026-08-10/34_add_jump_vmc_eval_scripts.md) |

> 仓库管理类日志 (瘦身/shell/清理/批量训练) 见 [`../repo/INDEX.md`](../repo/INDEX.md)

## 2026-07-25

| 序号 | 标题 | 文件 |
|------|------|------|
| 13 | lean_forward trigger-gating + 站姿修复 + 最终收敛 | [→](2026-07-25/13_lean_forward_final.md) |

## 2026-07-24

| 序号 | 标题 | 文件 |
|------|------|------|
| 12 | 符号 bug + entropy + FSM 时序全面修复 | [→](2026-07-24/12_sign_bugs_entropy_fsm_fixes.md) |

## 2026-07-22

| 序号 | 标题 | 文件 |
|------|------|------|
| 08 | Wheeled-SRL 框架构建 & 训练诊断修复 | [→](2026-07-22/08_wheeled_srl_framework_diag.md) |
| 09 | FSM 参数修复 + 预热逻辑 + 自适应增益 | [→](2026-07-22/09_fsm_warmup_adaptive_gain.md) |
| 10 | 站姿参数修复 + 高度目标修正 | [→](2026-07-22/10_stand_height_platform_fix.md) |
| 11 | SRL 跳跃训练成功收敛 | [→](2026-07-22/11_srl_convergence.md) |
| 15 | jump gain 修复 (0.05→0.2) | [→](2026-07-22/15_jump_gain_fix.md) |

## 2026-07-18

| 06 | 跳跃姿态优化 + 站姿根因 | [→](2026-07-18/06_posture_knee_wheel.md) |

## 2026-07-17

| 序号 | 标题 | 文件 |
|------|------|------|
| 05 | 站立姿态 + 先蹲后跳 + 按键优化 | [→](2026-07-17/05_stand_still_crouch_first.md) |

## 2026-07-16

| 序号 | 标题 | 文件 |
|------|------|------|
| 03 | 姿态修正 + 腾空优化 | [→](2026-07-16/03_posture_airtime.md) |

| 04 | 键盘控制修复 | [→](2026-07-16/04_keyboard_control.md) |

## 2026-07-15

| 序号 | 标题 | 文件 |
|------|------|------|
| 02 | Kp 适配 + 存活率 + 后仰修正 | [→](2026-07-15/02_kp_survival_posture.md) |

## 2026-07-14

| 序号 | 标题 | 文件 |
|------|------|------|
| 01 | Phase-gated 奖励 + 多轮迭代 | [→](2026-07-14/01_phase_gated_rewards.md) |

## 2026-07-12

| 序号 | 标题 | 文件 |
|------|------|------|
| 01 | jump 训练配置 | [→](2026-07-12/01_jump.md) |
