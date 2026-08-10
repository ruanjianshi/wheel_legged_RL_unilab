# 时间线 · 跳跃 (xqrobotwl_jump_flat / jump_srl / jump_vmc / jump_srl_vmc)

> 一句话: 两轮足机器人按键触发的蹲→蹬→腾空→落地跳跃, 四算法 (纯PPO / SRL / PPO+VMC / SRL+VMC) 2×2 对比 — 已达标 (四算法收敛, 论文对比图已出; 纯 PPO 后期退化待确认)。
> 来源: _devlog/xqrobotwl/jump/ppo/  (实时更新, 每完成一个阶段追加一行)

| 日期 | 阶段 | 做了什么 | 影响/效果 | 问题与解决 |
|---|---|---|---|---|
| 2026-07-12 | 起步 | 新建 `XqRobotWLJumpFlat` PPO 配置 (1024 envs × 24, 10000 iter, tracking=2.0, crouch_prep=2.0, wheel_air_time=2.0) `[[01_jump]]` | — | — |
| 2026-07-14 ~ 07-15 | v1 奖励设计 | Phase-gated 跳跃奖励 (crouch/thrust/height/air/landing 按 phase 门控) + 权重重平衡 (thrust 4→30, air 2→20, leg_mirror -0.5→-3); 后 Kp 回退 30 + anti_loiter 反蹲罚 `[[01_phase_gated_rewards]] [[02_kp_survival_posture]]` | wheel_air_time 0.005→3.43 (会跳) → 后蹲而不跳 (phase 窗口锁死); ep_len 47→943 | 蹲姿局部最优 (蹲住混分), thrust 窗口锁死, 后仰冒充蹲 |
| 2026-07-16 ~ 07-18 | 姿态修正 | 腾空蹬腿修复 (vertical_thrust 轮地门控), hip_roll 约束防叉腿, 站立站姿 + 先蹲后跳 (thrust 门槛 phase≥25), 膝锁死修复 (膝罚平方化 + jump_height 膝门控), 轮空转罚 `[[03_posture_airtime]] [[05_stand_still_crouch_first]] [[06_posture_knee_wheel]]` | max_z 0.98→1.03m, air_steps 293/500→360/500, 站立 z 0.46→0.49, 站立存活 95步→200步 | DEFAULT_LEG_ANGLES 非稳定平衡点 (站姿根因); 不蹲就跳; 按一次跳两次 (H 键 150→80 帧) |
| 2026-07-22 ~ 07-25 | Wheeled-SRL 框架 | 构建 jump_srl.py (SLIP-FSM 六态前馈 + 相位门控奖励), FSM/预热/自适应增益修复, 站姿与高度目标修正 (0.65→0.55), 符号 bug + entropy + FSM 时序修复, lean_forward trigger-gating `[[08_wheeled_srl_framework_diag]] [[09_fsm_warmup_adaptive_gain]] [[10_stand_height_platform_fix]] [[11_srl_convergence]] [[12_sign_bugs_entropy_fsm_fixes]] [[13_lean_forward_final]]` | SRL iter 3655: ep_len 977 (98%), action_std 0.11 收敛; 站姿后仰 37°→5° (8x); 模型备份 `backup/XqRobotWLJumpSRLFlat/jump_srl_ppo_v1/` | action_std 爆炸 5.98 (feedback_gain 0.2 放大噪声); 预热只 3 iter (×num_envs 遗漏); fsm_tracking 符号反; entropy_coef 0.03 太高 |
| 2026-08-06 | 四算法选型 | 补齐 PPO+VMC 与 SRL+VMC (虚拟腿 VMC 力矩控制), 构成 2×2 (关节空间 vs 虚拟腿 × 纯 PPO vs SRL); 数值标定 VMC FK (l1=0.3005/l2=0.3007, 双腿 RMSE<3mm) `[[16_ppo_vmc_srl_vmc_algorithms]]` | 10 项 VMC 测试通过; obs PPO+VMC 369/468, SRL+VMC 387/486 | 膝关节在 qknee=0 已弯 ~96°, 腿无法完全伸直, 需重新标定符号约定 |
| 2026-08-06 | VMC 起跳修复 | feedforward_force 80→110 N (>体重/腿 ~91N), action_scale_l0 0.05→0.12 (覆盖下蹲 0.28~蹬伸 0.50); SRL+VMC 混合公式翻转为"FSM 参考主导 + 策略残差" `[[17_vmc_jump_param_fix]] [[18_srl_vmc_fsm_reference_dominant]]` | SRL+VMC 起跳 0.033→0.17m (开环); 不再策略抵消参考 | 80N 撑不住 18.65kg 无法离地; L0 动作范围够不到蹲/蹬; 策略学抵消 FSM 参考 |
| 2026-08-06 ~ 08-07 | 四算法对比 (首轮) | 四算法各 10000 iter + compare_jump 多速度对比; 纯 PPO 加 landing_recovery 落地恢复, PPO+VMC 加 SLIP-FSM 参考 (全动作) `[[19_four_algo_training_compare]] [[20_vmc_slip_refactor_ppo_landing]] [[21_final_compare_paper_figs]]` | 首轮: SRL 0.423m/air 21.6%/存活 100%, 纯PPO 0.496m/存活 0%; 优化后 SRL 0.437m/22.1%/100%, 纯PPO 0.286m/50%, PPO+VMC 0.053m, VMC+SRL 0.015m | 纯PPO 落地必摔 (存活 0%); 纯VMC 无法腾空 (air 0.5%, 力控难打破轮地接触) |
| 2026-08-07 | verify 配置漂移修复 | verify_jump/compare_jump 从 `run_config.json` 重建 env (kd_l0/feedback_gain 等与训练一致) `[[23_verify_jump_config_drift]]` | 旧 VMC+SRL air 3%→6%, 新 PPO+VMC 8%→9%; 历史 four_algo_comparison.json 数据污染清理 | verify 用硬编码默认配置与训练 Hydra 配置错位, 旧模型验证失真 |
| 2026-08-08 | VMC 改进 + 纯PPO 定案 | VMC 腿力矩 ±30→±50 N·m (蹬伸需 -60~-134), FSM crouch 0.25→0.35s/thrust 0.20→0.30s; 纯PPO v5-v9 多轮 anti-lazy/base_height 锚定尝试后定案 v5 `[[25_jump_weakness_diag_fix]] [[26_final_compare_improved_vmc]]` | PPO+VMC 跳高 0.271→0.382m (+41%), VMC+SRL 0.251→0.369m (+47%) 存活 0.80→0.95; 最终: SRL 0.550m/air 22%, VMC+SRL 0.369m, PPO+VMC 0.382m, 纯PPO 0.184m/air 0% | VMC 力矩饱和 ±30 (蹬伸无力); 纯PPO 6 轮假跳高 (air 恒 0%, base_height 罚 vs jump_height 奖死锁) |
| 2026-08-08 ~ 08-09 | 纯PPO 奖励对抗 | 定位 landing_soft 白嫖 exploit (phase≥30 站着不动每步拿 40) → 落地奖励门控 + vertical_thrust 二次; vertical_thrust→launch_rise (实际升高) 破局 `[[27_landing_soft_exploit_fix]] [[28_pure_ppo_reward_hacking_v10_v12]]` | 纯PPO 首次真实腾空 (探测 0.203m/air 11%; 全量 0.278m/air 8.5%/成功 100%); PPO+VMC v12 存活 100%/air 17.4% | v10 推而不蹲, v11 推而不起 (stutter), v12 用 base_z−下蹲最低点 破"代理量奖励"盆地 |
| 2026-08-09 | 论文图 | 0.8 EMA 平滑训练曲线 + 参考风格高度曲线 (2x2 + FSM 相位色带) + nature/dataviz 规范重出图 4.1 (2x3 训练指标) `[[29_paper_figs_smooth_single_view]] [[30_paper_figs_nature_style]]` | 输出 paper_fig_training/trajectory/validation/jump_joints 全套 PNG+PDF, Okabe-Ito 配色 4/4 PASS | 纯PPO 曲线末尾摔倒终止标 × (无参考不可控直观可见); 论文 caption 占位值须按真实数据修订 |
| 2026-08-10 | 交付 + 全量重训 | 补全 VMC 变体验证脚本 (eval_ppo_jump_vmc/srl_vmc.sh); 8 任务批量重训 (jump×4 各 10000 iter) `[[34_add_jump_vmc_eval_scripts]] [[repo/05_verify_8_trained_policies]]` | 8/8 eval 冒烟通过; 四算法 08-10 重训: jump_srl mean_reward 97.05 / jump_srl_vmc 94.68 / jump_vmc 87.92 / jump_flat 43.73 | ⚠️ jump_flat 最终迭代 jump_height=0.0000, 疑似后期退化回不跳跃 (reward hacking), 待确认 |

## 当前状态
- 四算法最优 checkpoint (thesis/experts/04): SRL 0.540m/air 22%/存活 100% > PPO+VMC v12 0.179m/air 17.4% > VMC+SRL 0.354m > 纯PPO v12 0.278m/air 8.5%/存活 50%
- 达标情况: ✅ 已训 (四算法收敛 + 论文 2×2 对比图已出); 核心结论 = SLIP-FSM 参考提供控制力与高度, 纯 PPO 无参考只能学到不可控小跳
- 遗留: jump_flat (纯 PPO) 08-10 重训最终迭代跳高 0.000 ⚠️, 需确认是否实际退化

## 下一步
- [ ] 确认 jump_flat 是否实际退化 (交互观察/评估), 退化则重训或加 anti-hack 奖励
- [ ] 论文 caption 数值按真实数据修订 (reward≈99 / jump≈2.2 / std≈0.8)
- [ ] 论文图 3.1 框架概览图 (按需)
