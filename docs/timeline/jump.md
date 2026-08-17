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
| 2026-08-15 | 四算法基线评估 | 08-14 model_9999 实测四算法: verify_jump + eval_jump_repeat + diag。纯PPO 几何接触下实为火箭 (2.9m/air97%/0%成功率); SRL 0.59m/100%; PPO+VMC 假跳 (air2%); SRL+VMC 0.42m/81% `[[assess: 2026-08-15_four_algo_baseline_assess.md]]` | 定位纯PPO/PPO+VMC 两大静默失效 bug: ① landing_recovery 符号 bug (恒0) ② VMC 力阈值接触检测 (air 奖励恒0, PPO+VMC air_frac 实测 0.02 实为 0.20) | 四算法全未达 §7.5 |
| 2026-08-15 | 纯PPO+PPO+VMC bug 修复 + v3 重训 | 修 landing_recovery 符号 (jump.py) + VMC XML 加 framepos 传感器 + VMC env 改几何接触 (commit 96add2f); 两算法 4000 iter 并行重训 `[[NN_jump_ppo_vmc_bugfix_v3]]` | 测试 10 passed; landing_recovery iter 28 即发放 (1.94, 修复前恒0); PPO+VMC 基线 air_frac 0.02→0.20 (几何检测) | 训练中, 待评估 |
| 2026-08-15~16 | 四算法 10000 轮训练 + 完整评估 | 四算法各 10000 iter/1024 env (供对比实验), 训练后 eval_jump_full 六阶段完整评估 `[[assess: 2026-08-16_four_algo_final_assessment]]` | 纯PPO 0.089m (固有极限); SRL 0.224m 7/7; PPO+VMC 0.304m 6/7; SRL+VMC 0.183m 6/7 | 姿态普遍差: SRL 深蹲过深/恢复晃动; SRL+VMC 恢复晃动 | 
| 2026-08-16 | SRL v5: height_progress bug 修复 | 修 `_reward_height_progress` 时序 bug (episode_max_height 更新前存 prev); 训练 4000 iter `[[NN_jump_srl_v5_heightfix]]` | 跳高 0.347→0.277m (bug 修复生效), 但站姿 \|gyro\| 1.6、落地未恢复站立 | 跳高 vs 姿态权衡; 收敛后站姿晃动加剧 |
| 2026-08-16 | SRL+VMC v5: anti-sway | kd_l0 5→8 + ang_vel_xy -0.2 + orientation -10, 训练 4000 iter `[[NN_jump_srl_vmc_v5_antisway]]` | iter-1000 跳高 0.267m, **但收敛退化到 0.134m** (anti-sway 过强→安全低跳局部最优, height_progress 恒0) | v4 教训重演: 过强姿态惩罚压垮跳高 |
| 2026-08-16 | SRL v6: 加强姿态项 | ang_vel_xy -0.15 + landing_recovery 4→8 + anti_drift -3, 训练 4000 iter `[[NN_jump_srl_v6_posture]]` | **跳高 0.268m + 落地恢复 58→0 步** + 站姿 \|gyro\| 1.595→1.321 → 定稿 | 站姿轻微晃动残留 |
| 2026-08-16 | SRL+VMC v6: 重平衡 | ang_vel_xy -0.1 + orientation -5 + jump_height 30 + height_progress 25 + kd_l0 12, 训练 4000 iter `[[NN_jump_srl_vmc_v6_rebalance]]` | **跳高 0.247m (退化解决) + 站姿 \|gyro\| 3.6→1.36** + 最终姿态完全正常 → 定稿 | 站姿轻微晃动残留 |
| 2026-08-16 | 可视化问题量化诊断 | 负责人交互查看器观察四问题 → 写 diag_jump_problems/postjump 脚本量化 (trigger 精确受控) `[[assess: 2026-08-16_four_algo_visual_problems_diag]]` | ①纯PPO 下蹲0.006m/跳高0.089 (固有极限) ②PPO+VMC 下蹲浅0.07+**膝过伸1.045超±0.85** ③SRL 跳后trigger=0持续蹲-起 (FSM全idle, 策略自持振荡, \|gyro\|最高5.2) ④SRL+VMC **髋外展0.611+无下蹲+跳过flight相** | 三算法膝过伸全超限; SRL/SRL+VMC 站立本身不稳; FSM 周期不随 trigger 停止 |
| 2026-08-16 | 外部调研参考 | 按 §2 查文献: 站立稳定 RL (显式站立奖励/LPF/gait-conditioned) + 跳跃姿态/膝限 (侧向对称奖励/CMJ/VMC膝限约束/CaT) `[[2026-08-16_jump_standing_stabilization_rl]] [[2026-08-16_jump_posture_knee_limit_vmc]]` | 修复方案有文献依据: 站立须显式奖励; 外展轴须姿态奖励; 膝限须在 VMC 前馈/动作层约束 | — |
| 2026-08-16 | SRL v7: 站立稳定修复 | landing_recovery 改为站立期恒开 (standing cost) + action_smoothing 0.3 (LPF) + ang_vel_xy -0.15→-0.5 + 膝目标裁剪 ±0.85 `[[NN_jump_srl_v7_standing_fix]]` | **✅ P3 解决**: 站立 \|gyro\| 0.92→0.51 (末期 0.06), 跳后 2.05→0.20, 无自跳, 跳高 0.266→**0.329m**; 髋外展 0.04 干净; 膝 1.023 (动态瞬态残余) | 膝过伸为高速蹬伸瞬态 (MuJoCo 允许 ~0.13 过冲) |
| 2026-08-16 | SRL+VMC v7: 髋外展修复 | 新增 lateral_posture 15 (跳跃期髋外展惩罚) + jump_height/height_progress 门控 window_crouched (先深蹲) + vmc.py 膝守卫 (前馈reflex削减+力矩硬守卫) `[[NN_jump_srl_vmc_v7_abduction_fix]]` | ①首轮 (对称守卫) iter-1000: 髋外展 0.611→0.074 ✅ 深蹲 0.271 ✅ 但**蹲了不跳 (stutter)** → 守卫对称|knee|误削深蹲支撑 → **改方向性守卫**, v6验证膝 0.897→0.743 ②v7b: 外展 0.277 ✅ 但收敛塌缩 (跳高 0.123m, 站立 2.25) ③v7c: 补 LPF+ang_vel-0.4, iter-1000 站立 0.68/跳 0.255m, 但收敛后站立 3.06 仍压不住 | 髋外展 P4 持续修复; 站立振荡成唯一顽固项 |
| 2026-08-16 | SRL+VMC v7d: 站立专项惩罚 | 新增 standing_still 奖励 (站立期 \|gyro\|>1 阶梯惩罚 -4) + kd_l0 12→20 `[[NN_jump_srl_vmc_v7_abduction_fix]]` | 负责人改变方向 → **弃用**, 改为干净消融 (见 v8) | 站立振荡追了 3 轮未根治, 转向方法论层面 |
| 2026-08-16 | SRL+VMC v8: 干净消融 | 负责人思路 "SRL+VMC 只输出层与 SRL 不同" → 统一奖励 (SRL 集, 去 VMC 专属塑形) + 统一观测 (315D 关节) + 仅控制层不同 (关节PD vs 虚拟腿VMC) `[[NN_jump_srl_vmc_v8_clean_ablation]]` | 发现 SRL+VMC 参考双重混合 bug (step 混两次 → 参考×2), 修复后 v8b 重启 | 消融对比才可归因控制层 |
| 2026-08-16 | PPO+VMC v9: 无参考消融臂 | 负责人思路 "PPO+VMC 只输出层与纯PPO 不同" → 去 SLIP 参考混合/分阶段增益/FSM观测, 奖励观测对齐纯PPO (297D) `[[NN_ppo_vmc_v9_no_reference_ablation]]` | **2×2 消融最终**: 纯PPO 0.081m / PPO+VMC v9 0.090m / SRL v7 **0.328m** / SRL+VMC v8b 0.197m(髋外展 0.627)。结论: ①SLIP参考是跳跃关键 (无参考对 << 有参考对) ②有参考对里关节PD(SRL) > VMC(SRL+VMC), VMC 靠外展借力 (锁roll 仅 0.087m) | 附带修复 SRL+VMC 双混合 (参考×2) |
| 2026-08-16 | SRL+VMC v8d: 实用版调优 | 负责人选 B: 加回 lateral_posture 15 (压外展) + 强蹬伸 (ff 3.5/kp 3.0/kd 0.3) + 深蹲 (0.24) + 加长 crouch (0.5) `[[NN_jump_srl_vmc_v8d_best_tuning]]` | iter-1000 跳高 0.273m/外展 0.296 理想, 但收敛后塌缩到 0.12m + 起跳仍外展 (负责人验证) | 深度诊断: ①"0.156m天花板"为测试缺陷 ②真根因: 蹲下相前馈对抗压缩/奖励梯度小跳划算/起跳roll脉冲 |
| 2026-08-16 | SRL+VMC v8e: 根因修复 | 3 项修复: ①蹲下相支撑前馈归零 ②跳高最低门槛 (base_z>目标+0.15) ③lateral_posture 加 roll 速度惩罚 `[[NN_jump_srl_vmc_v8e_rootfix]]` | 重训中 (GPU0) — 待评估 | 冲着根因, 不再盲目调参 |

## 当前状态
- **四算法最终定稿**: SRL v6 (0.268m) / SRL+VMC v6 (0.247m) 达标; PPO+VMC (0.303m) 跳高最高但膝过伸; 纯PPO (0.097m) 姿态优秀但跳高固有极限
- 🔧 **v7 修复进行中**: SRL/SRL+VMC 4000 iter 重训中 (GPU0/1); 改动含站立期显式奖励/LPF/外展惩罚/下蹲门控/膝守卫 (基于文献调研, 见参考文档)
- ⚠️ 已确认四问题 (可视化诊断): ①纯PPO 下蹲0.006m (固有) ②PPO+VMC 膝过伸1.045→守卫后0.884/跳高0.217m ③SRL 跳后持续蹲-起 (策略自持振荡) ④SRL+VMC 髋外展0.611+无下蹲
- 详细诊断报告: `_devlog/assess/reports/jump/2026-08-16_four_algo_visual_problems_diag.md`
- 参考文档: `docs/references/2026-08-16_jump_standing_stabilization_rl.md` + `2026-08-16_jump_posture_knee_limit_vmc.md`

## 下一步
- [x] SRL/SRL+VMC v5 姿态问题修复 → v6 定稿
- [x] 外部调研 + 参考文档 (§2 开发需借鉴)
- [~] **SRL v7 / SRL+VMC v7 重训 + 重新诊断** (训练中)
- [ ] PPO+VMC v7 重训 (膝守卫已生效, 重训回收跳高; 待 GPU 空闲)
- [ ] 渲染四算法最终跳跃视频 + 导出姿态 CSV + 版本备份
- [ ] 论文 caption 数值按真实数据修订 (reward≈99 / jump≈2.2 / std≈0.8)
