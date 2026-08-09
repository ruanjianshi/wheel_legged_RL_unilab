# XqRobotWL 跌倒恢复 (fall_recovery) PPO 开发索引

平地倒地 → 自恢复平衡。方案: **FTSR 力引导 + 高度分阶段奖励 + CPO 约束**
(2026-08-07 用户指定论文), 无脚本纯学习。

| 日期 | 序号 | 标题 |
|------|------|------|
| 2026-08-09 | 23 | [交付 — v7 model_4000: 4 姿态恢复 + yaw≈walk](2026-08-09/23_delivery_v7_4pose.md) |
| 2026-08-09 | 22 | [残留转圈+蹲姿 → 收紧 wheel_symmetry + 门控站姿奖励](2026-08-09/22_v7_tight_wheel_symmetry_stand_pose.md) |
| 2026-08-09 | 21 | [转圈更明显 → 轮速约束 wheel_symmetry + wheel_speed](2026-08-09/21_v6_wheel_symmetry.md) |
| 2026-08-08 | 20 | [v4 姿态约束过强致保持崩 → v5 温和约束](2026-08-08/20_v5_gentle_posture.md) |
| 2026-08-08 | 19 | [腿一前一后+转圈 → 自然站姿 dof_pos + no_yaw](2026-08-08/19_v4_natural_stance_no_yaw.md) |
| 2026-08-08 | 18 | [交付 — v3b 指数 gyro 门, 站立 0.4s→3.4-6.5s](2026-08-08/18_delivery_v3b_standing_time.md) |
| 2026-08-08 | 17 | [站立太短 → settle 加角速度静止项罚摇摆](2026-08-08/17_v3_gyro_stillness.md) |
| 2026-08-08 | 16 | [交付 — v2 姿态修复 + 4 姿态恢复视频](2026-08-07/16_delivery_4pose_posefix.md) |
| 2026-08-07 | 15 | [v1 过度约束致恢复率崩 → v2 宽松 band + 独立 stand_still](2026-08-07/15_v2_redo_overshoot_band_standstill.md) |
| 2026-08-07 | 14 | [恢复后姿态差 → base_height 超高惩罚 + settle 水平静止](2026-08-07/14_fix_pose_overshoot_drift.md) |
| 2026-08-07 | 13 | [交付 — 从零训练 100% 恢复率 + 稳定站立](2026-08-07/13_delivery_100pct_recovery_stable.md) |
| 2026-08-07 | 12 | [站不稳 → settle 平衡奖励 + 1024 envs 从零训练](2026-08-07/12_settle_balance_1024envs_fromscratch.md) |
| 2026-08-07 | 11 | [交付 — 无辅助单机器人恢复视频](2026-08-07/11_delivery_single_robot_recovery.md) |
| 2026-08-07 | 10 | [Phase 2 — 纯无辅助 + rise_vel, 恢复成形 (30%)](2026-08-07/10_phase2_forcefree_risevel.md) |
| 2026-08-07 | 09 | [rise_vel 奖励 — 搭"推腿自举"脚手架](2026-08-07/09_rise_vel_scaffold.md) |
| 2026-08-07 | 08 | [Plan B — 里程碑阶梯 + 慢衰减重启](2026-08-07/08_milestone_ladder_planB.md) |
| 2026-08-07 | 07 | [力衰减单位 bug — step_counter 批次步 vs force_end env 步](2026-08-07/07_force_decay_unit_bug.md) |
| 2026-08-07 | 06 | [3000 iter 评估 — 力引导阶段门控死锁 → 解耦 h_cmd2](2026-08-07/06_force_hcmd2_deadlock_fix.md) |
| 2026-08-07 | 05 | [首次训练评估 — 恢复未成形 (局部最优 0.26m)](2026-08-07/05_first_train_eval.md) |
| 2026-08-07 | 04 | [CPO 独立 conf/cpo 目录 + 约束 value loss 数值修复](2026-08-07/04_cpo_conf_dir_and_value_scale.md) |
| 2026-08-07 | 03 | [FTSR 环境实现 + 冒烟验证 + 终止门控修复](2026-08-07/03_fsr_env_and_verify.md) |
| 2026-08-07 | 02 | [移植 CPO 算法 (惩罚函数法)](2026-08-07/02_cpo_port.md) |
| 2026-08-07 | 01 | [方案改向 FTSR (力引导+分阶段+CPO)](2026-08-07/01_fsr_design.md) |
| 2026-08-06 | 02 | ~~环境+配置 (贴地后空翻, 已废弃)~~ |
| 2026-08-06 | 01 | ~~P1 可行性 (贴地后空翻, 已废弃)~~ |
