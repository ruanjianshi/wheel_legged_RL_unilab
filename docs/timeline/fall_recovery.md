# 时间线 · 跌倒恢复 (xqrobotwl_fall_recovery_flat)

> 一句话: 两轮足机器人 4 种倒地姿态 (仰/俯/左/右) 无脚本自恢复站立的 CPO 步态 (FTSR 力引导) — 已交付 (v7 model_4000, 4 姿态 100% 恢复, 姿态接近自然, yaw≈walk)。
> 来源: _devlog/xqrobotwl/fall_recovery/ppo/  (实时更新, 每完成一个阶段追加一行)

| 日期 | 阶段 | 做了什么 | 影响/效果 | 问题与解决 |
|---|---|---|---|---|
| 2026-08-06 | P1 可行性 (废弃) | 开环贴地后空翻起身验证 (tuck→kick→catch→lift FSM) `[[01_p1_feasibility]]` | 恢复点 (轮着地+直立>0.9+z>0.35) 全 start_z 可靠到达, 结束倾角 4.7°; 视频待训 | 该方案后被 FTSR 取代 (见 08-07); 轮加速把身体掀飞 → 轮制动最佳 |
| 2026-08-06 | 环境开发 (废弃) | fall_recovery.py (FSM 前馈翻身 + RL 平衡) + start_in_balance 课程 `[[02_env_and_config]]` | smoke: 恢复后平衡位 z=0.40 零动作稳定; 30 iter 训练 reward 37.3 | 该方案后被 FTSR 取代 |
| 2026-08-07 | 方案改向 FTSR | 用户指定论文 FTSR: 删除脚本化 FF, 纯学习恢复 + 力引导 F (高度相关) + 高度分阶段 (ru→rs) + 4 姿态复位 + CPO 约束 `[[01_fsr_design]]` | 力引导 F≈38N/T≈5Nm; 终止门控修复后 episode 1→415 步 | 初始终止 (tilt>55°) 杀掉倒地起始态 → has_recovered 门控 |
| 2026-08-07 | CPO 移植 | 移植惩罚函数法 CPO (cpo.py, 约束代价由 env 提供) + 独立 conf/cpo 目录 + 约束 value loss 数值修复 `[[02_cpo_port]] [[04_cpo_conf_dir_and_value_scale]]` | constraint_value loss 爆炸 665M→稳定 ~250, reward 10.93, ep_len 653 | MPS float64 崩溃 → float32; cost return 尺度 ~100 → 目标×(1-γ), coef 0.1 |
| 2026-08-07 | 首次训练评估 | warmstart walk + CPO 300/3000 iter 后确定性评估 `[[05_first_train_eval]] [[06_force_hcmd2_deadlock_fix]]` | 恢复率 0%, 平均最大高度 0.26m (局部最优"蟹式半撑"); 力目标解耦 h_cmd2 后冒烟 0.28→0.38m | 局部最优: 半撑 0.26m 躲过贴地终止 + 高度奖励 exp 饱和无梯度; 阶段门控死锁 (力目标锁 h_cmd1) |
| 2026-08-07 | 力衰减 bug + Plan B | 力衰减单位 bug 修复 (step_counter 批次步 vs env 步); Plan B 里程碑阶梯 (rise 30 分) + 慢衰减 3000→5000 iter `[[07_force_decay_unit_bug]] [[08_milestone_ladder_planB]]` | 力 iter 9000 精确归零; 冒烟 rise 100%/recover 100%; 但力 <40% 后策略仍塌缩 | 力永不衰减 (×num_envs 错位) → 训练 32% 恢复=满力抬起来的; 半撑→站立梯度死区 |
| 2026-08-07 | rise_vel 脚手架 + Phase 2 | 新增 rise_vel 密集奖励 (双轮着地门控的向上速度) → 放弃力, 纯无辅助训练 `[[09_rise_vel_scaffold]] [[10_phase2_forcefree_risevel]]` | 恢复技能首次成形: model_14000 恢复率 25% → model_15000 **30%** (无辅助) | 策略"力锁定"缺推腿自举技能; 端点式 height 奖励逐帧梯度太稀 |
| 2026-08-07 | 交付 v0 + settle | model_15000 交付 (30% 恢复率, 视频); 用户反馈站不稳 → 新增 settle 稳定站立奖励 + rise_vel 门控 + 1024 envs 从零训练 `[[11_delivery_single_robot_recovery]] [[12_settle_balance_1024envs_fromscratch]] [[13_delivery_100pct_recovery_stable]]` | **model_5000: 恢复率 100%, 最长连续站立 1.74s** (v0 只有 0.4s); 视频 `2026-08-07_从零训练_恢复并稳定站立_model5000.mp4` | v0 会"推上去"不会"站稳" (rise_vel 顶部继续奖推 → 冲过头); 从 model_15000 (30%) → model_5000 (100%) |
| 2026-08-07 ~ 08-08 | v1/v2 姿态修复 | 超高惩罚 band 0.08 (v1 崩) → 0.12 + h_cmd2 0.55→0.52 + 独立 stand_still; 4 姿态分别评估 `[[14_fix_pose_overshoot_drift]] [[15_v2_redo_overshoot_band_standstill]] [[16_delivery_4pose_posefix]]` | **v2 model_4000: 4 姿态恢复率 80-95%, 站立高度 0.50-0.54m (对齐 0.52), 水平漂移 0.52-0.77m (改善 5-8x)** | v1 三个改动全跟恢复打架 (band 0.08 杀起身冲高, settle 水平乘法乘死, dof_pos -0.5 太重) → 恢复率崩到 55% |
| 2026-08-08 | v3 站立时间 | 站立太短根因: \|gyro\| 6-9 rad/s 剧烈摇摆 → settle 加角速度项; 硬门失败改指数衰减 `[[17_v3_gyro_stillness]] [[18_delivery_v3b_standing_time]]` | **v3b model_4000: 4 姿态 100% 恢复, 最长连续站立 3.37-6.50s (v2 的 10-15 倍)** | v3a 硬门 `1-clip(|gyro|/0.5)` 无梯度 (0.27s 更差) → v3b `exp(-|gyro|/2)` 全程连续梯度 |
| 2026-08-08 | v4/v5 姿态约束 | 腿一前一后+转圈根因: DEFAULT_LEG_ANGLES 膝符号错 → standing_angles 目标 + no_yaw; v4 过强致保持崩 → v5 温和 `[[19_v4_natural_stance_no_yaw]] [[20_v5_gentle_posture]]` | v4: 恢复率 100% 但保持崩 (0.2-0.4s); v5: 保持 3.37s, 腿角对称, yaw 232° | 起身蓄力必须深蹲屈膝, dof_pos -2.0 强罚压坏起身 → 降到 -0.5 |
| 2026-08-09 | v6/v7 转圈根治 | 转圈根因: 轮速差 474 rad/s 狂转 → wheel_symmetry + wheel_speed; 残留小差速+蹲姿 → 收紧 /5 + 门控站姿奖励 stand_pose `[[21_v6_wheel_symmetry]] [[22_v7_tight_wheel_symmetry_stand_pose]] [[23_delivery_v7_4pose]]` | **v7 model_4000: 4 姿态 100% 恢复, yaw 63°≈walk 56° (固有物理特性), 保持 0.63-0.81s**; 各版本进展: v3b yaw 260° → v5 232° → v6 299°(轮速差治) → v7 63° | walk 模型站立也转 yaw 56° → 轻微转圈是两轮倒立摆固有物理特性, 无法归零 |
| 2026-08-10 | 移植 | 从 fall_recovery 分支移植 env/CPO/conf/脚本/23 条日志到主仓库 `[[repo/08_port_fall_recovery_branch]]` | 新任务端到端跑 1 迭代成功; 训练脚本 `shell/xqrobotwl/fall_recovery/train_ppo_fall_recovery.sh` (走 train_cpo.py) | 现待本地 quick 训练测试 + 全量重训 |

## 当前状态
- 交付 checkpoint: **v7 model_4000** (run `2026-08-09_15-14-54_mujoco`, 从零训练, 1024 envs, CPO)
- 关键指标 (无辅助确定性, 每姿态 20 eps): 4 姿态恢复率 **100%**, 站立高度 ~0.59m, 水平漂移 0.57-0.89m, 最长连续站立 0.63-0.81s, yaw 63° (≈walk 水平)
- 达标情况: ✅ 已交付 — 4 姿态恢复 + 姿态接近自然 + yaw≈walk; 剩余: 漂移略超 0.5m、保持短于 v3b (姿态约束与保持的权衡)
- 当前仓库状态: ✅ 代码已移植 (2026-08-10), 🚧 本地训练测试中 (thesis/experts/08)

## 下一步
- [ ] 本地训练测试通过 (quick 模式, CPO)
- [ ] 全量训练 + 评估验收 (恢复成功率、恢复时间、恢复后姿态稳定)
- [ ] (可选) 更长保持: 权衡姿态约束 vs 保持时间; 师生蒸馏 / 行走阶段 (rw)
