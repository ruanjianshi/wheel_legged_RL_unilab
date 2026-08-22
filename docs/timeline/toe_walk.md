# 时间线 · 点足/抬腿行走 (xqrobotwl_toe_walk_flat)

> 一句话: 两轮足机器人点足/抬腿行走 — **双模式 (站立⇄点足抬腿) 训练 v6.1 首次达成"双腿交替抬腿"(L12/R10.5) + 切换稳定 PASS; 站立微动超标 (0.375m/s) 待精修**。

| 日期 | 阶段 | 做了什么 | 影响/效果 | 问题与解决 |
|---|---|---|---|---|
| 2026-07-22 | v1 训练 | 新建 toe_walk 配置 + 参数调优 (cycle_time 0.5→0.7s, ref_scale 0.12→0.18, action_scale 0.3→0.5, thigh×0.8) `[[01_toe_walk_success]]` | iter 1409: ep_len 1165 (97%), action_std 0.15 收敛, swing_lift 2.39 (轮确实离地), wheel_balance 0.99, ref_tracking 3.3 — 点足行走验证成功 | — |
| 2026-07-22 | 符号修复 | 参考轨迹符号修复: L_knee 摆动相伸膝→屈膝, R_thigh 摆动相后仰→前倾 `[[02_reference_sign_fix]]` | 抬腿幅度增大, 轮可离地 | L_knee `DEFAULT−lift×scale×5` 方向反; R_thigh 符号反 |
| 2026-07-22 | 相位门控 | 新增相位门控奖励模式 (无参考轨迹, 策略自由探索步态): phase_swing_lift 5.0 / phase_knee_lift 3.0 / phase_stance_contact 2.0 `[[03_phase_gated]]` | 不指定关节角度, 策略自由发现最优姿态 | 参考轨迹自由度低 |
| 2026-07-23 | bug 修复 | 5 个符号/逻辑 bug + 2 终止阈值修正 (swing_mask 从 sin(phase) 生成; 3 个奖励符号反; leg_mirror 减→加; calf_extreme 0.85→2.0; contact 0.1N→5.0N) `[[04_sign_bugs_and_termination_fixes]]` | 弯膝不再被终止误杀, 轮子不再被奖着地 | 致命链: 弯膝→calf>0.85→终止→不弯膝→phase_knee_lift=0; 轮着地→+25 奖励→主动着地 |
| 2026-07-24 ~ 07-25 | 论文驱动重设计 | knee_lift 解锁 (移除 l_air/r_air 鸡生蛋门控), 课程平滑 (curriculum_steps 4000, swing_contact_penalty 25→5), 读 BoltLocomotion/tron1/TitaRL 重写奖励 (删 4 冲突项, action_scale 0.5→0.25, init_noise 0.7→0.4) `[[05_knee_lift_redesign_final]]` | iter 9999: reward 2.8→**167** (60x), ep_len 25→**922** (37x), phase_swing_lift 0.13→0.48 (轮 48% 摆地离地), action_std 0.15; 备份 `backup/XqRobotWLToeWalkFlat/toe_walk_ppo_v1/` | phase_knee_lift 要求轮先离地 (鸡生蛋); 课程 ramp 太陡 (2.5→25 崩); 15+ 奖励冲突 |
| 2026-07-28 | 高度提升 + 柔化 | base_height -60→-150, action_scale 0.25→0.18, joint_action_rate -0.03→-0.08 `[[07_height_up_smooth_action]]` | 蹲姿 h 0.38→0.48-0.52m (峰值 0.49 已验证可达), 抬腿不再碰机身, 动作从 bang-bang 猛抬转多步平缓 | 蹲太低 (0.38m) 抬腿碰机身; action_scale 0.25 每步 14° 猛抬不平滑 |
| 2026-08-10 | 全量重训 | 8 任务批量重训 (1024 envs), toe_walk 跑满 10000 iter | mean_reward **144.81 (全任务最高)**, ep_len 1000 (满), action_std 0.24, SPS 9955; checkpoint `logs/rsl_rl_ppo/XqRobotWLToeWalkFlat/2026-08-10_01-26-06_mujoco/model_9999.pt` | eval 冒烟通过, 无回归 |
| 2026-08-14 | 全量重训 | toe_walk 最后重训 (配置与 08-10 相同但共享基座代码有漂移) | mean_reward 134.7 / best 184.1 / ep_len 858.6 — 略低于 08-10 | 无日志、未回查 |
| 2026-08-18 | 交替/对称性验证 | 新增 `tools/xqrobotwl/verify_toe_walk_symmetry.py`, 对 08-14/07-28/v1 各 checkpoint + 多指令确定性验证 `[[08_verify_alternation_symmetry]]` | **最新 m9999: 左腿 0 次/右腿 12 次抬腿, 8s 转圈 197°, vx=0.3 时 0 次抬腿纯轮滚 — 已退化**; v1 是历史最接近 (双腿都抬, 相位差 0.41-0.50) 但仍幅度不对称+蹲低+位移 | 全部 checkpoint 均不合格: 奖励无左右对称约束, 后期迭代塌缩到"单侧抬腿转圈"或"纯滚动"局部解 |
| 2026-08-18 | 双模式升级 (老板新需求) | 老板确认新框架「站立⇄点足抬腿 + 按键切换 + 指令追踪」; 新增 `toe_walk_mode.py` 环境 (commands 5D mode 通道, 奖励 mode 门控, **lift_symmetry 对称约束**, 模式课程 2000/5000), conf + shell + play_interactive H 键; 参考/框架: `docs/references/2026-08-18_mode_switching_multi_skill.md` + `docs/plans/2026-08-18_toe_walk_mode_switch_framework.md` `[[09_mode_switch_env_interactive]]` | 冒烟通过 (obs 315/342, mode 门控正确); 训练启动 GPU1 1024envs×10k (SPS~55k, iter~4863 reward 转正 20.7) | 方案: 单策略+mode 命令通道 (Uni-Match/so100 思路), 课程分练后混合防冲突 |
| 2026-08-18 | 双模式训练 v1→v6.1 | 六轮迭代: v1 (站立达标/抬腿无) → v2 (推力翻倍/抬腿必死) → v3 (终止放宽/抬腿涌现但单侧) → v4 (罚重→弯膝刷分) → v5 (窗级机制/稀疏反馈死锁) → v6.1 (**符号 bug 修复+双层考核**) `[[10..15]]` | **v6.1 突破**: Stage3 混合下 swing_lift 保持 0.92-1.03 (历史版本全崩); 最终评估 **交替 PASS (L12/R10.5) + 切换 PASS** | 根因链: 终止误杀→单侧经济性→弯膝刷分→稀疏反馈→符号 bug; 每次用训练 reward 分解定位 |
| 2026-08-18 | v6.1 最终评估 | 确定性序列评估 + 渲染演示视频 (video/toe_walk/) `[[16_v61_final_alt_pass]]` | **腿交替抬腿首次达成**; 站立微动 0.375 (>0.2 阈值, 待精修); 追踪中 (fwd rmse 0.24) | 站立漂移: Stage3 混合污染 + stand_still 权重弱 |

## 当前状态
- **双模式最佳模型**: `logs/rsl_rl_ppo/XqRobotWLToeWalkMode/2026-08-18_15-16-18_mujoco/model_9999.pt` (v6.1)
- **核心目标达成**: 双腿交替抬腿 (L12/R10.5, 交替 PASS) + 模式切换稳定 (0.5s 不倒 PASS)
- **待精修**: 站立微动 0.375 m/s (阈值 0.2), gyro 1.07 (阈值 1.0), 高度 0.461 (略低)
- 交互: H 键切换站立/点足抬腿, ↑/↓←/→/A/D 指令追踪
- devlog: `_devlog/xqrobotwl/toe_walk/ppo/2026-08-18/` 08-16 全记录

## 下一步
- [ ] 视频渲染完成 → 交付老板验收 (video/toe_walk/)
- [ ] (老板确认后) 站立微动精修: stand_still 权重加强 / Stage3 站立占比提升 / 站立期轮速罚
- [ ] 对称量化补测 (膝弯比) + 达标后备份 toe_walk_mode_v6.1

## 2026-08-19~20 单模式回归 (老板指示: 单模式点足抬腿, conf/ppo/task/xqrobotwl_toe_walk_flat)

| 日期 | 版本 | 做了什么 | 结果 | 结论 |
|------|------|----------|------|------|
| 08-19 | v10/v10.1/v10.2 | 单模式移植"窗级交替+对称+追踪门控" | 窗级罚(500/课程化)在无平衡预热下压死探索 → 崩盘/动作爆炸 | 窗级重罚对从零学习致命 (第 N 次实证) |
| 08-19 | v1 重训 | 回滚 v1 精确配方 (backup 07-25) | 能走能抬但单侧偏 (L13/R3) | v1 从未真正达标"一左一右" |
| 08-19 | v11/v12 | 平衡预热1200 + 窗罚课程 / 去窗罚+lift_sym | 预热后抬腿仍崩/极慢 | 预热反而加深站姿先验 |
| 08-20 | **v13/v13b** | **参考轨迹模式** (use_reference=true, 正弦ref强制左右交替) | **交替数学解决**: ref_tracking 42.7/20 完美, ep_len 1000, reward 403; 但 ref_scale 0.15 下轮不离地 (swing≈0) | 唯一"一左一右"从机制上成立的路线, 待幅度参数调整 |

## 当前状态 (2026-08-20)
- **最佳机制**: 参考轨迹模式 (v13b, run `2026-08-19_18-17-37` 之前的 `2026-08-20_13-56-32`) — 左右交替由正弦参考强制, 稳定满分存活
- **待调**: ref_scale 0.15→0.22~0.25 (让轮真正离地) — 待老板拍板 A/B
- 双模式 (toe_walk_mode) v6.1 备选 (交替 PASS 但站立冲突)
- devlog: `_devlog/xqrobotwl/toe_walk/ppo/2026-08-19/` + `2026-08-20/` 22-28 全记录

## 当前状态 (2026-08-20 交付)
- ✅ **v14 已交付**: `backup/XqRobotWLToeWalkFlat/toe_walk_v14/` (开箱即跑) + 视频 `video/toe_walk/2026-08-20_v14_点足演示_录屏.mp4`
- 参考轨迹模式 (ref_scale 0.20): 一左一右交替 (ref 42.7/20) + 轮摆动窗近离地 (4-6N) + 幅度老板认可 (0.8 rad)
- 高 0.485 / 完全离地待老板按需追加调优

## 下一步
- [ ] (按需) 轮完全离地 (thigh 前摆加强) / 高度 0.485→0.52
- [ ] 论文第5章素材: 参考轨迹点足 (正弦步态)
