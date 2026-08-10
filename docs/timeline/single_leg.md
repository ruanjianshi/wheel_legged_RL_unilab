# 时间线 · 单腿平衡 (single_leg_flat / single_leg_move / single_leg_unicycle)

> 一句话: 两轮足机器人单轮支撑的倒立摆平衡 (横躺姿态 + 直配重 + 扭矩源轮) — 已突破 (unicycle RL model_3000/4000 完整 8s 单轮平衡), 2026-08-10 移植后待本地重训。
> 来源: _devlog/xqrobotwl/single_leg/ppo/  (实时更新, 每完成一个阶段追加一行)

| 日期 | 阶段 | 做了什么 | 影响/效果 | 问题与解决 |
|---|---|---|---|---|
| 2026-08-05 | P1 可行性 | single_leg_balance_feasibility.py 验证三点: 静态平衡位/折腿过渡/横滚控制权 `[[01_p1_feasibility]]` | 平衡位可达 (CoM 距支撑轮 0.026m), 折腿过渡 CoM 偏移 0.102m < 轮半宽 0.19m, 横滚控制权充足 | 必须用收膝 (knee bend) 而非髋外展 (hip_roll 甩离 CoM) |
| 2026-08-05 | env 设计 + roll 符号 | 新建 single_leg.py; roll_ref 符号修复 (+28°), 奖励改线性梯度, FF 只脚本化自由腿, 折腿 1.0s `[[02_env_design_and_roll_sign_fix]]` | balance_upright 从 0 → run3 iter 196 达 2.9 | roll_ref 方向反 (往支撑轮外侧倒); dot 奖励梯度太弱 |
| 2026-08-05 | start_in_balance | 折腿过渡太难 (kp=60 撑不住, base 塌缩 z 0.30) → reset 直接置平衡位先学保持 `[[03_transition_too_hard_start_in_balance]]` | 训练能对齐平衡倾角 (align 0.97→0.38 但自由轮离地仅 8%) | balance_complete 高度阈值过渡中物理不可达 |
| 2026-08-05 | roll_rate 符号 + pitch 瓶颈 | roll_rate 符号写反 (奖励抖动); 里程碑放宽 dot 0.88; run11 平台化分析 `[[04_roll_rate_sign_and_hold_improvements]] [[05_pitch_is_the_real_bottleneck]]` | roll_rate 真惩罚后抖动减少; 崩溃是 pitch 发散 (-5→+48°) 非 roll | `-rate × scale -0.3 = 奖励抖动`; warmstart pitch 能力不迁移单腿 |
| 2026-08-06 | 配重 + 经典控制 | 用户姿态落地: 30° 侧压 + 自由腿 L_hip_roll 给 RL 当配重; 经典 PD 物理验证 `[[06_counterweight_design]] [[07_classic_control_physical_findings]]` | 30° 侧压 CoM 精确落轮 (|dy|=0.000); 默认执行器 kp=60 静态 2s (物理可行), 但经典 PID ~1.3s 崩 | 机身 90° 立起不可行; 腿屈-轮动耦合振荡, 主动控制反而破坏平衡 |
| 2026-08-06 | RL 卡点 + LQR 假象 | 独立 env single_leg_move 从零训练; LQR 参考验证 `[[08_single_leg_move_env_first_train]] [[09_single_leg_move_balance_stuck]] [[10_lqr_not_breakthrough_honest]]` | 30° 配重单轮物理稳定上限 ~1.35s; RL 从零学不会 (0.36s 倒); LQR 实际 ~1.29s | LQR "10s" 是宽松判据假象 (up[2]<0.35); 配重主动控制搅乱 roll |
| 2026-08-06 | 严格评估 | 三次"突破" (LQR/用户姿态/kp 调优) 全部评估判据 bug 假象; 最终物理判定 `[[11_user_pose_breakthrough]] [[12_strict_eval_all_fail]] [[13_final_physical_verdict]]` | 严格判据下 PD/LQR/用户姿态/kp 调优全部 ~1.3s; CoM 高度结构限制 0.55m (落轮需 >0.7m) → 判定物理不可行 | 判据 bug 连续三次 (force 误判/up[2]<0.35/z<0.18); 执行器弱(过冲) + CoM 高 |
| 2026-08-06 | 用户姿态验证 + kp bug | 用户 v3/v4 姿态精确验证; 发现位置执行器 kp 须同时设 gainprm[0] 与 biasprm[1]=−kp (之前 kp>60 结论全无效); 膝塌陷=边际稳定 (载荷负刚度≈kp60) `[[14_user_pose_both_knees_verify]] [[15_user_pose_v3_breakthrough]] [[16_v3_fake_breakthrough_corrected]] [[17_hip_reference_straight_leg_final]] [[18_v4_user_pose_full_eval]] [[19_knee_collapse_mechanism_and_kp_bug]]` | v3 "30s" 判据漏洞误报 (漏查左小腿触地, 实际 0.03s 轮离地); kp≥200 实测稳住膝; 但膝修好后单轮仍不行 (0.3-0.6s 小腿触地) | 折叠支撑腿是载荷下的正反馈坍塌机构; kp>60 结论因执行器参数 bug 作废 |
| 2026-08-06 | 结构搜索 | 结构搜索找到 CoM 0.5cm 对齐 + 近直支撑膝 + 伸直配重 ±5cm 权限 + 轮=扭矩源 `[[20_structure_search_unicycle_viable]]` | 膝塌陷修复, 静态 0.43→1.0s, 控制 0.86s (视频对比原 0.18s); 轮控 pitch 有效, 伸直配重控 roll 有效 | 折叠配重权限不足会饱和震荡; pitch 须用 base 长轴测 (up∥轮轴不可用) |
| 2026-08-06 | unicycle RL env | 新建 single_leg_unicycle.py + XML 加 basexvector/left_wheel_world_pos 传感器; obs 36D; 钉支撑腿 + RL 控 [配重 roll, 轮 pitch] `[[21_unicycle_rl_env]]` | smoke + 39 iter 冒烟通过 (reward 4.19, episode 0.58s); 奖励迭代 v1 无惩罚 2.75s / v2 线速度 1.2s (有害) / v3 位置 drift | 罚线速度阻止了平衡必需的轮动 (倒立摆需轮滚动) → 只罚净位置漂移 |
| 2026-08-06 | RL 突破 | 训练至 model_3000/4000, 修复 eval bug (active mask 冻结已 done env) `[[22_unicycle_rl_breakthrough]]` | **model_1000 4.1s (PD 0.86s 的 4.8x) → model_3000/4000: 20 env 全跑满 8s (左轮贴地 100%, pitch/roll 对齐 0.995/1.0)**; 视频 `video/single_leg/2026-08-06_RL单轮平衡_8s.mp4`; 部署用 model_4000 | model_4999 回退 1.68s (PPO 后期不稳定) → 取中途甜点位; --keyboard 命令 3D 探测破坏 5D obs 帧 → require_keyboard_command_obs=false |
| 2026-08-10 | 移植 | 从 fall_recovery 分支移植 single_leg{,_move,_unicycle} env/conf/脚本/11 条日志 `[[repo/08_port_fall_recovery_branch]]` | 5 新 env import 通过, 训练脚本 `shell/xqrobotwl/single_leg/{train,eval}_ppo_single_leg{,_move,_unicycle}.sh` | 现待本地 quick 训练测试 (3 变体) |

## 当前状态
- 突破 checkpoint: **single_leg_unicycle model_4000** (部署用, run `2026-08-06`, 从零训练, 1024 envs) — 20 env 全跑满 8s 单轮平衡, pitch 对齐 0.995, roll 对齐 0.999
- 关键指标: PD 0.86s → RL 8s (**9.3x**); 关键奖励设计 = 位置 drift 惩罚 (非线速度惩罚)
- 完成路径: 30° 配重 0.2s 塌落 → 结构搜索物理可行 → RL 8s 平衡, 闭环走通
- 当前仓库状态: ✅ 代码已移植 (2026-08-10), 🚧 本地训练测试中 (thesis/experts/07)

## 下一步
- [ ] 本地训练测试通过 (quick 模式, 3 变体)
- [ ] 全量训练 + 评估验收 (平衡保持时长、独木桥通过率)
- [ ] 第二阶段: vx 前进/后退移动
