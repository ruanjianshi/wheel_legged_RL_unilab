# xqrobotwl/single_leg/ppo 开发索引

XqRobotWL 单腿平衡(单轮支撑)步态 — PPO + FSM前馈过渡 + RL横滚平衡。

## 2026-08-17

| # | 主题 | 结论 | 链接 |
|---|------|------|------|
| 23 | 站立按键切换横躺独轮车式 | 🚧 4态FSM+H键锁存+分阶段左支撑/右配重轨迹；v3失败对照完成，v4训练中 | [→](2026-08-17/23_stand_to_unicycle_keyboard_transition.md) |

## 2026-08-05

| # | 主题 | 结论 | 链接 |
|---|------|------|------|
| 01 | P1 物理可行性验证 | ✅ 平衡位可达(CoM距支撑轮 0.026m) + 收膝折腿过渡稳定 + 横滚控制权充足 | [→](2026-08-05/01_p1_feasibility.md) |
| 02 | env 设计 + roll_ref 符号修复 | ✅ roll_ref 反号已修(+28°), 奖励改线性梯度, FF只脚本化自由腿, 折腿1.0s | [→](2026-08-05/02_env_design_and_roll_sign_fix.md) |
| 03 | 过渡太难 → start_in_balance | ✅ 折腿过渡 base 塌缩(kp60撑不住)→ 改为 reset 直接置平衡位先学保持 | [→](2026-08-05/03_transition_too_hard_start_in_balance.md) |
| 04 | roll_rate 符号修复 + 里程碑放宽 | ✅ 倒立摆保持需阻尼震荡, roll_rate 返回+误差(scale负)才是惩罚; dot>0.88 | [→](2026-08-05/04_roll_rate_sign_and_hold_improvements.md) |
| 05 | pitch 是真正瓶颈 | ⚠️ run11 平台化: 崩溃是 pitch 发散(-5→+48°)非 roll; warmstart pitch 不迁移单腿 | [→](2026-08-05/05_pitch_is_the_real_bottleneck.md) |

## 2026-08-06

| # | 主题 | 结论 | 链接 |
|---|------|------|------|
| 06 | 配重设计: 30°侧压+自由腿当配重 | ✅ 用户姿态反馈落地: 自由腿 L_hip_roll 给RL当配重, 30°CoM精确落轮, 90°立起不可行 | [→](2026-08-06/06_counterweight_design.md) |
| 07 | 经典控制物理验证: 单腿可行但PID难稳 | ⚠️ kp=60柔性腿静态2s(物理可行), 30°侧压偏难, 腿屈-轮动耦合振荡→经典PID难稳定, 最优=微扰动保持(转RL) | [→](2026-08-06/07_classic_control_physical_findings.md) |
| 08 | 独立 env 首训: 从零学会存活但两轮作弊 | ⚠️ quick 500iter 从零存活800步, 但策略放下自由腿两轮支撑作弊→wheel_off 权重4→15 + balance_complete 大奖防作弊 | [→](2026-08-06/08_single_leg_move_env_first_train.md) |
| 09 | RL 训练卡点: 单轮物理上限 1.35s | ❌ 30°配重单轮物理稳定上限~1.35s(腿屈-轮动耦合), RL从零学不会(两轮作弊/0.36s倒), 试遍权重/终止/探索/侧压→需LQR参考或结构改进 | [→](2026-08-06/09_single_leg_move_balance_stuck.md) |
| 10 | LQR 未突破: "10s"是评估假象 | ❌ LQR 实际~1.29s(与PD一致), "10s"因宽松终止+两轮支撑假象, 配重主动控制反而搅乱roll→物理上限~1.3s交叉确认 | [→](2026-08-06/10_lqr_not_breakthrough_honest.md) |
| 11 | 用户姿态误判为突破 | ❌ 用户姿态(横躺90°左腿支撑+kp=200)"10s"是 z<0.18 判据宽松假象, 实际~0.59s崩 (L_hip_roll过冲3.54) | [→](2026-08-06/11_user_pose_breakthrough.md) |
| 12 | 严格评估: 所有方案~1.3s极限 | ❌ 三次"突破"均评估判据bug假象; 严格判据下 PD/LQR/用户姿态/kp调优全部~1.3s, 物理极限确认 (执行器弱/过冲 + CoM高 + 单轮本质) | [→](2026-08-06/12_strict_eval_all_fail.md) |
| 13 | 最终物理判定: 单轮平衡不可行 | ❌ 严格判据PD仅0.2s, 执行器增强(kp/kv)全失败, CoM结构限制0.55m(落轮需>0.7m, 低则落不了轮)→物理不可行, 需结构升级 | [→](2026-08-06/13_final_physical_verdict.md) |
| 14 | 用户新姿态(双腿外展+双膝弯)验证 | ❌ 双膝弯极限把两轮收回髋部→竖直趴地/横躺CoM 0.635反而高; 但单弯支撑膝CoM 0.55→0.27真降重心, 严格单轮0.2s→0.9s; 低重心自由腿无法停放(触地或CoM偏)→仍<2s | [→](2026-08-06/14_user_pose_both_knees_verify.md) |
| 15 | 用户姿态v3(左腿支撑+右腿平衡) — ⚠️被16更正 | 声称"严格单轮30s+" — 判据漏洞误报(漏查左小腿触地), 用户看视频识破 | [→](2026-08-06/15_user_pose_v3_breakthrough.md) |
| 16 | v3真实结果: 支撑轮0.03s离地 → 不可行 | ❌ 补全判据(任何非支撑轮触地=崩): 轮0.03s离地+左小腿0.22s触地, 与膝角/刚度/轮自由度无关=结构几何失稳; 综合全部尝试确认单轮平衡在xqrobotwl不可行 | [→](2026-08-06/16_v3_fake_breakthrough_corrected.md) |
| 17 | 用户参考系洞察 + 直腿验证 → 最终物理判定 | ❌ 支撑腿为基准/髋转动机身在原理上正确但逐一验证失败; 直腿(髋在轮上)CoM偏3.3cm→L_pitch偏转0.2rad撬轮; 轮kv增强无效; dt敏感性证明轮离地~0.05s是真实物理; ~40配置全失败 | [→](2026-08-06/17_hip_reference_straight_leg_final.md) |
| 18 | 用户v4精确姿态+控制架构全评估 (Task1a/1b/2) | ❌ 观测修正: pitch须用base长轴(up∥轮轴不变), 这是v3假象更深根因; ①左轮机械可承重但v4无法维持单轮(kp=60膝0.53→0.90折叠→小腿0.21s触地/kp=200轮0.012s离地) ②左膝承载下跟不动目标(误差1.37rad) ③闭环架构+dt/kv/gap敏感性全失败 | [→](2026-08-06/18_v4_user_pose_full_eval.md) |
| 19 | 膝塌陷机制 + kp执行器bug更正 | ⚠️ **重大更正**: 位置执行器kp须同时设 gainprm[0] 与 biasprm[1]=−kp, 之前只改gainprm→目标×kp/60, devlog 16/17/18 的 kp>60 结论全无效; 膝塌陷=边际稳定(载荷负刚度−55~−66≈kp60, 净刚度0→屈曲), 非强度; **kp≥200 实测稳住膝**; 但膝修好后单轮平衡仍不行(0.3-0.6s小腿触地, 右腿配重反效果, CoM x偏移结构性) | [→](2026-08-06/19_knee_collapse_mechanism_and_kp_bug.md) |
| 20 | 结构搜索: 单轮平衡物理可行 | ⚠️→✅(结构) 结构搜索找到 CoM 0.5cm 对齐 + 近直支撑膝 + **伸直配重±5cm权限**; 轮=扭矩源控pitch有效, 伸直配重控roll有效 (0.6s两轴稳); 膝塌陷已修; 静态0.43→1.0s, 控制0.86s (视频对比原0.18s); **剩余=配重roll耦合pitch的3D控制**, 简单PD不够, LQR/RL有机会 | [→](2026-08-06/20_structure_search_unicycle_viable.md) |
| 21 | 改进结构 RL env (single_leg_unicycle) | 🚧→✅(env) 新 env: XML+basexvector/pitch观测+left_wheel_world_pos, 腿kp300+轮扭矩源(运行时), obs 36D, 钉支撑腿+RL控[配重roll,轮pitch]; smoke test + 39iter冒烟通过 (reward 4.19, episode 0.58s); 奖励迭代 v1无惩罚2.75s/v2线速度1.2s(有害)/v3位置drift | [→](2026-08-06/21_unicycle_rl_env.md) |
| 22 | RL 单轮平衡突破: model_3000 完整 8s | ✅✅ **里程碑**: model_1000 4.1s (PD 0.86s 的4.8倍) → model_3000 **20 env 全跑满 8s** (左轮贴地100%, pitch/roll对齐0.995/1.0); 视频 8s 满窗口; eval bug修复(active mask); 结构搜索+位置drift奖励闭环走通, v4 0.2s塌落→RL 8s平衡 (9.3倍 PD) | [→](2026-08-06/22_unicycle_rl_breakthrough.md) |
