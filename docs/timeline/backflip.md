# 时间线 · 后空翻 (xqrobotwl_backflip_flat)

> 一句话: 两轮足机器人按键触发的 360° 后空翻 (FSM 前馈 + 确定性翻转) — 原开发已交付 (model_1000 32/32 翻转+直立), 2026-08-10 移植后待本地重训。
> 来源: _devlog/xqrobotwl/backflip/ppo/  (实时更新, 每完成一个阶段追加一行)

| 日期 | 阶段 | 做了什么 | 影响/效果 | 问题与解决 |
|---|---|---|---|---|
| 2026-08-04 | P1 可行性 | 开环脚本 backflip_feasibility.py: 6 相位 FSM (蹲→蹬→飞→展→缓冲→恢复) 验证 360° 后空翻 `[[01_physics_feasibility]]` | 翻转进度 -6.28 rad (正好 360°), 落地倾角 3.8°, 腾空 0.41s, 最高 0.72m, 膝峰值扭矩 94 N·m; 视频 `video/backflip/2026-08-04_01_开环可行_360°落地3.8°_lean06_ch-025.mp4` | 符号约定与直觉相反: 负 flip=后翻, 膝猛伸+髋后仰=后翻, 轮加速=前翻 |
| 2026-08-04 | 环境开发 | 新建 backflip.py (7 态 FSM 前馈 + flip_progress 测量 + 相位门控奖励 + 宽松终止 + reset 硬重置) + conf + 训练管线 `[[02_env_development]]` | env 功能完整, 纯 ff 复现达 304° (100Hz 下 vs P1 200Hz 有差距, 正是 PPO 要补的) | flip_progress 世界系 XZ 投影在偏航下失真 → 改机身系俯仰角速度积分; 恢复相位 ff 瞬跳打断旋转 → 渐变 blend |
| 2026-08-04 | 训练 bug 修复 | 修复预热斜坡累乘 bug (trigger 被乘没) + 站立态 flip_progress 重置; init_noise_std 0.3→0.05 防探索 std 暴涨 `[[03_warmup_trigger_bugfix]] [[04_exploration_std_explosion_fix]]` | 翻转正常触发 (纯 ff 86%), 探索 std 受控 0.05→0.06 | 预热 `commands[:,4] *= alpha` 累乘 2400 步 ≈0 → FSM 永不触发; std 涨到 3.9 淹没 ff |
| 2026-08-04 | 站立平衡 | 新增 stand_balance 奖励 (up+2*orient_pen) + clip_actions 3→100 + base_height_target 0.65; walk 模型 warmstart (actor 前 297 维继承) `[[06_standing_balance_fix]] [[07_warmstart_from_walk]]` | 站立平衡解决 (倾角 1-7°, stand_balance 3-4), 但翻转未完成 (最大 1.8 rad) | 两轮足是倒立摆天生不稳, clip_actions=3 压掉了平衡所需大动作; 平衡 vs 翻转对抗 |
| 2026-08-04 | 任务范围 + 防取巧 | 用户放宽: 只训练后空翻落地不倒 (flip_trigger_prob 1.0, max_tilt 75); 后加 stand_height 高度奖励 + flip_complete 高度条件 + 接触式终止 `[[08_relaxed_backflip_only]] [[09_anti_cheat_posture_fix]] [[10_flip_complete_height_requirement]] [[11_not_fallen_not_standstraight]] [[12_max_episode_6s]] [[13_contact_termination]] [[14_staged_relax_termination]]` | 防蹲低/贴地取巧; max_episode 4→10s; 严格终止 (z<0.20) 后翻不过 → 落地 0.3s 恢复窗口 | 策略翻完蹲低 (z=0.41) 拿"不倒"分; 小腿贴地 (z=0.1) 仍算 flip_complete; 严格终止把翻转杀卡死 |
| 2026-08-05 | 测量根因修复 | flip_complete 改锁存 (转完站起来即触发); flip_progress 改世界系 Y 角速度积分; 物理事件判据锁存 (up 翻过身) + 补轮子急加速 + 去掉 jump_srl 双重前馈 `[[01_flip_complete_latch]] [[02_flip_progress_measurement_fix]] [[04_flip_complete_event_latch]]` | flip_complete 从恒 0 → 稳定非 0 (0.45-0.75), ~650 iter 学会完成后空翻+落地直立 | 机身系 gyro 积分在带横滚/偏航时失真 (物理 360° 只测 299°); 旋转积分阈值不可达 |
| 2026-08-05 | ff_tracking + 重平衡 | 新增 ff_tracking 参考跟踪 (文献 OPT-Mimic, 防甩腿/麻花); flip_progress 封顶; A/B 实验重平衡 (ff_tracking 10 / flip_progress 30) `[[03_ff_tracking_reference]] [[05_rebalance_ff_tracking_vs_flip_progress]]` | A/B 对照: ff_tracking=10 过强 → ep_len 卡 50, flip_complete 极弱 → 恢复基线权重 (负结果) | 结果奖励不约束动作质量 → 策略靠甩腿伪造旋转 (BAV 缺陷); ff_tracking 过强罚死探索 |
| 2026-08-05 | 模型选择教训 | V2 续训发散诊断: 确定性评估全部 checkpoint `[[06_overfit_action_explosion_v2]]` | 基线 model_999 为甜点位 (flip_complete 96, up_z -1.0, max\|a\| 0.39); v2 5999 iter 动作爆炸 (2.84) 不翻转 | 训练日志指标是 on-policy 随机采样不可靠; 长训收敛到"站姿 hack" (不翻转靠站姿奖励) |
| 2026-08-05 | 200Hz 控制率 | 用户实测"一翻就倒" → 根因 env 100Hz 太粗 (爆发蹬地 0.07s 猛伸) → ctrl_dt 0.005 + P1 复现 ff 重写 `[[07_control_rate_200hz_fix]]` | 200Hz 纯 ff: 16/16 翻转+站立 (up_z 0.89); 100Hz 对照打滚 (up=0.17) | 100Hz 下翻转不足 → 地面打滚 → 落地倒挂 |
| 2026-08-05 | 确定性翻转 + 交付 | 200Hz 下 4 次奖励配置全部失败 → 用户决定翻转期策略屏蔽, 纯 ff 驱动 (stand_mask) `[[08_deterministic_flip]]` | **交付 model_1000: 32/32 翻转完成+站起, 32/32 最终直立 (up_z 1.0), max\|a\| 0.55**; 视频 `video/backflip/2026-08-05_确定性翻转成功_model1000.mp4` | 奖励结构把策略从"近零策略=ff 翻转"的好初始解拉走; model_1999 又发散 (过训练), 取中途 checkpoint |
| 2026-08-10 | 移植 | 从 fall_recovery 分支移植 env/conf/脚本/14 条日志到主仓库 `[[repo/08_port_fall_recovery_branch]]` | 5 新 env + CPO import 通过, backflip 端到端跑 1 迭代成功; 训练脚本 `shell/xqrobotwl/backflip/{train,eval}_ppo_backflip.sh` | 现待本地 quick 训练测试 + 全量重训 |

## 当前状态
- 原交付模型: model_1000 (1000 iter) — 32/32 翻转完成+站起, 32/32 最终直立, 动作干净 (max|a|=0.55)
- 关键教训: 一次性动作任务, 训练日志不可靠, 过训练必发散 → 用中途 checkpoint 做甜点位
- 当前仓库状态: ✅ 代码已移植 (2026-08-10), 🚧 本地训练测试中 (thesis/experts/06)

## 下一步
- [ ] 本地训练测试通过 (quick 模式, 支持 warmstart/resume)
- [ ] 全量训练 + 评估验收 (翻转完成率)
- [ ] 若需抗扰动, 可尝试 ff_gain 调度 (翻转期策略轻微修正)
