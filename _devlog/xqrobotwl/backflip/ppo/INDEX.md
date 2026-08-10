# xqrobotwl/backflip/ppo 开发索引

XqRobotWL 后空翻步态 — PPO + 相位门控奖励 + SLIP FSM 前馈引导。

## 2026-08-05

| 序号 | 标题 | 文件 |
|------|------|------|
| 08 | 确定性翻转: 翻转期策略屏蔽, 纯 ff 驱动 (用户决定) | [→](2026-08-05/08_deterministic_flip.md) |
| 07 | "一翻就倒"根因: 控制率 100Hz 太粗, 改 200Hz (ctrl_dt=0.005) | [→](2026-08-05/07_control_rate_200hz_fix.md) |
| 06 | V2 续训发散 (动作爆炸) — 模型选择教训 | [→](2026-08-05/06_overfit_action_explosion_v2.md) |
| 05 | 奖励重平衡: ff_tracking 10 / flip_progress 30 (防刷旋转) | [→](2026-08-05/05_rebalance_ff_tracking_vs_flip_progress.md) |
| 04 | flip_complete 判据改物理事件锁存 (根因: 旋转积分阈值不可达) | [→](2026-08-05/04_flip_complete_event_latch.md) |
| 03 | ff_tracking 参考跟踪 (文献 OPT-Mimic, 防甩腿/麻花) | [→](2026-08-05/03_ff_tracking_reference.md) |
| 02 | flip_progress 测量修复: 世界系 Y 角速度 (根因) | [→](2026-08-05/02_flip_progress_measurement_fix.md) |
| 01 | flip_complete 改锁存 (旋转完成→站起来给奖励) | [→](2026-08-05/01_flip_complete_latch.md) |

## 2026-08-04

| 序号 | 标题 | 文件 |
|------|------|------|
| 14 | 分阶段方案C: 放宽终止让翻转学成 | [→](2026-08-04/14_staged_relax_termination.md) |
| 13 | 接触式终止: 机身/小腿碰地立即终止 | [→](2026-08-04/13_contact_termination.md) |
| 12 | max_episode_seconds 调整到 10s (1000 步) | [→](2026-08-04/12_max_episode_6s.md) |
| 11 | 修正需求: 后空翻后"不倒"即可, 不要求"站直" | [→](2026-08-04/11_not_fallen_not_standstraight.md) |
| 10 | flip_complete 加高度条件 (防小腿贴地取巧) | [→](2026-08-04/10_flip_complete_height_requirement.md) |
| 09 | 防取巧: 站立高度 + 倒地判定修正 | [→](2026-08-04/09_anti_cheat_posture_fix.md) |
| 08 | 放宽要求: 只训练后空翻 (落地不倒即可) | [→](2026-08-04/08_relaxed_backflip_only.md) |
| 07 | walk 热启动 + 平衡/翻转对抗修复 | [→](2026-08-04/07_warmstart_from_walk.md) |
| 06 | 站立平衡修复 (两轮足需主动平衡) | [→](2026-08-04/06_standing_balance_fix.md) |
| 05 | 键盘控制后空翻 (play 模式预热绕过 + H 键触发) | [→](2026-08-04/05_play_mode_keyboard_control.md) |
| 04 | 修复策略探索 std 暴涨 (init_noise_std 0.3→0.05) | [→](2026-08-04/04_exploration_std_explosion_fix.md) |
| 03 | 修复预热斜坡累乘 bug + 站立态 flip_progress 重置 | [→](2026-08-04/03_warmup_trigger_bugfix.md) |
| 02 | 环境开发: backflip.py + FSM前馈 + 相位门控奖励 + 训练管线 | [→](2026-08-04/02_env_development.md) |
| 01 | 开环脚本物理可行性验证 (360° 后空翻 + 落地 3.8°) | [→](2026-08-04/01_physics_feasibility.md) |
