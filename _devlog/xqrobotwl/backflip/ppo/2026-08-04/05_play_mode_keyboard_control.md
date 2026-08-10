# 05 — 键盘控制后空翻 (play 模式预热绕过 + H 键触发)

## 日期

2026-08-04

## 来源

用户反馈训练版后空翻视频: 翻转完成但落地站不稳。需要键盘控制验证指令触发效果。

## 问题描述

`play_interactive.py --keyboard` 对 backflip 任务有 2 个问题:
1. **预热未绕过**: play 模式只绕过 jump 的 `_jump_curriculum_end`, backflip 的 `_flip_warmup_env_steps` 没绕过 → 前 24s flip_trigger 被门控为 0, H 键无效
2. **绕过数学错**: backflip 预热在 [warmup, 2×warmup] 斜坡, 需跳到 `2*warmup+1` 而非 `warmup+1`

## 解决方案

`scripts/play/play_interactive.py` `_build_keyboard_commander`:
- 加 backflip 预热绕过: `env._total_env_steps = 2*env._flip_warmup_env_steps + 1`

H 键已有 (jump_trigger → commands[:,4]=1 脉冲 0.8s), backflip 复用 (flip_trigger 同位置)。

**额外修复**:
- `interactive.require_keyboard_command_obs` 是 struct 外字段, 需 `+` 前缀追加 (backflip 命令 vx/vy=0 原地翻转, 跳过速度命令 obs 检查)
- macOS 需 `mjpython` 运行原生窗口 (eval 脚本加平台检测)
- `.venv/bin/mjpython` exec 路径硬编码错 (缺 CodeBase 段), 已修复

## 修改文件

| 文件 | 内容 |
|------|------|
| `scripts/play/play_interactive.py` | 加 backflip 预热绕过 (跳到 2×warmup) |
| `shell/xqrobotwl/eval_ppo_backflip.sh` | `+interactive.require_keyboard_command_obs=false` + mjpython 平台检测 |
| `.venv/bin/mjpython` | 修复 exec 路径 (CodeBase 段缺失) |

## 验证方法

headless 测试: bypass 后 progress=1.0, 80 帧 trigger 脉冲 → FSM 进入飞行态, flip_progress 达 4.14 rad。

键盘命令: `bash shell/xqrobotwl/eval_ppo_backflip.sh --keyboard`, 按 H 触发。

## 后续计划

- 用户键盘测试后评估翻转/落地质量
- 已知问题: 训练策略过度旋转(8.7 rad) + 落地站不稳 → 调训练或站立平衡

## 关联日志

- [04_exploration_std_explosion_fix](2026-08-04/04_exploration_std_explosion_fix.md) — 训练收敛
