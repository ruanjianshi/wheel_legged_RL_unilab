# 02 — 后空翻环境开发 (P2): backflip.py + 配置 + 训练管线

## 日期

2026-08-04

## 来源

P1 验证后空翻物理可行(360° 落地 3.8°)。进入环境开发, 把 P1 参数固化为 FSM 前馈, 供 PPO 强化。

## 问题描述

把 P1 开环验证的后空翻转化为可训练的 RL 环境。核心挑战:
1. FSM 前馈需在 policy 空间表达 (apply_action 有 flip/scale/default 多层符号)
2. flip_progress 追踪需对随机偏航鲁棒
3. 终止条件不能让翻转中途被倾角杀
4. 一次性动作需 reset 硬重置

## 根因分析

- **apply_action 符号**: P1 raw MuJoCo 目标 → policy 空间 ff 需反算 `ff=(T-default)/(flip*scale)`, 验证 4 个状态全部精确复现
- **flip_progress 测量**: 世界系 XZ 投影在偏航下失真(4 env 符号不一致) → 改机身系俯仰角速度积分 `-∫gyro_y`, 对偏航鲁棒
- **终止过严**: 翻转后摔地(与 P1 相同, z 到 0.15)但恢复相位救不回 → 放宽终止, 仅持续摔倒(0.4s)才终止
- **恢复相位**: ff=0 瞬跳默认姿态打断旋转 → 改 当前→默认 渐变 blend

## 解决方案

新建 `src/unilab/envs/locomotion/xqrobotwl/backflip.py` (继承新 PPO-SRL jump_srl):
- 7 状态 FSM: -1站立→0蹲→1蹬→2飞(收腿)→3展(轮匹配)→4缓冲→5恢复
- FSM 前馈 = P1 验证参数反算的 policy 空间值 (`_FLIP_FF`)
- flip_progress = -∫gyro_y dt (后翻=正)
- 相位门控奖励: flip_progress(飞行)/launch_thrust/upright_landing/landing_soft/wheel_ground_matching/flip_complete
- 宽松终止: 飞行态(1/2/3)不按倾角, 落地/恢复(4/5)持续 0.4s 高倾角才终止
- reset 硬重置 FSM/flip_progress (`_reset_done_envs` override)
- 预热: 前 100 iter 纯站立, 100-200 斜坡放开 trigger

配套: `conf/ppo/task/xqrobotwl_backflip_flat/mujoco.yaml`, `shell/xqrobotwl/train_ppo_backflip.sh`, `eval_ppo_backflip.sh`, 注册到 `__init__.py`。

## 修改文件

| 文件 | 内容 |
|------|------|
| `src/unilab/envs/locomotion/xqrobotwl/backflip.py` | 新建。后空翻 env: FSM 前馈 + flip_progress + 相位门控奖励 + 宽松终止 + reset 硬重置 |
| `conf/ppo/task/xqrobotwl_backflip_flat/mujoco.yaml` | 新建。奖励表 + 预热 + 命令配置 |
| `shell/xqrobotwl/train_ppo_backflip.sh` | 新建。训练脚本 |
| `shell/xqrobotwl/eval_ppo_backflip.sh` | 新建。评估脚本 |
| `src/unilab/envs/locomotion/xqrobotwl/__init__.py` | 注册 backflip 模块 |

## 验证方法

1. **纯 ff 复现**: env 单 env, warmup 关, 纯 ff → 翻转达 +5.30 rad (304°), 与 P1 中期轨迹一致
2. **1024 env 稳定性**: 构造 0.1s, 3ms/步, 无 NaN, 观测维度 324/351 正确
3. **训练管线冒烟**: `train_rsl_rl.py` 2 iter (64 env) → checkpoint/ONNX/回放视频全部生成

## 评估结果

- ✅ 环境功能完整: FSM/奖励/终止/重置/观测全部工作
- ⚠️ 纯 ff 达 304° 后落地阶段机器人趴下(与 P1 的完整 360°+落地 3.8° 有差距)。原因: env ctrl_dt=0.01(100Hz) 与 P1 脚本 200Hz 控制率不同, 以及落地恢复相位细节差异。**这正是 PPO 要学的**: flip_progress 奖励推完最后 60°, upright_landing/flip_complete 教学会落地。

## 后续计划

- P3: 三阶段课程训练 (Stage A 起跳+旋转 → B +落地 → C +恢复站立)
- 训练在远程 GPU 服务器跑, 监控 flip_progress/成功率
- 若训练不收敛, 回头调 ff (或匹配 P1 的 200Hz 控制率)

## 关联日志

- [01_physics_feasibility](2026-08-04/01_physics_feasibility.md) — P1 开环验证, 本环境参数来源
- [12_sign_bugs_entropy_fsm_fixes](../jump/ppo/2026-07-24/12_sign_bugs_entropy_fsm_fixes.md) — 跳跃 FSM 时序修复经验
