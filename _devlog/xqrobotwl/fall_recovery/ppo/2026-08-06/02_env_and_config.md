# [02] 跌倒恢复 RL 环境 + 配置 + 启动脚本

**日期**: 2026-08-06
**来源**: P1 可行性验证 (01) 完成, 进入 RL 开发
**关联**: [01_p1_feasibility](01_p1_feasibility.md)

---

## 问题描述

P1 验证了起身机制（贴地后空翻, FF 即可完成），需固化为 RL 环境：FSM 前馈驱动
确定性翻身，RL 专注起身后的两轮足倒立摆平衡。对齐后空翻"确定性翻转"决定和
单腿"start_in_balance 课程"突破。

## 根因分析

- 起身是弹道大动作，策略会被奖励诱导偏离 FF 把翻身搞坏（后空翻反复教训）→ 翻身期
  (FSM 0-3) 策略屏蔽为 0，纯 FF 驱动。
- FF 站直会立即失衡 → 状态 3 目标改为"落地蹲姿" (z~0.40, 膝 0.35)，交棒 RL。
- 初始 reset (DR manager) 不更新 `_fsm_state` → 需覆写 `reset()` 同步 FSM。
- 翻身期关节合法大摆动 → 关节塌陷终止只作用于平衡态 (-1)。

## 解决方案

新建 `src/unilab/envs/locomotion/xqrobotwl/fall_recovery.py`（注册 `XqRobotWLFallRecoveryFlat`）:

- **FSM**: -1平衡(RL) ← 0收腿(0.30s) → 1甩腿(0.25s) → 2飞行(0.30s) → 3落地蹲姿(0.30s)
- **前馈**: P1 参数复现 (tuck 0.85/0.80, kick -0.60/0.30, catch 0.45, land 0.20/0.35), 轮制动
- **奖励** (相位门控): getup_progress(翻转监控) / upright_landing / wheel_ground /
  stand_balance / stand_height / **recover_complete**(轮着地+直立+高度, 连续 0.5s 大奖) /
  posture_stand / ff_tracking / action_rate / leg_mirror / alive
- **恢复点判据**: 轮着地 + up>0.9 + z>0.35
- **课程**: `start_in_balance` (True=reset 恢复后平衡位 z=0.40, 先学保持)
- **obs**: 297/324 + 36 (fsm, timer, wheel_contact×2) = 333/360
- 配套: 配置 `conf/ppo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml` (100Hz, action_scale 0.6,
  init_noise_std 0.05)、启动 `shell/xqrobotwl/launch_ppo_fall_recovery.sh`、
  热启动 `scripts/xqrobotwl/warmstart_from_walk_fall_recovery.py`、
  冒烟 `scripts/xqrobotwl/smoke_fall_recovery.py`

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/envs/locomotion/xqrobotwl/fall_recovery.py` | 新建: FSM 前馈 + 相位门控奖励 + start_in_balance 课程 + DR Provider |
| `src/unilab/envs/locomotion/xqrobotwl/__init__.py` | + fall_recovery 导出 |
| `conf/ppo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml` | 新建: 训练配置 (100Hz, start_in_balance: true) |
| `shell/xqrobotwl/launch_ppo_fall_recovery.sh` | 新建: quick/full/warmstart/resume 启动 |
| `scripts/xqrobotwl/smoke_fall_recovery.py` | 新建: 双模式冒烟 (FF 翻身 + 平衡保持) |
| `scripts/xqrobotwl/warmstart_from_walk_fall_recovery.py` | 新建: walk 模型热启动 (333=297+36) |

## 验证方法

1. `uv run mjpython scripts/xqrobotwl/smoke_fall_recovery.py` — 双模式冒烟
2. `uv run mjpython scripts/training/train_rsl_rl.py task=xqrobotwl_fall_recovery_flat/mujoco training.task_name=XqRobotWLFallRecoveryFlat algo.num_envs=16 algo.max_iterations=30` — 训练管线
3. `uv run ruff format` + `uv run python -m mypy` (与 single_leg/backflip 相同预存模式)

## 评估结果

- **Smoke (start_in_balance=True)**: 恢复后平衡位 z=0.40 零动作稳定, recover_completed=True ✅
- **Smoke (start_in_balance=False)**: FF 翻身 → 轮上直立 (t=120: up=0.998, z=0.36) 交棒 RL;
  零动作下 0.1s 后倒 (倒立摆需 RL 主动平衡, 属预期) ✅
- **训练 (30 iter, 16 env)**:
  - start_in_balance=true: reward 37.3, ep_len 82 (平衡位晃动, RL 需学主动纠正)
  - start_in_balance=false: reward 14.9, ep_len 162 (> 翻转 115 步, 大部分活过翻转) 
  - 无 NaN, actor 333 / critic 360, checkpoint + play_video 正常

## 后续计划

- [ ] Phase 1 训练 (start_in_balance=true): 学恢复后平衡保持 → 选模型确定性评估
- [ ] Phase 2 训练 (start_in_balance=false): 学完整恢复, 或热启动 walk 模型
- [ ] 平衡位晃动问题: balance 课程/奖励调优 (roll_rate 阻尼等)
- [ ] DR 鲁棒性 + 俯卧恢复 + 落腿/恢复后行走

## 关联日志

- [01_p1_feasibility](01_p1_feasibility.md)
- 后空翻确定性翻转: `_devlog/xqrobotwl/backflip/ppo/2026-08-05/08_*`
- 单腿 start_in_balance: `_devlog/xqrobotwl/single_leg/ppo/2026-08-05/03_*`
