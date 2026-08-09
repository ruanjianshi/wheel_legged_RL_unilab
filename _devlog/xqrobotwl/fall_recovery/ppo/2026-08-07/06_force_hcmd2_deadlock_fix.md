# [06] 3000 iter 评估 — 恢复仍 0%, 力引导阶段门控死锁 → 解耦 h_cmd2

**日期**: 2026-08-07
**来源**: 长训 3000 iter (warmstart walk + CPO, 128 envs) 后确定性评估
**关联**: [05_first_train_eval](05_first_train_eval.md) / [01_fsr_design](01_fsr_design.md)

---

## 问题描述

3000 iter 训练 (run `2026-08-07_00-54-47_mujoco`) 完成后评估 (model_2999.pt):
- 恢复率 (base_z 达 0.55m) = **0%** (仍 0, 同 05)
- 平均最大高度 = **0.28m** (比 05 的 0.26 略升, 仍在半撑局部最优 0.28-0.31m)
- 平均最大躯干直立度 = **0.51** (有进步, 从 ~0 提升 — 学会半撑)
- 双轮着地率 98.8% — 半撑是"轮着地 + 躯干半直立"的稳定姿态
- 训练 reward 0→11.4 持续增长 (upright 0.15→0.47), 但 **reward/base_height ≈ 0.00** 全程,
  **reward/recover_complete = 0.00** 全程 (里程碑从没触发)

## 根因分析

**阶段门控死锁 (chicken-and-egg)**:
1. `update_state` 中批次全局阶段 `self._stage = (mean(base_z > h_cmd1=0.32) > 2/3)`。
   1024 env 中大部分 agent 卡在地面, n1 长期 < 2/3 → 阶段永远停在 ru。
2. `_apply_force_assist` 的力目标 `h_cmd = [h_cmd1, h_cmd2][stage]` 锁死 **h_cmd1=0.32**:
   F = Fmax·(1-e^{-μ(h_cmd-h)}), 在 0.28m 处只剩 ~33N, 0.31m 处 ~9N —
   **力恰好跨不过的起身坎 (0.28→0.45) 处衰减殆尽**。
3. `_reward_base_height` 在 ru 阶段目标 0.32, `clip((h-0.25)/(0.32-0.25),0,1)` 到 0.32 就饱和
   (满分 4.0) → 0.32 以上无高度梯度, 只剩 100 分 recover_complete 悬崖 (0.45) — 彩票, 从不触发。
4. 结果: 机器人半撑到 0.28-0.31m (轮着地, 躯干半直立) 是稳定局部最优, 混 alive 奖励。

**验证**: 同一条"腿伸直"动作, 修复前 base_z 卡 0.28-0.31m; 力目标改为 h_cmd2 后冲到 0.38m。

## 解决方案

把"上升"的主信号 (力辅助 + 高度奖励) 与批次阶段课程**解耦**, 始终目标完整站立 h_cmd2=0.55:
1. **力辅助 h_cmd 恒 = h_cmd2**: 力从贴地 (145N) 到 0.45m (72N) 全程强力, 站立 (0.55) 归零。
   阶段门控不再锁死力目标。
2. **高度命令/奖励 h_cmd 恒 = h_cmd2**: base_height 奖励 `clip((h-0.25)/(0.55-0.25),0,1)`
   单调到站立, 消除 0.32 饱和墙。
3. 批次阶段保留但仅切换次要权重集 (scales_ru/rs) — 恢复扩散后自然切 rs, 不再是阻塞点。
4. **评估禁用辅助力** (`force_assist_enabled=false`): 之前评估带着辅助力跑 (评估环境
   step_counter=0 → t_coeff=1 → 满力), 测的是"有辅助还失败"; 现测真实无辅助部署恢复。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/envs/locomotion/xqrobotwl/fall_recovery.py:388` | `_apply_force_assist` 力目标 `[h_cmd1,h_cmd2][stage]` → 恒 `h_cmd2` |
| `src/unilab/envs/locomotion/xqrobotwl/fall_recovery.py:437-445` | `update_state` h_cmd 恒 `h_cmd2`; 阶段仅选 scales 集 |
| `scripts/xqrobotwl/eval_fall_recovery.py:60` | 评估 override `force_assist_enabled=false` |

## 验证方法

冒烟测试: 复位 + 恒定腿伸直动作 + 小幅随机, 300 步:
- 修复前: base_z 卡 0.28-0.31m
- 修复后: base_z 达 0.38m, step 120 阶段切 rs (3/4 agent 过 0.32)
- constraint_costs ~0.8 (力更强, 符合预期)

## 后续计划

- [ ] 重训 5000 iter (force_end_iters=3000 → 后 2000 iter 无辅助力, 训练力独立)
- [ ] 每 1000 iter 检查增长趋势 (用户规则)
- [ ] 恢复成形 → 无辅助评估 recovery_rate → 渲染视频 → 停止

## 关联日志

- [05_first_train_eval](05_first_train_eval.md) — 首次评估 (0.26m 局部最优)
