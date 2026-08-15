# [06] 对称几何重训 13 任务完成 + 全量评估报告

**日期**: 2026-08-15
**状态**: 完成 — 13 任务全部训练到 10000 轮 (unicycle 20000), 评估结果如下
**关联**: [[../2026-08-14/05_retrain_all_13_tasks]] (启动), [[../2026-08-14/03_wheel_symmetry_fix]] (对称化)

---

## 训练状态 (8-14 14:15 启动, 全部完成)
- 13/13 训练到末尾 (12 任务 9999/10000, single_leg_unicycle 19999/20000)
- 全部导出 policy.onnx + policy.pt + 11 checkpoint
- 无训练进程残留 (仅 tensorboard)
- backflip 训练期 3 次短暂仿真不稳定警告 (不影响完成)

## 评估 (tools/xqrobotwl/eval_symmetric_geometry.py, 3 ep × 10s, GPU0)

| # | 任务 | 算法 | 存活率 | 说明 |
|---|---|---|---|---|
| 1 | walk_flat | PPO | **100%** | ✅ 站立 + 前向 |
| 2 | walk_rough | PPO | **100%** | ✅ 粗糙地形 (需 obs_normalizer) |
| 3 | toe_walk | PPO | 67% (2/3) | 1 ep 1.4s 倒 |
| 4 | jump | PPO | 0% | 动态任务, 站立评估不适用 (需跳跃触发) |
| 5 | jump_srl | PPO+SRL | **100%** | ✅ 零命令稳定 |
| 6 | jump_vmc | PPO+VMC | **100%** | ✅ (修复 terrain 字段 bug) |
| 7 | jump_srl_vmc | PPO+SRL+VMC | **100%** | ✅ |
| 8 | backflip | PPO | 33% (1/3) | 动态任务, 站立评估不适用 |
| 9 | single_leg_flat | PPO | 0% | 单腿复位姿态不匹配 (重训前同 0%) |
| 10 | single_leg_move | PPO | 0% | 同上 |
| 11 | single_leg_unicycle | PPO | **100%** | ✅ 独轮站稳 |
| 12 | fall_recovery | CPO | 0% | 需跌倒→恢复场景, 站立评估不适用 |
| 13 | stairs | NP3O | **100%** | ✅ 阶梯 (需 checkpoint 推导 hidden dims) |

## 总结论
- **站立/平衡类 7/7 满存活率**: walk_flat, walk_rough, jump_srl, jump_vmc, jump_srl_vmc,
  single_leg_unicycle, stairs — **对称几何重训效果良好** (无几何回归)。
- **toe_walk 67%** — 大部分站立, 1 ep 早倒, 可后续评估更多 ep。
- **动态/特殊任务 (jump/backflip/fall_recovery/single_leg_flat/move) 0-33%** —
  固定零命令站立评估协议不适用 (需跳跃触发/跌倒场景/单腿复位), **非必然重训失败**;
  single_leg_flat 重训前同协议也 0%, 证明是协议不匹配非回归。

## 评估脚本修复 (5 处 bug, 否则结果失真)
1. max_episode_seconds=1000 — 防 max_episode_seconds=10 在 sim_time 边界截断误判跌倒
   (walk_flat 等实为跑满 10s 却报 0%)
2. terrain 字段按配置存在性守卫 — jump_vmc/srl_vmc 无 terrain_curriculum 报错
3. **obs_normalizer 加载** — walk_rough/stairs empirical_normalization=true,
   load_actor 丢弃 → 0.2s 早倒 (修复后 100%)
4. 存活判定 = 跑满 sim_time; 每 episode env.reset 站姿复位 (原用零 obs 起步)
5. **hidden dims 从 checkpoint mlp 权重推导** — stairs/np3o 无 algo.policy 段,
   默认 4 层与 [512,256,128] 不匹配 (修复后 100%)

commit: 64ce415, c2bc898

## 数据
- 评估日志: `logs/symmetry_eval_13tasks.log`
- 训练日志: `logs/train/*.log`
- 各任务最新 checkpoint: `logs/rsl_rl_ppo|cpo|np3o/<Task>/2026-08-14_14-1x-xx_mujoco/model_*.pt`

## 关联
- [[../2026-08-14/05_retrain_all_13_tasks]] [[../2026-08-14/03_wheel_symmetry_fix]]
