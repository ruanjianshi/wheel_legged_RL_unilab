# [29] 论文图重出: 0.8 平滑训练曲线 + 参考风格高度曲线 (2x2 + FSM 相位)

**日期**: 2026-08-09
**来源**: 用户要求 "出图, 0.8平滑, 跳跃验证高度对比放一张图"; 后追加 "高度变化曲线, 参考 paper_fig_trajectory.png 的方式绘图"
**关联**: [[28_pure_ppo_reward_hacking_v10_v12]]

---

## 问题描述

1. 训练曲线 (尤其纯 PPO) 毛刺多、波动大, 直接出图难以阅读趋势。
2. 用户要的是**高度随时间变化的曲线** (蹲→蹬→腾空→落地), 且绘图方式参考
   `jump_management/results/paper_fig_trajectory.png` 的 **2x2 每算法一子图 + SLIP-FSM 相位色带** 风格,
   而非单图叠四条曲线。

## 解决方案

| 文件 | 改动 |
|------|------|
| `scripts/make_paper_figures.py` | 新增 `ema_smooth()` (TB 风格 EMA, `--smooth` 默认 0.8); `fig_training` 四曲线统一平滑; `fig_validation` 简化为 **Air Fraction + Survival Rate 两栏柱状** (高度曲线移出) |
| `scripts/verify_jump_trajectory.py` (重写) | 对四算法**当前最优 checkpoint** 各跑一次跳跃 (settle 50 / trigger ON 160 / tail 170), 记录 `t / base_z / hip_pitch / knee / phase / standing_z / terminated`, 存 `jump_traj_{srl,ppo,vmc,srlvmc}.npz` (与 record_jump_trajectory 同 schema) |
| `scripts/plot_jump_trajectory.py` | `fig_height` 输出改名 **`paper_fig_trajectory`** (2x2 + FSM 相位色带, 参考风格); 终止的算法 (纯 PPO) 曲线末尾标 ×; `fig_joints` 沿用新数据 |

## 输出

- `jump_management/results/paper_fig_training.{png,pdf}` — 训练曲线, 0.8 EMA 平滑
- `jump_management/results/paper_fig_trajectory.{png,pdf}` — **高度曲线**, 2x2 + 相位色带
- `jump_management/results/paper_fig_validation.{png,pdf}` — 腾空/存活柱状
- `jump_management/results/jump_traj_{srl,ppo,vmc,srlvmc}.npz` — 轨迹数据

## 验证方法

- 出图成功 (exit 0), 目视确认: 参考风格一致 (SRL/VMC+SRL 有相位色带, PPO/PPO+VMC 无 FSM 无色带); ruff format + check 通过。
- 单跳峰值与 compare_jump 平均跳高一致: SRL 0.505≈0.540, VMC+SRL 0.344≈0.354,
  PPO+VMC 0.210≈0.179, PPO 0.255≈0.278; **PPO 在 step 321 摔倒终止**, 曲线截断标 ×,
  无参考不可控直观可见。
- 四个 run 均完整训满 10000 iter (RSL-RL checkpoint 0..9999, model_9999 即最后);
  纯 PPO 最优 checkpoint 取早期 model_1000 (后期发散病态)。

## 后续计划

- [x] 出图 (0.8 平滑 + 参考风格高度曲线 + 腾空/存活)
- [ ] 论文图微调 (配色/标注) 按需
