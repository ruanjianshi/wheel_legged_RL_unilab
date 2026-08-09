# [30] 论文图 nature/dataviz 规范重出: 新增图4.1 训练指标 + 全图统一风格

**日期**: 2026-08-09
**来源**: 用户要求 "利用 nature skills 指导重新出论文图", 并按论文 (武汉科技大学研究生小论文) 图 4.1 caption 结构出图; 范围经确认 = 图4.1 训练指标 (四算法对比版) + 统一风格化现有三张对比图
**关联**: [[29_paper_figs_smooth_single_view]]

---

## 问题描述

1. 论文图 4.1 (Wheeled-SRL 训练指标曲线) 要求 4 面板: (a) 平均奖励 (b) 回合长度
   (c) 奖励分量分解 (d) 动作标准差与训练 FPS。现有 `paper_fig_training` 只有 2x2
   (mean_reward / ep_length / jump_height / landing_soft), 缺 (d) 的 action std 与 FPS。
2. 论文 caption 的数值 (平均奖励 ≈47, 跳跃高度奖励 ≈11.7, 动作标准差 ≈2.6) 与真实数据
   全部不符: SRL run 实际 mean_reward ≈99、jump_height reward ≈2.2 (论文的 11.7 实为
   vertical_thrust ≈11.9)、action std ≈0.8。即 caption 是占位值。
3. 现有图用 `tight_layout` 有 gridspec 警告, 风格不统一 (字号/网格/图例各异)。

## 根因分析

- 论文 text 部分指标为写作期占位 (与 4.2/4.3 节 "[待补充]" 一致), 出图必须用真实数据,
  caption 数值需按实际数据修订。
- 奖励分量 tag 名随算法不同 (SRL/VMC+SRL 有 vertical_thrust/height_progress, PPO 系有
  launch_rise/wheel_air_time), 无法跨算法画完全一致的分解; 取四 run 都有的
  jump_height / landing_soft 作为 (c)(d) 面板。

## 解决方案

| 文件 | 改动 |
|------|------|
| `scripts/make_paper_figures.py` | 新增 `_style_ax()` (衰退网格 alpha 0.2 + 细脊线 0.6) 与 `_plot_curves()` (四算法同色并排); 新增 **`fig_training_metrics()` → `paper_fig_training_metrics`** (图 4.1, 2x3, constrained layout): (a) Mean Reward (b) Episode Length (c) Jump Height Reward (d) Landing Soft Reward (e) Action Std (f) Training FPS; reward 类 EMA 0.8, **std/FPS 用原始值** (各自单轴, 不画双 y 轴); 共享图例 `loc="outside upper center"` frameon=False 黑色文字 (缓解 #E69F00 对比度 WARN) |
| 同 | `fig_training` 改 constrained layout + 统一字号 (title 9 / label 7.5 / tick 7) + 共享图例; `fig_validation` 柱状同风格 (值标签 7.5、网格 0.2) |
| `scripts/plot_jump_trajectory.py` | `fig_height`/`fig_joints` 改 constrained layout + `_style_ax`; 移除 fig_joints 未用的 `phase` 变量 (ruff F841) |
| 配色 | Okabe-Ito 四色经 dataviz `validate_palette.js --mode light` 验证: **4/4 PASS** (CVD ΔE≥11.0, normal ΔE≥24.2), #E69F00 对比度 2.19 WARN → 黑色图例文字缓解 |

## 输出

- `jump_management/results/paper_fig_training.{png,pdf}` — 训练对比 2x2 (0.8 EMA)
- `jump_management/results/paper_fig_training_metrics.{png,pdf}` — **图 4.1** 2x3 (a-f)
- `jump_management/results/paper_fig_validation.{png,pdf}` — 腾空/存活柱状
- `jump_management/results/paper_fig_trajectory.{png,pdf}` — 高度曲线 2x2 + FSM 相位
- `jump_management/results/paper_fig_jump_joints.{png,pdf}` — 关节角 2x2

## 验证方法

- 全部 exit 0; `ruff format` + `ruff check` 通过。
- 目视确认 (Read 渲染): paper_fig_training / training_metrics / trajectory / jump_joints
  风格统一 (衰退网格、细脊线、顶部共享图例、面板字母标号)。
- `paper_fig_validation` 在本会话与上一会话均遇 Read 工具 "[Unsupported Image]" 渲染问题
  (RGB/JPEG/缩放转换均失败, 与之前一致); PIL 验证文件有效 (1804x784 RGBA, 非空 bbox),
  柱状逻辑为简单 bar+值标签, 未阻塞。
- 图 4.1 (e)(f) 面板确认 std/FPS 为原始值, (a)-(d) 为 EMA 0.8。

## 后续计划

- [x] 图 4.1 训练指标 (四算法对比, 2x3)
- [x] 三张既有图统一 nature 风格
- [ ] 论文 caption 数值需按真实数据修订 (reward≈99 / jump≈2.2 / std≈0.8; 论文 11.7 → vertical_thrust)
- [ ] 图 3.1 框架概览图 (用户本次未选, 按需另行出图)
