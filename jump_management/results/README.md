# jump_management/results — 目录说明

顶层只保留**当前论文图与数据**(脚本直接引用,勿移动)。

## 顶层 (当前, 脚本引用)

| 文件 | 说明 | 生成脚本 |
|------|------|----------|
| `paper_fig_training.{png,pdf}` | 训练对比 2x2 (EMA 0.8 平滑) | `scripts/make_paper_figures.py` |
| `paper_fig_training_metrics.{png,pdf}` | **图 4.1** 训练指标 2x3 (a-f) | 同上 |
| `paper_fig_validation.{png,pdf}` | 腾空/存活柱状 | 同上 |
| `paper_fig_trajectory.{png,pdf}` | 高度曲线 2x2 + FSM 相位色带 | `scripts/plot_jump_trajectory.py` |
| `paper_fig_jump_joints.{png,pdf}` | 关节角 2x2 | 同上 |
| `four_algo_comparison.json` | 四算法评估汇总 (跳高/腾空/存活/checkpoint) | compare_jump 流程 |
| `jump_traj_{srl,ppo,vmc,srlvmc}.npz` | 各算法最优 checkpoint 单跳轨迹 (t/z/hip/knee/phase/terminated) | `scripts/verify_jump_trajectory.py` |

## legacy/ (2026-07 中间产物, 脚本不再引用)

| 子目录 | 内容 |
|--------|------|
| `paper_figs/` | 旧版论文图 (paper_fig_jump_height/phase/rewards/summary/vx_tracking + 旧 training/validation 的 svg/tiff) |
| `fig_analysis/` | 07-26 分析批 fig1–fig15 + 独立分析图 (phase_space/reward_comparison 等) |
| `srl_sequences/` | 07-27 SRL 跳序列图 + `jump_frames*` 帧目录 |
| `ablation/` | 07-28 消融旧批 (metrics/success/table/training_curves) |
| `eval_data/` | 旧评估 JSON (eval_data/scenario_eval*/vx_tracking)、comparison_report.md、被取代的 `jump_trajectories.npz` |

> 重跑某个旧图时,从 legacy 对应目录取回文件即可;脚本路径仍指向顶层。
