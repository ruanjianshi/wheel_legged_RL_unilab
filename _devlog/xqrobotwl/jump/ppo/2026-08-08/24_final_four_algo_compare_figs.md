# [24] 最终四算法对比 + 论文图 (10000 轮, 修复后配置)

**日期**: 2026-08-08
**来源**: 训练完成, 用户要求最终对比 + 出图
**关联**: [[23_verify_jump_config_drift]]

---

## 问题描述

三个重训 (PPO+VMC / VMC+SRL / 纯PPO v5) 全部跑到 10000 iter 完成。需要:
1. 用修复后的 verify (见 [[23]]) 跑最终四算法对比
2. 更新 four_algo_comparison.json (旧数据被配置漂移污染)
3. 重出论文图 (2x2 训练 + 2x1 验证)

## 训练完成情况

| 算法 | run | 完成 iter | mean_reward |
|------|-----|----------|------------|
| 纯PPO v5 | 2026-08-07_22-27-29 | 9999 | 147.9 |
| PPO+VMC | 2026-08-07_22-21-54 | 9999 | 88.4 |
| VMC+SRL | 2026-08-07_22-21-55 | 9999 | 86.8 |
| SRL | 2026-08-06_01-16-20 | 9999 (已有) | - |

## 修复内容

`scripts/compare_jump.py` 同 [[23]] 的问题:
- env 构建改用 `trained_env_overrides()` (从 run_config.json 读训练配置)
- standing_z 改用"触发关闭 + 轮接地"中位数 (非 step-0 hover)

## 最终四算法对比 (修复后配置)

| 算法 | 跳跃高度 | air | 存活率 | 跳跃成功率 |
|------|---------|-----|--------|-----------|
| SRL | 0.552 | 0.221 | 1.00 | 1.00 |
| VMC+SRL | 0.251 | 0.107 | 0.80 | 1.00 |
| PPO+VMC | 0.271 | 0.051 | 0.95 | 0.85 |
| 纯PPO | 0.190 | 0.000 | 0.95 | 0.00 |

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `scripts/compare_jump.py` | trained_env_overrides + 真实 standing 测量 |
| `jump_management/results/four_algo_comparison.json` | 更新为修复后数据 (SRLVMC jump 0.015→0.251) |
| `jump_management/results/paper_fig_training.png/pdf` | 重生成 (2x2, SciencePlots 样式) |
| `jump_management/results/paper_fig_validation.png/pdf` | 重生成 (2x1) |

## 验证方法

- 四算法对比输出: 跳高/air/存活率/成功率
- 训练图 (2118x1887, 2x2): mean_reward / episode_length / vertical_thrust / jump_height
- 验证图 (1879x1876, 2x1): (a) 跳高柱状 (b) air+存活率
- 程序化验证颜色: 四种 Okabe-Ito 色全部存在于图中

## 评估结果

- SRL 最强 (0.55m, air 22%)
- VMC+SRL 第二 (0.25m, air 10.7% 达标, 成功率 100%)
- PPO+VMC 第三 (0.27m 但腾空短)
- 纯PPO 垫底 (0.19m, air 0%, jump_success 0%)
- 论文叙事: 参考轨迹 (SRL/SRL+VMC) 显著优于无参考纯学习

## 后续计划

- VMC+SRL 存活率 0.80 (高跳后偶发落地翻车) — 可后续优化落地
- 纯PPO jump_success 0% — 作为 baseline 成立, 如需提升可调
