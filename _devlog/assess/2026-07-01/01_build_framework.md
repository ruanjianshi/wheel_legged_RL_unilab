# 01 构建评估框架 v1.0

**日期**: 2026-07-01  
**来源**: 开发需求（需要标准化评估流程）

---

## 问题描述

缺乏系统化的策略评估工具。每次需要手动写脚本测试模型，无法跨版本/跨算法对比。

## 解决方案

构建 `assess/` 框架，支持多任务 × 多算法 × 多 session 的自动化评估。

### 架构

```
assess/
├── tasks.py         # TaskDef + AlgoDef + TaskAlgoPair 注册表
├── runner.py        # CLI: -t <task> -a <algo> -r <run> -c <ckpt>
├── metrics.py       # 22 指标（5 类：跟踪/稳定/运动/能效/步态）
├── scenarios.py     # 4 套场景（decoupling/full/standing/toe_walk）
├── recorder.py      # 全轨迹时间序列录制
├── plotter.py       # 6 种图表（velocity/stability/gait/bars/radar/comparison）
├── exporter.py      # CSV 导出 + JSON 累积数据库
├── reporter.py      # 自动生成 Markdown 分析报告
├── results/         # <task>/<algo>/<session>/{metrics,track}.json|csv
├── plots/           # <task>/<algo>/<session>/<chart>.png
└── reports/         # <task>/<algo>/<session>/analysis.md
```

### 关键决策

- 独立于 Hydra，不依赖训练配置
- 三级分类：`task/algo/session`
- 指标分类参考论文：RMA, ANYmal, Cassie, IsaacGym, DreamWaQ
- `--record` 录制全轨迹（供深度分析）
- `--trend` 支持跨 checkpoint 趋势
- `--cmp` 支持跨算法对比（雷达图 + 柱状图）

## 已注册内容

- 4 算法: ppo, sac, appo, td3
- 2 任务+算法: flat_walk/ppo, toe_walk/ppo

## 文档

- [README.md](../../assess/README.md) — 完整说明
- [AGENTS.md](../../../AGENTS.md) — 已更新加入评估相关指引
- [_devlog/README.md](../README.md) — 开发日志规范

---

*记录人: AI (opencode)*
