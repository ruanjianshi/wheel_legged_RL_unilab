# thesis — 研究生大论文开发指导中心

> **论文题目**: 基于强化学习的两轮足机器人多步态运动控制与自适应调度方法研究
> **英文**: Research on Multi-Gait Motion Control and Adaptive Scheduling Method for Two-Wheeled Legged Robot Based on Reinforcement Learning
> **配套代码库**: `wheel_legged_RL_unilab` (UniLab 异构强化学习训练框架, 机器人 xqrobotwl / xqrobotV2)

本目录是论文开发的**指导中心**: 承载架构设计、8 个步态专家、统一调度器、整体验证与开发文档, 随研究进展同步更新。

## 核心架构 (三要素 × 两阶段)

**三要素**: 🗺️ 多地形 × 🦵 多步态(8种) × 🧮 多算法 → 见 [architecture/](architecture/README.md)

```
分训-整合两阶段范式:
  第1阶段 · 分训: 8 个步态专家独立训练 (参数解耦)
  第2阶段 · 整合: 学习式统一调度器 按地形自动切换/混合步态
  最终:        混合地形环境完整遍历
```

## 目录导航

| 路径 | 内容 |
|------|------|
| [论文框架与实施计划.md](论文框架与实施计划.md) | 📌 **总纲**: 九章结构、核心问题、8专家、调度器、实现步骤、创新点、指标体系 |
| [architecture/](architecture/README.md) | 🖼️ 架构图: 原始绘制 (png/xmind) + Mermaid 重绘 + 跳跃 2×2 对比 |
| [experts/](experts/README.md) | 🦵 第1阶段分训: 8 个步态专家开发文档 (含训练记录/验收) |
| [scheduler/](scheduler/README.md) | 🎛️ 第2阶段整合: 学习式统一调度器 (第7章核心创新, 未开发) |
| [integration/](integration/README.md) | 🧪 整体验证: 混合地形遍历 + 消融 + 基线对比 (第8章, 未开发) |
| [progress/](progress/README.md) | 📊 开发进度追踪: 8专家+调度器+阶段 状态总表 |
| [templates/](templates/template_dev_doc.md) | 📝 开发文档模板 |

## 当前进度一览 (2026-08-10)

- ✅ **已训 (5/8 专家)**: 滚动行走 · 粗糙地形行走 · 抬腿行走 · 跳跃(四算法) · 抬腿上台阶
- 🚧 **训练测试中 (3/8)**: 后空翻 · 单腿平衡(含 move/unicycle) · 跌倒恢复(CPO) —— 2026-08-10 从 `fall_recovery` 分支移植代码
- ⏳ **未开发**: 学习式统一调度器 · 系统集成验证
- ⚠️ **待处理**: 跳跃专家纯 PPO 疑似退化 (jump_height→0)

## 使用约定

1. **架构图**在 `architecture/`, 源文件 `.mmd` (GitHub/mermaid.live 预览), 论文用图导出 PNG/SVG。
2. **专家文档**在 `experts/`, 每个专家一份, 用 `templates/template_dev_doc.md` 模板维护, 训练达标后更新。
3. **进度**统一在 `progress/README.md` 追踪; 逐日开发细节在 `_devlog/`。
4. 与本仓库其他目录的关系: `_devlog/` 逐日日志 · `shell/xqrobotwl/` 训练/验证脚本 · `logs/` 训练产出。
