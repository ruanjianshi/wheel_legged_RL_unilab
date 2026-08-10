# docs/timeline — 开发进展时间线

> 规范 CLAUDE.md §2.5: 每个任务先制定开发框架, 逐步开发, 框架实时更新;
> **开发进展时间线文档 (实时更新)**: 当前开发位置 / 每个开发部分带来的影响和效果 / 历史阶段记录。

## 开发框架 (任务启动时制定, 给负责人审阅)

§2.5 把"开发框架"与"进展时间线"并列为两个交付物:
- **开发框架** — 任务**启动前**制定, 一份给负责人审阅的实施方案
- **进展时间线** — 开发过程中**实时更新**的历史记录 (本目录的 `<task>.md`)

新任务启动时, 先把以下框架写入 `<task>.md` 顶部, 再开始开发:

```markdown
## 开发框架 (启动时给负责人审阅)

- **目标**: 最终要达成的效果 (对应 §7 哪项任务 / 达标标准)
- **约束**: 机械限制 / 电机限制 / 不允许的行为 (如辅助力)
- **方案**: 拟采用的算法 / 奖励结构 / 参考来源 (docs/references/)
- **分阶段计划**: 起步 → 可行性 → v1 训练 → 调参 → 达标 → 交付, 每阶段预计产出
- **风险**: 预判的技术难点与备选方案
```

> 已有任务的 目标/约束/方案 见 `thesis/experts/` 与 `thesis/论文框架与实施计划.md`;
> 本时间线记录"开发进行到哪、每个改动带来什么效果"。

## 文件

| 文件 | 任务 | 来源 devlog |
|------|------|------|
| [walk_flat.md](walk_flat.md) | 平地滚动行走 | `_devlog/xqrobotwl/walk_flat/ppo/` |
| [walk_rough.md](walk_rough.md) | 粗糙地形行走 | `_devlog/xqrobotwl/{walk_rough,rough}/ppo/` |
| [jump.md](jump.md) | 跳跃 (flat/srl/vmc/srl_vmc) | `_devlog/xqrobotwl/jump/ppo/` |
| [stairs.md](stairs.md) | 上下楼梯 | `_devlog/xqrobotwl/stairs/np3o/` |
| [toe_walk.md](toe_walk.md) | 点足/抬腿行走 | `_devlog/xqrobotwl/toe_walk/ppo/` |
| [backflip.md](backflip.md) | 后空翻 | `_devlog/xqrobotwl/backflip/ppo/` |
| [fall_recovery.md](fall_recovery.md) | 跌倒恢复 | `_devlog/xqrobotwl/fall_recovery/ppo/` |
| [single_leg.md](single_leg.md) | 单腿平衡 (base/move/unicycle) | `_devlog/xqrobotwl/single_leg/ppo/` |

仓库/工程管理类进展见 `_devlog/xqrobotwl/repo/`(任务无关)。

## 每份时间线的结构

```markdown
# 时间线 · <任务名> (<conf 任务名>)

> 一句话: 任务是什么 / 当前状态 (训练中 / 已达标 / 已交付 / 待完善)。
> 来源: _devlog/xqrobotwl/<task>/<algo>/  (实时更新, 每完成一个阶段追加一行)

| 日期 | 阶段 | 做了什么 | 影响/效果 | 问题与解决 |
|---|---|---|---|---|
| 2026-08-07 | 方案 | 制定 xxx 框架 | — | — |

## 当前状态
- 最新 checkpoint / 关键指标 / 达标情况

## 下一步
- [ ] 待办项
```

## 表格填写约定

- **日期**: 该阶段的日期或起止区间 (如 `2026-08-06 ~ 08-07`)
- **阶段**: 阶段名 (起步 / 方案 / v1 训练 / 调参 / 对比 / 论文图 / 交付 …)
- **做了什么**: 具体改动 (参数、结构、脚本), 可引用 devlog 序号 `[[NN_slug]]`
- **影响/效果**: 有据数据优先 (reward、指标、训练曲线), 无则写 "—"
- **问题与解决**: 踩到的坑与处理, 无则写 "—"

## 更新规则

- 每次开发推进 → 在本文件追加一行, **不重写历史行**
- 任务状态变化 → 更新顶部一句话 + 当前状态段
