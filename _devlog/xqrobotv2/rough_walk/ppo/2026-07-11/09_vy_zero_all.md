# 09 三个训练任务 Vy 归零

**日期**: 2026-07-11
**来源**: 用户反馈不需要 Vy 横向移动
**关联**: [08_fix_spin_decouple](2026-07-10/08_fix_spin_decouple.md)

---

## 问题描述

rough_walk, stairs, jump 三个训练不需要 Vy（横向移动）能力，当前 vel_limit 仍含 Vy 范围。

## 解决方案

三个 config 的 `vel_limit` 中 Vy 归零：

| 任务 | 旧 Vy | 新 Vy |
|------|------|:--:|
| rough_walk | [-0.5, 0.5] | [0, 0] |
| stairs (np3o) | [-0.5, 0.5] | [0, 0] |
| jump | [-0.1, 0.1] | [0, 0] |

配合 #08 的命令解耦修复，Vy=0 时解耦固定 axis=0，不会产生零速度命令。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `conf/ppo/task/xqrobotV2_walk_rough/mujoco.yaml:L52-53` | Vy [0,0] |
| `conf/np3o/task/xqrobotV2_stairs/mujoco.yaml:L52-53` | Vy [0,0] |
| `conf/ppo/task/xqrobotV2_jump_flat/mujoco.yaml:L52-53` | Vy [0,0] |

## 验证方法

训练正常启动，不再产生 Vy 非零命令。

---

*记录人: AI (opencode)*
