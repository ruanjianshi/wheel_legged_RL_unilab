# 02 — flip_progress 测量修复: 世界系 Y 角速度 (根因)

## 日期

2026-08-05

## 来源

存活率持续低, flip_complete 永远 0。深度排查发现**测量方法错误**是根因。

## 问题描述

- 机器人实际翻转 (纯 ff up-向量测 107%, 翻转模型测 122%)
- 但 env 的机身系 gyro 积分测量显示 **-0.57 rad (乱)** → flip_complete 永远不触发
- 机身系 pitch 角速度在翻转带横滚/偏航时严重失真 (同一 run: body-pitch +207% vs up-vector +6%)

## 根因分析

机身系 gyro[1] 积分假设翻转是纯俯仰旋转。但机器人翻转时带横滚 (hip_roll 外展), 机身系 pitch 轴随身体旋转, 积分失真。

对比 (同一 run):
| 测量 | 结果 |
|------|------|
| body-pitch (机身系) | +207% (乱) |
| up-vector XZ (世界系近似) | +6% (乱) |
| world Y 角速度 (P1方法) | **122% (正确)** |

## 解决方案

`_update_flip_progress` 改用 `get_base_ang_vel()[:,1]` (世界系 Y 角速度, P1 方法): `flip_progress = -∫ω_y dt, 后翻为正`。

## 修改文件

| 文件 | 内容 |
|------|------|
| `src/unilab/envs/locomotion/xqrobotwl/backflip.py` | flip_progress 改世界系 Y 角速度积分 |

## 验证方法

翻转模型测试: flip_progress 122%, 锁存触发, flip_complete 奖励出现 (1.19/步)。修复前 flip_complete 永远 0。

## 后续计划

- resume 续训: flip_complete 应从 0 出现, 存活率回升
- 这是测量根因, 修复后训练应能学成"翻转+落地"

## 关联日志

- [01_flip_complete_latch](2026-08-05/01_flip_complete_latch.md) — 锁存机制
