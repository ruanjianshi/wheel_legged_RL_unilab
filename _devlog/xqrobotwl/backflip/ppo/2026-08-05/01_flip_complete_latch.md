# 01 — flip_complete 改锁存 (旋转完成 → 站起来给奖励)

## 日期

2026-08-05

## 来源

存活率低 (ep_len 166), flip_complete 一直 0。根因: 奖励要求"旋转完成"和"站高 z>0.25"同时发生, 但机器人转完时在低处(z=0.13), 转完才站起来 → 永不触发。

## 解决方案

1. 加 `_flip_completed` 锁存: flip_progress >= 5.98 时锁存 True
2. flip_complete 奖励: `flip_completed & up>0.6 & z>0.25` — 转完后的任意时刻站起来即触发
3. 单次: 触发后重置锁存

## 修改文件

| 文件 | 内容 |
|------|------|
| `src/unilab/envs/locomotion/xqrobotwl/backflip.py` | 加 _flip_completed 锁存, flip_complete 改锁存逻辑, 单次重置 |

## 验证方法

用能翻转的模型测试: 锁存在 t=1.2 触发, 站起来后 flip_complete 奖励出现 (单步奖励 1.2 = flip_complete 1.0 + 其他)。以前 flip_complete 永远是 0。

## 后续计划

- resume 续训, 验证 flip_complete > 0 且存活率回升
- 翻转学成后再收紧终止 (分阶段 C 第二阶段)

## 关联日志

- [14_staged_relax_termination](2026-08-04/14_staged_relax_termination.md) — 分阶段方案C
