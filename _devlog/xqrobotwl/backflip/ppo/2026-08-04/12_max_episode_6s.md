# 12 — max_episode_seconds 调整到 10s (1000 步)

## 日期

2026-08-04

## 来源

用户要求 max episode length = 1000, 与其他配置一致 (10s / 0.01s = 1000 步)。最初 4s (400 步) 太短。

## 解决方案

`conf/ppo/task/xqrobotwl_backflip_flat/mujoco.yaml`: `max_episode_seconds: 4.0 → 10.0` → 最大 ep_len 400→1000。

翻转 ~1.7s, 10s 留 ~8s 观察落地后不倒。与其他配置 (jump_srl 用 10s) 一致。

## 修改文件

| 文件 | 内容 |
|------|------|
| `conf/ppo/task/xqrobotwl_backflip_flat/mujoco.yaml` | env.max_episode_seconds 4→10 |

## 关联日志

- [11_not_fallen_not_standstraight](2026-08-04/11_not_fallen_not_standstraight.md) — "不倒"需求
