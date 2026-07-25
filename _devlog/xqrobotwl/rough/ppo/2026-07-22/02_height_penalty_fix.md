# 02 — 高度惩罚 + 高度命令修复

## 日期
2026-07-22

## 来源
评估发现站立高度不对、Q/E 高度调节无用。

## 问题
1. `base_height: -5` 太弱 (平坦用 -60)
2. 模型对高度命令不敏感

## 修复
- `base_height`: -5 → -30
- `max_episode_seconds`: (default 20.0 → YAML 10.0)
- `base_height_target`: 保持 0.55

## 关联日志
- 01 — 粗糙地形增强 + 高度课程
