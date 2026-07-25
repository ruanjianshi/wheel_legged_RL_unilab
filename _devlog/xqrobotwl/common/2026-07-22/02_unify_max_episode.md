# 02 — 统一 max_episode_seconds

## 日期
2026-07-22

## 来源
各任务最大步长不统一 (flat=2000, rough=2000, toe_walk=1200)。

## 修复
所有 xqrobotwl 任务 YAML 统一 `max_episode_seconds: 10.0` (1000 步)
Python 默认一并修正 `joystick.py: max_episode_seconds: 20.0 → 10.0`

| 任务 | 旧 | 新 |
|------|----|----|
| flat | 20.0 (default) | 10.0 |
| rough | 20.0 (default) | 10.0 |
| jump | 10.0 | 10.0 |
| SRL jump | 10.0 | 10.0 |
| toe_walk | 12.0 (default) → 10.0 | 10.0 |
| stairs | 20.0 (inherit) | 10.0 (inherit) |

## 修改文件
- `conf/ppo/task/xqrobotwl_walk_flat/mujoco.yaml`
- `conf/ppo/task/xqrobotwl_walk_rough/mujoco.yaml`
- `src/unilab/envs/locomotion/xqrobotwl/joystick.py`

## 关联日志
- common/01 — 平台检测
