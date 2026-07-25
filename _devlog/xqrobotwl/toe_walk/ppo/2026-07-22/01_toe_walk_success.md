# 12 — 点足行走训练成功 + 参数调优

## 日期
2026-07-22

## 来源
创建 xqrobotwl toe_walk 配置并调优训练。

## 工作内容

### 1. 新建 toe_walk 配置
- `conf/ppo/task/xqrobotwl_toe_walk_flat/mujoco.yaml`
- `shell/xqrobotwl/train_ppo_toe_walk.sh`
- `shell/xqrobotwl/eval_ppo_toe_walk.sh`
- `base_height_target: 0.55`（与 xqrobotwl 物理上限一致）

### 2. 参数调优

| 参数 | 旧 | 新 | 原因 |
|------|----|----|------|
| cycle_time | 0.5s | **0.7s** | 步态太急促，减慢更稳 |
| ref_scale | 0.12 | **0.18** | 抬腿幅度不够，增大 50% |
| action_scale | 0.3 | **0.5** | 策略校正权不足 |
| thigh multiplier | ×0.5 | **×0.8** | 大腿前摆更有力 |
| max_iterations | 5000 | **10000** | 对标其他任务 |
| num_steps_per_env | 25 | **24** | 对标其他任务 |
| max_episode_seconds | 12.0 | **10.0** | 1000 步 |

### 3. 训练结果 (iter 1409/10000)

| 指标 | 值 | 评判 |
|------|----|------|
| ep_len | 1165 (97%) | ✅ 几乎满存活 |
| action_std | 0.15 | ✅ 完美收敛 |
| **swing_lift** | **2.39** | ✅ 轮子确实离地 |
| wheel_balance | 0.99 | ✅ 轮子用于平衡 |
| ref_tracking | 3.3 | ✅ 正弦轨迹跟踪 |

点足行走验证成功：
- 相位时钟驱动交替正弦轨迹
- 摆动相：膝弯收腿 + 轮子离地（swing_lift 奖励驱动）
- 支撑相：轮子平衡机身（wheel_balance 奖励驱动）
- 重心转移：hip_roll 偏置到支撑腿

## 设计原理
```
相位时钟 sin/cos → 交替步态
  swing phase: sin > 0.4 → 膝弯×5, 大腿前摆×0.8, swing_lift 强制离地
  stance phase: 轮子低速保持平衡, hip_roll 重心转移
  L/R 交替: sin/-sin = 180° 相位差
```

## 后续计划
- 训练到 iter 10000
- 评估步态稳定性
- 可扩展：加入速度命令使点足行走支持移动

## 关联日志
- `2026-07-22/10` — 高度目标修正 (0.65→0.55)
