# Toe Walk v14 — 单模式点足抬腿 (参考轨迹) 备份

## 任务
XqRobotWL 单模式点足抬腿 (`xqrobotwl_toe_walk_flat`) — **原地小踏步点足**: 一左一右交替、平缓小幅、机身高度正常。

## 关键指标 (2026-08-20 实测, model_9999)

| 项 | 值 | 判定 |
|----|-----|------|
| 参考跟踪 | 42.7/20 (正弦左右交替完美执行) | ✅ 一左一右 |
| 轮离地 | 摆动窗轮力 100N→**4~6N** (接近完全离地) | ✅ 点足观感 (老板可视化确认) |
| 幅度 | 膝摆参考 0.8 rad 完整执行 | ✅ 小幅平缓 (老板确认) |
| 高度 | z=0.485 (目标 0.52, 轻低) | ⚠️ 可接受 |
| 存活 | ep_len 1000 / reward 403 | ✅ 极稳 |
| 训练 | 1024 envs × 10000 iter, run `2026-08-20_16-22-25_mujoco` | completed |

## 技术方案 (v14)
**参考轨迹模式** (use_reference=true): 相位时钟正弦参考直接定义左右交替步态, 策略学习跟踪+平衡修正。
- ref_scale 0.20 (膝弯摆幅 0.8 rad), cycle_time 0.7s
- 奖励: ref_tracking 20 + swing_lift 10 + base_height -150 (目标0.55) + orientation -30 + 平滑
- 终止放宽: thigh ±0.9 + 异常帧延迟 5 (探索不误杀)
- tracking 0 (原地踏步, 不做行走)

## 运行 (开箱即跑)

```bash
PROJ=/home/robot/xiaoq/wheel_legged_RL_unilab
B=backup/XqRobotWLToeWalkFlat/toe_walk_v14
RUN=2026-08-20_16-22-25_mujoco

# 恢复备份 (如工作区被改动)
cp $B/conf/* ${PROJ}/conf/ppo/task/xqrobotwl_toe_walk_flat/
cp $B/src/* ${PROJ}/src/unilab/envs/locomotion/xqrobotwl/
cp $B/shell/* ${PROJ}/shell/xqrobotwl/toe_walk/

# 模型到 logs
mkdir -p ${PROJ}/logs/rsl_rl_ppo/XqRobotWLToeWalkFlat/$RUN
cp $B/model_9999.pt ${PROJ}/logs/rsl_rl_ppo/XqRobotWLToeWalkFlat/$RUN/

# 交互演示
bash ${PROJ}/shell/xqrobotwl/toe_walk/eval_ppo_toe_walk.sh --keyboard
```

演示视频: `2026-08-20_v14_点足演示_录屏.mp4` (本备份内, 交互窗口实录)

## 开发历程 (简)
v1(相位门控, 单侧偏) → v10-v12(窗级罚/预热/对称, 崩或慢) → **v13/v14 参考轨迹 (交替机制成立, 交付)**。全天 devlog 见 `_devlog/xqrobotwl/toe_walk/ppo/2026-08-18~20/`。

## 已知残留
- 轮"基本离地"但未完全力=0 (4-6N); 如需完全离地 → thigh 前摆加强方案 (备用)
- 高度 0.485 略低于 0.52; 如需 → base_height_target 0.55→0.57