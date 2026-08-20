# Toe Walk MODE v6.1 — 双模式点足行走 (站立 ⇄ 点足抬腿) 备份

## 任务
XqRobotWL 双模式: 默认站立 (mode=0) → H 键切换点足抬腿 (mode=1) → 指令追踪 (前进/后退/侧移/转向) → 切回站立。

## 关键指标 (2026-08-18 确定性评估, model_9999)

| 项 | 值 | 判定 |
|----|----|------|
| 抬腿交替 | **L=12.0 / R=10.5 次** (双腿都抬) | ✅ PASS (历史首次) |
| 切换稳定 | 切换后 0.5s 不跌倒 | ✅ PASS |
| 站立微动 | linvel_xy 0.375 m/s (>0.2) / gyro 1.07 | ⚠️ 待精修 |
| 追踪 | fwd rmse 0.237, 侧移/转向有响应 | ⚠️ 中 |
| 训练 | 1024 envs × 10000 iter, run 2026-08-18_15-16-18_mujoco | completed |

## 机器人 / 算法
- xqrobotwl (8DOF), 相位时钟 0.7s, action_scale 0.18, Kp=60
- PPO + mode 命令通道 (5D: vx,vy,vyaw,tsk,mode) + 奖励 mode 门控
- **v6.1 核心机制**: 密集离地奖 (phase_swing_lift 30) + 窗级交替考核 (window_penalty 500/窗, 未离地窗结算时罚) + knee_lift 窗内离地门控 (防刷分) + 模式课程 [1800,5000]

## 运行 (开箱即跑)

```bash
PROJ=/home/robot/xiaoq/wheel_legged_RL_unilab
VER=toe_walk_mode_v6.1
B=backup/XqRobotWLToeWalkMode/$VER

# 拷贝配置/代码/脚本 (如当前工作区被改动)
cp $B/conf/* ${PROJ}/conf/ppo/task/xqrobotwl_toe_walk_mode/
cp $B/src/* ${PROJ}/src/unilab/envs/locomotion/xqrobotwl/
cp $B/shell/* ${PROJ}/shell/xqrobotwl/toe_walk_mode/

# 模型到 logs (评估用)
RUN=2026-08-18_15-16-18_mujoco
mkdir -p ${PROJ}/logs/rsl_rl_ppo/XqRobotWLToeWalkMode/$RUN
cp $B/model_9999.pt ${PROJ}/logs/rsl_rl_ppo/XqRobotWLToeWalkMode/$RUN/

# 确定性评估 (站立→抬腿→追踪→站立 序列)
uv run python tools/xqrobotwl/verify_toe_walk_mode.py --run $RUN --ckpt model_9999.pt

# 交互演示 (H 键切换站立/抬腿, ↑↓←→ 指令)
bash ${PROJ}/shell/xqrobotwl/toe_walk_mode/eval_ppo_toe_walk_mode.sh --keyboard
```

演示视频: `2026-08-18_v6.1_双模式点足演示.mp4` (本备份内)

## 已知遗留
- 站立微动超标 (0.375 m/s): 精修方向 stand_still 权重提升 / Stage3 站立占比提升 / 站立期轮速罚
- 对称量化 (膝弯比) 待补测