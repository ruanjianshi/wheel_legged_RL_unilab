# Toe Walk PPO v1 — 最终版本

## 任务

XqRobotWL 点足抬腿行走（相位门控，无参考轨迹）

## 关键指标

| 指标 | 值 (iter 9999) | 解读 |
|------|----------------|------|
| reward | 166.77 | 正向 |
| ep_len | 922 / 1000 | 存活 92% |
| action_std | 0.15 | 完全收敛 |
| **phase_swing_lift** | **14.34 / 30** | **轮子 48% 摆地时离地** |
| phase_knee_lift | 9.12 / 15 | 摆地弯膝 0.61 rad |
| phase_knee_stance | -1.02 / 5 | 支撑微弯 0.20 rad |
| swing_contact_penalty | -1.93 | 接触罚 (weight=5) |
| base_height_err | 0.16m | 高度稳定 |
| orientation | 0.96 / 30 | tilt ~0.03 rad |
| tracking_lin_vel | 1.58 | 速度跟踪 |

## 机器人

xqrobotwl (8DOF), Kp=60, Kv=1, action_scale=0.25

## 算法

PPO + 相位门控奖励（无参考轨迹）
13 项精简奖励（删 conflict terms: wheel_balance, wheel_symmetry, feet_distance, soft_landing）
max_tilt=90°, orientation=-30, ang_vel_xy=-0.3

## 运行

训练日期: 2026-07-25
运行 ID: 2026-07-24_23-40-42_mujoco
代码 commit: 见 git_commit.txt

## 恢复训练

```bash
PROJ=/home/robot/xiaoq/wheel_legged_RL_unilab
VER=toe_walk_ppo_v1

# 拷贝配置、代码、脚本
cp -r backup/XqRobotWLToeWalkFlat/${VER}/conf/* ${PROJ}/conf/
cp -r backup/XqRobotWLToeWalkFlat/${VER}/src/* ${PROJ}/src/
cp -r backup/XqRobotWLToeWalkFlat/${VER}/shell/* ${PROJ}/shell/
cp backup/XqRobotWLToeWalkFlat/${VER}/xqrobotwl.xml ${PROJ}/src/unilab/assets/robots/xqrobotwl/

# 拷贝模型到 logs
mkdir -p ${PROJ}/logs/rsl_rl_ppo/XqRobotWLToeWalkFlat/2026-07-24_23-40-42_mujoco/
cp backup/XqRobotWLToeWalkFlat/${VER}/model_9999.pt ${PROJ}/logs/rsl_rl_ppo/XqRobotWLToeWalkFlat/2026-07-24_23-40-42_mujoco/

# 训练
CUDA_VISIBLE_DEVICES=0 bash ${PROJ}/shell/xqrobotwl/train_ppo_toe_walk.sh
```

## 验证播放

```bash
cd ${PROJ}
bash shell/xqrobotwl/eval_ppo_toe_walk.sh 2026-07-24_23-40-42_mujoco
```
