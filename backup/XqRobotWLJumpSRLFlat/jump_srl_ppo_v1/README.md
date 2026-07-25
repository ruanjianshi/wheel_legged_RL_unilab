# Jump SRL PPO v1 — 最终版本

## 任务

XqRobotWL Wheeled-SRL 跳跃训练（SLIP FSM 观测 + PPO 策略）

## 关键指标

| 指标 | 值 (iter 9999) |
|------|----------------|
| reward | 97.05 |
| ep_len | 920 / 1000 |
| action_std | 0.77 |
| jump_height | 1.45 / 15 |
| lean_forward | -0.25（髋后仰 5°，trigger-gated） |
| base_height_err | 0.20m |
| vertical_thrust | 12.0 / 30 |

## 机器人

xqrobotwl (8DOF), Kp=60, Kv=1

## 算法

PPO + SLIP FSM 观测（6 状态 × 相位 × 计时器 = 315D 观测）
FSM 前馈 gain=0.15

## 运行

训练日期: 2026-07-24
运行 ID: 2026-07-24_23-31-38_mujoco
代码 commit: 见 git_commit.txt

## 恢复训练

```bash
PROJ=/home/robot/xiaoq/wheel_legged_RL_unilab
VER=jump_srl_ppo_v1

# 拷贝配置、代码、脚本
cp -r backup/XqRobotWLJumpSRLFlat/${VER}/conf/* ${PROJ}/conf/
cp -r backup/XqRobotWLJumpSRLFlat/${VER}/src/* ${PROJ}/src/
cp -r backup/XqRobotWLJumpSRLFlat/${VER}/shell/* ${PROJ}/shell/
cp backup/XqRobotWLJumpSRLFlat/${VER}/xqrobotwl.xml ${PROJ}/src/unilab/assets/robots/xqrobotwl/

# 拷贝模型到 logs
mkdir -p ${PROJ}/logs/rsl_rl_ppo/XqRobotWLJumpSRLFlat/2026-07-24_23-31-38_mujoco/
cp backup/XqRobotWLJumpSRLFlat/${VER}/model_9999.pt ${PROJ}/logs/rsl_rl_ppo/XqRobotWLJumpSRLFlat/2026-07-24_23-31-38_mujoco/

# 训练
CUDA_VISIBLE_DEVICES=0 bash ${PROJ}/shell/xqrobotwl/train_ppo_jump_srl.sh
```

## 验证播放

```bash
cd ${PROJ}
bash shell/xqrobotwl/eval_ppo_jump_srl_flat.sh 2026-07-24_23-31-38_mujoco --keyboard
```
