# flat_walk_ppo_v2

> 路径: `backup/XqRobotV2WalkFlat/flat_walk_ppo_v2/`

XqRobotV2 平坦地面 PPO 行走 v2 — 速度控制 + 100Hz + 轮子对称

## v1 → v2 变更

| 改动 | v1 | v2 |
|------|-----|-----|
| 轮子控制 | 位置控制 (kp=3) | **速度控制 (kv=1)** |
| 控制频率 | 50Hz (ctrl_dt=0.02) | **100Hz (ctrl_dt=0.01)** |
| wheel_action_rate | -0.015 | **-0.005** |
| 新增奖励 | — | **wheel_symmetry: -0.5** |
| xqrobotV2.xml | 位置 actuator | **速度 actuator (wheel)** |

## 备份内容

| 文件 | 说明 |
|------|------|
| `conf/ppo/task/xqrobotV2_walk_flat/mujoco.yaml` | 训练配置 |
| `src/unilab/envs/locomotion/xqrobotV2/` | 环境代码 |
| `xqrobotV2.xml` | MuJoCo 模型（速度控制轮子） |
| `scene_flat.xml` | 场景 + keyframe |
| `shell/` | 训练/评估/tb 脚本 |
| `go.sh` | 一键启动键盘评估 |
| `restore.sh` | 一键恢复到项目目录 |

## 训练中指标 (iter 3640/5000)

| 指标 | 值 |
|------|-----|
| lin_vel | 1.10 / 1.5 |
| ang_vel | 1.42 / 1.5 |
| ep_length | 1852 / 2000 |
| entropy | 2.60 (仍在探索) |
