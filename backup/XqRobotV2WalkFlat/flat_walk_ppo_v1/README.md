# flat_walk_ppo_v1

> 路径: `backup/XqRobotV2WalkFlat/flat_walk_ppo_v1/`

XqRobotV2 平坦地面 PPO 行走 - 基准版本 v1

## 训练结果

| 指标 | 值 |
|------|-----|
| 算法 | PPO (rsl-rl) |
| 训练轮数 | 5000 |
| 最终奖励 | 54.1 |
| 最佳奖励 | 66.4 |
| 存活步数 | 905 / 1000 |
| 线速度追踪 | ~1.03 / 1.5 |
| 角速度追踪 | ~1.37 / 1.5 |
| 训练时长 | ~2.2h (RTX 4090) |

## 备份内容

| 文件 | 说明 |
|------|------|
| `conf/ppo/task/xqrobotV2_walk_flat/mujoco.yaml` | 训练配置 |
| `src/unilab/envs/locomotion/xqrobotV2/base.py` | 环境基础（DEFAULT_LEG_ANGLES 等） |
| `src/unilab/envs/locomotion/xqrobotV2/joystick.py` | Flat Walk 环境实现 |
| `src/unilab/envs/locomotion/xqrobotV2/__init__.py` | 模块导入 |
| `scene_flat.xml` | 机器人场景 + keyframe |
| `shell/train_ppo_flat.sh` | 训练启动脚本 |
| `shell/eval_ppo_flat.sh` | 评估脚本 |
| `shell/tensorboard.sh` | TensorBoard 查看 |
| `run_config.json` | 完整 Hydra 运行时配置快照 |
| `run_summary.json` | 训练摘要 |
| `git_commit.txt` | 代码版本号 |
| `uncommitted.diff` | 未提交改动（如有） |

## 启动训练

> **备份本身不能直接启动训练**（缺少 UniLab 框架代码如 `scripts/training/train_rsl_rl.py` 等）。
> 需先恢复到项目目录，再启动。

### 一键恢复

```bash
bash backup/XqRobotV2WalkFlat/flat_walk_ppo_v1/restore.sh
```

### 手动恢复

```bash
cp -r backup/XqRobotV2WalkFlat/flat_walk_ppo_v1/{conf,src,shell,scene_flat.xml} .
```

### 启动训练

```bash
CUDA_VISIBLE_DEVICES=1 bash shell/train_ppo_flat.sh
```

## 关键配置

- **默认站姿**: `[0.1, 0.1, -0.1, 0.1, 0.1, -0.1]`
- **髋关节**: 双髋 0.1（必须外展，改为 0.0 会导致训练崩溃）
- **腿部对称**: `similar_calf` 覆盖髋+大腿+小腿，权重 -1.0
- **机身约束**: `orientation: -10.0`, `base_height: -5.0`
- **5D 命令**: `[vx, vy, vyaw, tsk, height_target]`
- **课程学习**: 对称扩展

## 已知局限

- 髋=0.0 会导致训练崩溃（XqRobotV2 结构需要外展支撑）
- Vx 正向偶尔轻微倒退漂移
- 腿长域随机化未开启
