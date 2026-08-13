# 备份: 跌倒恢复 v8.2 交付 model_7000 (用户验证效果最佳)

> 用户验证 (2026-08-13): "跌倒恢复的效果最好, 恢复后的站立姿态很好, 左右腿基本不内收
> 或一前一后"。按 CLAUDE.md §6.1/6.2 备份, 开箱即跑。

## 交付 checkpoint

- **model_7000.pt** — run `2026-08-12_01-58-14_mujoco` (v8.2 交付, 通宵迭代甜点位)
- 恢复率 (确定性 20ep 无辅助): **0/65/75/90** (仰卧/俯卧/左躺/右躺)
- 最长站立: **5.6-7.2s** (远超 §7.8 ≥1.0s)
- gyro: **0.14-0.79** (<1) · yaw: **13-42°** (≈walk 56°) · 漂移: 0.38-0.56m
- 恢复后姿态正常 (§1.3): 站立高度≈0.52m, 左右腿对称, 不内收/不一前一后 (用户验证)

## 评估数据 (来源 devlog #24)

| pose | 恢复率 | 最长站立 | yaw 转圈 | 漂移 | gyro |
|---|---|---|---|---|---|
| 仰卧 | 0% | 0s | — | 0.07m | — |
| 俯卧 | 65% | 7.15s | 0.23rad | 0.43m | 0.14 |
| 左躺 | 75% | 5.59s | 0.48rad | 0.38m | 0.79 |
| 右躺 | 90% | 6.34s | 0.73rad | 0.56m | 0.59 |

**已知局限**: 仰卧恢复率 0% (180° 翻转最难, 后续 v8.10 已突破到 30-40%, 见主工作区)。

## 文件说明

| 文件 | 内容 |
|---|---|
| `model_7000.pt` | 交付 checkpoint (CPO actor, 与 PPO 兼容的 actor_state_dict) |
| `src/.../fall_recovery.py` | **v8.2 训练时代码** (git commit a08b9b6 + 01:58 run 的 dirty diff, 含 no_yaw 死区) |
| `conf/.../mujoco.yaml` | v8.2 训练配置 (no_yaw 死区 1.0 / stand_anchor 15 / 力引导等) |
| `run_config.json` | 训练 run 配置快照 |
| `shell/.../eval_ppo_fall_recovery.sh` | 评估/回放脚本 (提交版) |
| `shell/.../train_ppo_fall_recovery.sh` | 训练脚本 (提交版) |
| `git_commit.txt` | git 版本 a08b9b6 |
| `git/wheel_legged_RL_unilab.diff` | 训练时点的未提交改动快照 (v8.2 死区等) |

## 开箱即跑 (加载并评估)

本备份的代码与主工作区 v8.10 代码不同 (v8.2 奖励结构)。**评估 model_7000 行为
用主工作区当前代码即可 (obs/动力学未变, 行为一致)**; 若要精确复现 v8.2 奖励环境:

```bash
# 方式 A: 用主工作区评估 (行为一致, 推荐)
cd /home/robot/xiaoq/wheel_legged_RL_unilab
uv run python _devlog/assess/runner.py -t fall_recovery \
    -r 2026-08-12_01-58-14_mujoco -c model_7000.pt --pose 1 --num_envs 20

# 方式 B: 精确复现 v8.2 环境 (交换代码后评估, 用完恢复)
cp src/unilab/envs/locomotion/xqrobotwl/fall_recovery.py /tmp/fr_current.py
cp backup/XqRobotWLFallRecoveryFlat/fall_recovery_v82_delivery_model7000/src/unilab/envs/locomotion/xqrobotwl/fall_recovery.py \
   src/unilab/envs/locomotion/xqrobotwl/fall_recovery.py
uv run python _devlog/assess/runner.py -t fall_recovery \
    -r 2026-08-12_01-58-14_mujoco -c model_7000.pt --pose 1 --num_envs 20
cp /tmp/fr_current.py src/unilab/envs/locomotion/xqrobotwl/fall_recovery.py   # 恢复

# 交互回放 (键盘)
bash shell/xqrobotwl/fall_recovery/eval_ppo_fall_recovery.sh 2026-08-12_01-58-14_mujoco --keyboard
```

## 回滚/恢复说明

- 若要回到此版本训练: 用 `conf/` + `src/fall_recovery.py` + `shell/train_*.sh` 重新训练
  (力引导 force_end 3000, 训练到 ~7000 取甜点位, 过训会发散如 model_7999)
- 若主工作区代码回退: 用 `src/fall_recovery.py` (v8.2) + `conf/` (v8.2) 替换即可
