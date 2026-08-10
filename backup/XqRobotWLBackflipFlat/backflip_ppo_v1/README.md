# Backflip PPO v1 — 交付模型

## 任务

XqRobotWL 后空翻 (backflip)，独立训练，可完整翻越 360°。

## 关键指标

| 指标 | 值 (model_1000, 交付模型) |
|------|----------------|
| best_mean_reward | 5.66 |
| 完成迭代 | 1999 (全量训练) |
| 说明 | model_1000 可完整后空翻; model_1999 已发散, 不用 |

## 机器人

xqrobotwl (8DOF), Kp=60, Kv=1 (标准 XML)

## 算法

PPO (rsl_rl)，后空翻专用 FSM 前馈 + 轮匹配。

## 运行

训练日期: 2026-08-05
运行 ID: 2026-08-05_18-35-40_mujoco__detflip
代码 commit: 见 git_commit.txt

## 恢复 / 验证

```bash
# 拷贝配置与代码回项目根目录
VER=backflip_ppo_v1
cp -r backup/XqRobotWLBackflipFlat/$VER/{conf,src,shell,xqrobotwl.xml,scene_flat.xml} .

# 可视化验证 (MuJoCo 窗口, 指定交付 checkpoint)
bash shell/xqrobotwl/eval_ppo_backflip.sh <run_id> --keyboard algo.checkpoint=1000
```

## 文件

- `model_1000.pt` — 交付 checkpoint (策略权重, 需配合 policy.onnx 或重载)
- `run_config.json` / `run_summary.json` — 训练配置与摘要
- `conf/` `src/` `shell/` — 可恢复代码与配置
