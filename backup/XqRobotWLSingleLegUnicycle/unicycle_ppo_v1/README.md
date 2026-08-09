# Unicycle PPO v1 — 单轮平衡里程碑

## 任务

XqRobotWL 改进结构单轮平衡 (横躺独轮车式)。**完整 8 秒单轮平衡，PD 基线 0.86s 的 9.3 倍**。

## 关键指标 (model_4000, 交付模型)

| 指标 | 值 |
|------|----------------|
| 平均平衡时间 | **800/800 步满 8s** (20 env 全跑满) |
| 左轮贴地率 | 100% |
| pitch 对齐 (横躺) | 0.995 |
| roll 对齐 (横躺) | 0.999 |
| final_mean_reward | 53.3 (best 72.7) |
| ⚠️ 注意 | model_4999 已退化 (1.7s), **部署用 model_3000/4000** |

## 结构 (改进, 关键)

- 机身横躺 up=[0,-1,0], **左腿近直支撑** L=(1.49,-0.1,-0.2), CoM 偏移 0.5cm
- **右腿伸直配重** R=(-1.50,0,0), CoM 权限 ±5cm
- 执行器 (env init 运行时设置): **腿 kp=300 kv=10**, **轮=扭矩源**
- RL 控 [配重 R_hip_roll/pitch, 轮 L_wheel], 支撑腿钉住
- 奖励: upright + wheel_down + 位置drift惩罚 (v3; v1无惩罚2.75s, v2线速度反有害)

## XML 传感器 (本备份含)

- `basexvector` (framexaxis): 横躺下 pitch 观测 (up向量看不见 pitch, 必需)
- `left_wheel_world_pos` (framepos): 左轮离地检测

## 运行

训练日期: 2026-08-06
运行 ID: 2026-08-06_17-34-25_mujoco
代码 commit: 见 git_commit.txt

## 恢复 / 验证

```bash
# 拷贝配置与代码回项目根目录
VER=unicycle_ppo_v1
cp -r backup/XqRobotWLSingleLegUnicycle/$VER/{conf,src,shell,xqrobotwl.xml,scene_flat.xml} .

# 可视化验证 (MuJoCo 窗口, 默认最佳 model_4000)
bash shell/xqrobotwl/eval_ppo_single_leg_unicycle.sh

# 数值指标
uv run mjpython scripts/xqrobotwl/eval_single_leg_unicycle.py \
  --ckpt backup/XqRobotWLSingleLegUnicycle/$VER/model_4000.pt --episodes 10
```

## 文件

- `model_4000.pt` — 交付 checkpoint (8s 单轮平衡)
- `run_config.json` / `run_summary.json` — 训练配置与摘要
- `conf/` `src/` `shell/` — 可恢复代码与配置
